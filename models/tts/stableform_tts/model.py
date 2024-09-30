import math
import random
import torch
from torch import nn

from models.commons.layers import Embedding
from models.commons.nar_tts_modules import (
    EnergyPredictor,
    PitchPredictor,
)
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse
from utils.nn.seq_utils import sequence_mask

from .transformer import FastSpeechEncoder, FastSpeechDecoder
from .style_encoder import StyleEncoder
from .diffusion import Diffusion
from .utils import fix_len_compatibility
from .common import (
    get_mask_from_lengths,
    generate_path,
    average_to_ph,
    ConvAttention,
    LengthRegulator,
)
from .alignment import mas_width1 as mas
from .attribute_prediction_model import DAP


class StableFormTTS(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = hparams
        self.hidden_size = hparams["hidden_size"]
        self.n_feats = hparams["audio_num_mel_bins"]

        self.text_encoder = FastSpeechEncoder(
            dict_size,
            hparams["hidden_size"],
            hparams["encoder_layers"],
            hparams["encoder_ffn_kernel_size"],
            num_heads=hparams["num_heads"],
            saln_dim=hparams["spk_emb_dim"],
        )

        self.excitaion_generator = FastSpeechDecoder(
            hparams["hidden_size"],
            hparams["decoder_layers"],
            hparams["decoder_ffn_kernel_size"],
            hparams["num_heads"],
            saln_dim=hparams["spk_emb_dim"],
        )

        self.formant_generator = FastSpeechDecoder(
            hparams["hidden_size"],
            hparams["decoder_layers"],
            hparams["decoder_ffn_kernel_size"],
            hparams["num_heads"],
            saln_dim=hparams["spk_emb_dim"],
        )

        self.style_encoder = StyleEncoder(
            in_dim=out_dims, out_dim=hparams["spk_emb_dim"]
        )

        self.pitch_embed = Embedding(256, self.hidden_size, 0)
        self.energy_embed = Embedding(256, self.hidden_size, 0)
        self.energy_predictor = EnergyPredictor(
            self.hidden_size,
            n_chans=self.hidden_size,
            n_layers=hparams["predictor_layers"],
            dropout_rate=hparams["predictor_dropout"],
            odim=1,
            kernel_size=hparams["predictor_kernel"],
        )
        self.pitch_predictor = PitchPredictor(
            self.hidden_size,
            n_chans=self.hidden_size,
            n_layers=hparams["predictor_layers"],
            dropout_rate=hparams["predictor_dropout"],
            odim=1,
            kernel_size=hparams["predictor_kernel"],
        )

        self.diffusion = Diffusion(
            out_dims,
            hparams["dec_dim"],
            hparams["num_spk"],
            hparams["spk_emb_dim"],
            hparams["beta_min"],
            hparams["beta_max"],
            hparams["pe_scale"],
        )
        self.proj_excitation = nn.Linear(self.hidden_size, out_dims)
        self.proj_formant = nn.Linear(self.hidden_size, out_dims)

        self.dur_predictor = DAP(
            hparams["spk_emb_dim"],
            bottleneck_hparams=hparams["dur_model_config"]["bottleneck_hparams"],
            take_log_of_input=hparams["dur_model_config"]["take_log_of_input"],
            arch_hparams=hparams["dur_model_config"]["arch_hparams"],
        )

        self.attention = ConvAttention(
            self.n_feats, self.hidden_size + hparams["spk_emb_dim"]
        )
        self.length_regulator = LengthRegulator()

    def forward(
        self,
        x,
        x_length,
        y=None,
        y_length=None,
        attn_prior=None,
        f0=None,
        energy=None,
        infer=False,
        binarize_attention=False,
        temperature=1.5,
        n_timesteps=100,
        length_scale=1.0,
        stoc=False,
        out_size=128,
        sigma_dur=0.8,
        **kwargs,
    ):
        ret = {}

        ####################################################
        # Style vector
        ####################################################
        y_mask = sequence_mask(y_length).unsqueeze(1)
        s = self.style_encoder(y, y_mask)

        ####################################################
        # Text encoder
        ####################################################
        x = self.text_encoder(x, style_vector=s)  # [B, T, C]
        x_mask = sequence_mask(x_length)
        ret["x_mask"] = x_mask

        ####################################################
        # Aligner
        ####################################################
        if infer:
            # get token durations
            batch_size = x.shape[0]
            n_tokens = x.shape[1]
            z_dur = torch.FloatTensor(batch_size, 1, n_tokens).to(x.device)
            z_dur = z_dur.normal_() * sigma_dur
            s_expd = s[:, None, :].expand(-1, x.shape[1], -1)
            dur = self.dur_predictor.infer(z_dur, x.transpose(1, 2), s)
            if dur.shape[-1] < x.shape[1]:
                to_pad = x.shape[1] - dur.shape[2]
                pad_fn = nn.ReplicationPad1d((0, to_pad))
                dur = pad_fn(dur)
            dur = (dur + 0.5).floor().int() * length_scale
            dur = dur[:, 0]
            y_length = torch.clamp_min(torch.sum(dur.unsqueeze(1), [1, 2]), 1).long()
            y_max_length = int(y_length.max())
            y_max_length_ = fix_len_compatibility(y_max_length)
            y_mask = (
                sequence_mask(y_length, y_max_length_).unsqueeze(1).to(x_mask.dtype)
            )
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = (
                generate_path(dur, attn_mask.squeeze(1).long())
                .unsqueeze(1)
                .transpose(2, 3)
            ).float()

        else:
            attn = None
            attn_soft = None
            attn_hard = None
            attn_mask = get_mask_from_lengths(x_length)[..., None] == 0

            x_for_attn = x
            s_expd = s[:, None, :].expand(-1, x.shape[1], -1)
            x_for_attn = torch.cat((x_for_attn, s_expd.detach()), -1).transpose(1, 2)
            # attn_mask shld be 1 for unsd t-steps in text_enc_w_spkvec tensor
            attn_soft, attn_logprob = self.attention(
                y,
                x_for_attn,
                y_length,
                attn_mask,
                key_lens=x_length,
                attn_prior=attn_prior,
            )

            if binarize_attention:
                attn = self.binarize_attention(attn_soft, x_length, y_length)
                attn_hard = attn
                attn_hard = attn_soft + (attn_hard - attn_soft).detach()
            else:
                attn = attn_soft
            if attn_hard is None:
                attn_hard = self.binarize_attention(attn_soft, x_length, y_length)

            # Viterbi --> durations
            dur_tgt = attn_hard.sum(2)[:, 0, :]
            dur_output = self.dur_predictor(
                torch.detach(x.transpose(1, 2)),
                torch.detach(s),
                torch.detach(dur_tgt.float()),
                x_length,
            )
            ret["dur_output"] = dur_output
            ret["attn_soft"] = attn_soft
            ret["attn"] = attn
            ret["attn_logprob"] = attn_logprob

            f0 = average_to_ph(f0, dur_tgt)
            energy = average_to_ph(energy, dur_tgt)
            ret["f0_ph"] = f0
            ret["energy_ph"] = energy

        ####################################################
        # Prosody
        ####################################################
        # add pitch and energy embed
        pitch_embed = self.forward_pitch(x, f0, ret, infer, x_mask.long())

        # add pitch and energy embed
        energy_embed = self.forward_energy(x, energy, ret, infer)

        x_e_inp = x + pitch_embed + energy_embed
        x_e_inp = torch.bmm(
            x_e_inp.transpose(1, 2), attn.squeeze(1).transpose(1, 2)
        ).transpose(1, 2)
        x_f_inp = torch.bmm(
            x.transpose(1, 2), attn.squeeze(1).transpose(1, 2)
        ).transpose(1, 2)
        x_e = self.excitaion_generator(x_e_inp, style_vector=s)
        x_e = self.proj_excitation(x_e).transpose(1, 2)

        x_f = self.formant_generator(x_f_inp, style_vector=s)
        x_f = self.proj_formant(x_f).transpose(1, 2)
        ret["x_e"] = x_e.transpose(1, 2)
        ret["x_f"] = x_f.transpose(1, 2)

        if infer:
            noise = torch.randn_like(x_e, device=x_e.device) / temperature
            ret["z"] = noise.transpose(1, 2).contiguous()
            z = x_e + noise
            # Generate sample by performing reverse dynamics
            x_e_prime = self.diffusion(z, y_mask, x_e, n_timesteps, stoc, [s, x_f])

            ret["x_e_prime"] = x_e_prime.transpose(1, 2)
            ret["mel_out"] = (
                (x_e_prime + x_f)[:, :, :y_max_length].contiguous().transpose(1, 2)
            )
            ret["attn"] = attn[:, :, :y_max_length].squeeze(1)

        else:

            max_offset = (y_length - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_length)

            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )
            x_e_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )
            x_f_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )
            y_cut_lengths = []
            for i, (y_, x_e_, x_f_, out_offset_) in enumerate(
                zip(y, x_e, x_f, out_offset)
            ):
                y_cut_length = out_size + (y_length[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                x_e_cut[i, :, :y_cut_length] = x_e_[:, cut_lower:cut_upper]
                x_f_cut[i, :, :y_cut_length] = x_f_[:, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths, out_size).unsqueeze(1).to(y_mask)

            y = y_cut
            x_e = x_e_cut
            x_f = x_f_cut
            y_mask = y_cut_mask

            diff_loss, xt, z = self.diffusion.compute_loss(y, y_mask, x_e, [s, x_f])

            x_e = x_e * y_mask
            y = y * y_mask
            x_f = x_f * y_mask
            prior_loss = torch.sum(
                0.5 * ((x_e - (y - x_f)) ** 2 + math.log(2 * math.pi)) * y_mask
            )
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
            ret["prior_loss"] = prior_loss
            ret["diff_loss"] = diff_loss
            ret["z"] = z
        return ret

    def forward_pitch(self, decoder_inp, f0, ret, infer=False, pitch_padding=None):
        ret["f0_pred"] = pitch_pred = self.pitch_predictor(decoder_inp)[:, :, 0]
        f0 = pitch_pred if infer else f0
        f0_denorm = denorm_f0(f0, None, pitch_padding=None)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def forward_energy(self, decoder_inp, energy, ret, infer=False):
        ret["energy_pred"] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        energy_embed_inp = energy_pred if infer else energy
        energy_embed_inp = torch.clamp(
            energy_embed_inp * 256 // 4, min=0, max=255
        ).long()
        energy_embed = self.energy_embed(energy_embed_inp)
        return energy_embed

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS. These will
        no longer recieve a gradient
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas(attn_cpu[ind, 0, : out_lens[ind], : in_lens[ind]])
                attn_out[ind, 0, : out_lens[ind], : in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device()
                )
        return attn_out
