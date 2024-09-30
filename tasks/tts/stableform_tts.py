import torch
import torch.nn as nn
from torch.nn import functional as F
from tasks.tts.speech_base import SpeechBaseTask
from utils.commons.tensor_utils import tensors_to_scalars


class StableFormTTSTask(SpeechBaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = -1  # blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, sample, infer=False, *args, **kwargs):
        x = sample["txt_tokens"]  # [B, T_t]
        x_lengths = sample["txt_lengths"]
        y = sample["mels"]  # [B, T_s, 80]
        y_lengths = sample["mel_lengths"]  # [B, T_s, 80]
        spk_embed = sample.get("spk_embed")

        if self.global_step >= self.hparams["binarization_start_iter"]:
            binarize = True  # binarization training phase
        else:
            binarize = False  # no binarization, soft alignments only

        if not infer:
            f0 = sample.get("f0")
            energy = sample.get("energy")

            output = self.model(
                x,
                x_lengths,
                y=y.transpose(1, 2),
                y_length=y_lengths,
                f0=f0,
                energy=energy,
                spk=spk_embed,
                out_size=self.hparams["out_size"],
                infer=infer,
                binarize_attention=binarize,
            )
            losses = {}
            losses["prior_loss"] = output["prior_loss"]
            losses["diff_loss"] = output["diff_loss"]
            self.add_dur_loss(output, losses)
            self.add_pitch_loss(output, losses)
            self.add_energy_loss(output, losses)
            self.add_kl_loss(output, losses, binarize)
            self.add_ctc_loss(output["attn_logprob"], x_lengths, y_lengths, losses)

            return losses, output
        else:

            output = self.model(
                x,
                x_lengths,
                y=y.transpose(1, 2),
                y_length=y_lengths,
                spk=spk_embed,
                n_timesteps=50,
                infer=infer,
            )
            return output

    def add_ctc_loss(self, attn_logprob, in_lens, out_lens, losses):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(
            input=attn_logprob, pad=(1, 0, 0, 0, 0, 0, 0, 0), value=self.blank_logprob
        )
        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                : query_lens[bid], :, : key_lens[bid] + 1
            ]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            cost_total += ctc_cost
        cost = cost_total / attn_logprob.shape[0]
        losses["ctc_loss"] = cost * self.hparams["lambda_ctc"]

    def add_kl_loss(self, output, losses, binarize=False):
        w_bin = self.hparams["binarization_loss_weight"]
        hard_attention = output["attn"]
        soft_attention = output["attn_soft"]

        log_sum = torch.log(soft_attention[hard_attention == 1]).sum()
        if binarize and self.global_step >= self.hparams["kl_loss_start_iter"]:
            binarization_loss = -log_sum / hard_attention.sum()
        else:
            binarization_loss = 0
        losses["kl_loss"] = binarization_loss * w_bin

    def add_dur_loss(self, output, losses):
        dur_output = output["dur_output"]
        mask = output["x_mask"]
        dur_loss = F.mse_loss(
            dur_output["x_hat"].squeeze(1), dur_output["x"], reduction="sum"
        )
        dur_loss = dur_loss / mask.sum()
        losses["dur_loss"] = dur_loss

    def add_pitch_loss(self, output, losses):
        f0, f0_pred = output["f0_ph"], output["f0_pred"]
        nonpadding = output["x_mask"]
        loss = (
            F.l1_loss(f0_pred, f0, reduction="none") * nonpadding
        ).sum() / nonpadding.sum()
        loss = loss * self.hparams["lambda_f0"]
        losses["f0_loss"] = loss

    def add_energy_loss(self, output, losses):
        energy, energy_pred = output["energy_ph"], output["energy_pred"]
        nonpadding = output["x_mask"]
        loss = (
            F.mse_loss(energy_pred, energy, reduction="none") * nonpadding
        ).sum() / nonpadding.sum()
        loss = loss * self.hparams["lambda_energy"]
        losses["energy_loss"] = loss

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs["losses"] = {}
        outputs["losses"], _ = self(sample)
        outputs["nsamples"] = sample["nsamples"]

        if (
            self.global_step % self.hparams["valid_infer_interval"] == 0
            and batch_idx < self.hparams["num_valid_plots"]
        ):
            model_out = self(sample, infer=True)
            self.save_valid_result(sample, batch_idx, model_out)

        outputs = tensors_to_scalars(outputs)
        return outputs

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = self.hparams["audio_sample_rate"]
        gt = sample["mels"]
        pred = model_out["mel_out"]
        x_e_prime = model_out["x_e_prime"]
        x_e = model_out["x_e"]
        x_f = model_out["x_f"]
        attn = model_out["attn"].cpu().numpy()

        self.plot_mel(
            batch_idx,
            [gt[0], pred[0], x_f[0], x_e_prime[0], x_e[0]],
            title=f"mel_{batch_idx}",
        )
        self.logger.add_image(
            f"plot_attn_{batch_idx}", self.plot_alignment(attn[0]), self.global_step
        )

        wav_pred = self.vocoder.spec2wav(pred[0].cpu())
        self.logger.add_audio(f"wav_pred_{batch_idx}", wav_pred, self.global_step, sr)

        wav_pred = self.vocoder.spec2wav(x_f[0].cpu())
        self.logger.add_audio(f"wav_x_f_{batch_idx}", wav_pred, self.global_step, sr)

        wav_pred = self.vocoder.spec2wav(x_e_prime[0].cpu())
        self.logger.add_audio(
            f"wav_x_e_prime_{batch_idx}", wav_pred, self.global_step, sr
        )

        wav_pred = self.vocoder.spec2wav(x_e[0].cpu())
        self.logger.add_audio(f"wav_x_e_{batch_idx}", wav_pred, self.global_step, sr)

        if self.global_step <= self.hparams["valid_infer_interval"]:
            wav_gt = self.vocoder.spec2wav(gt[0].cpu())
            self.logger.add_audio(f"wav_gt_{batch_idx}", wav_gt, self.global_step, sr)
