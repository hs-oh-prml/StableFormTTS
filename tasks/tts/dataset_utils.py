import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
from utils.audio.pitch.utils import norm_interp_f0
from utils.commons.dataset_utils import (
    BaseDataset,
    collate_1d_or_2d,
    collate_1d,
)
from utils.commons.indexed_datasets import IndexedDataset
from utils.text.text_encoder import build_token_encoder
from utils.text import intersperse
import os
from scipy.stats import betabinom


class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None, train=False):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams

        self.data_dir = hparams["binary_data_dir"] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        self.train = train
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == "test" and len(hparams["test_ids"]) > 0:
                self.avail_idxs = hparams["test_ids"]
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == "train" and hparams["min_frames"] > 0:
                self.avail_idxs = [
                    x for x in self.avail_idxs if self.sizes[x] >= hparams["min_frames"]
                ]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, "avail_idxs") and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item["mel"]) == self.sizes[index], (
            len(item["mel"]),
            self.sizes[index],
        )
        max_frames = hparams["max_frames"]
        spec = torch.Tensor(item["mel"])[:max_frames]
        max_frames = (
            spec.shape[0] // hparams["frames_multiple"] * hparams["frames_multiple"]
        )
        spec = spec[:max_frames]
        ph_token = torch.LongTensor(item["ph_token"][: hparams["max_input_tokens"]])
        sample = {
            "id": index,
            "item_name": item["item_name"],
            "text": item["txt"],
            "txt_token": ph_token,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if hparams["use_spk_embed"]:
            sample["spk_embed"] = torch.Tensor(item["spk_embed"])
        if hparams["use_spk_id"]:
            sample["spk_id"] = int(item["spk_id"])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s["id"] for s in samples])
        item_names = [s["item_name"] for s in samples]
        text = [s["text"] for s in samples]
        txt_tokens = collate_1d_or_2d([s["txt_token"] for s in samples], 0)
        mels = collate_1d_or_2d([s["mel"] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s["txt_token"].numel() for s in samples])
        mel_lengths = torch.LongTensor([s["mel"].shape[0] for s in samples])

        batch = {
            "id": id,
            "item_name": item_names,
            "nsamples": len(samples),
            "text": text,
            "txt_tokens": txt_tokens,
            "txt_lengths": txt_lengths,
            "mels": mels,
            "mel_lengths": mel_lengths,
        }

        if hparams["use_spk_embed"]:
            spk_embed = torch.stack([s["spk_embed"] for s in samples])
            batch["spk_embed"] = spk_embed
        if hparams["use_spk_id"]:
            spk_ids = torch.LongTensor([s["spk_id"] for s in samples])
            batch["spk_ids"] = spk_ids
        return batch


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=0.05):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M + 1):
        a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
        rv = betabinom(P - 1, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


class StableFormTTSDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None, train=False):
        super().__init__(prefix, shuffle, items, data_dir, train)
        data_dir = self.hparams["processed_data_dir"]
        self.token_encoder = build_token_encoder(f"{data_dir}/phone_set.json")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        ph_token = sample["txt_token"]
        ph_token = intersperse(ph_token, len(self.token_encoder))
        ph_token = torch.IntTensor(ph_token)

        sample["txt_token"] = ph_token
        hparams = self.hparams
        item = self._get_item(index)
        item_name = item["item_name"]
        spk = item_name.split("_")[0]
        ph_token = sample["txt_token"]
        mel = sample["mel"]
        T = mel.shape[0]

        pitch = torch.LongTensor(item.get(hparams.get("pitch_key", "pitch")))[:T]
        f0, uv = norm_interp_f0(item["f0"][:T])
        uv = torch.FloatTensor(uv)
        f0 = torch.FloatTensor(f0)
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch

        energy = (mel.exp() ** 2).sum(-1).sqrt()
        sample["energy"] = energy

        processed_data_dir = hparams["processed_data_dir"]
        prior_path = f"{processed_data_dir}/prior/{spk}"
        if prior_path:
            os.makedirs(prior_path, exist_ok=True)
            n_tokens = ph_token.numel()
            n_frames = T
            attn_prior = beta_binomial_prior_distribution(
                n_tokens, n_frames, hparams["betabinom_scaling_factor"]
            )
            torch.save(attn_prior, f"{prior_path}/{item_name}.pt")
        else:
            attn_prior = torch.load(prior_path)
        sample["attn_prior"] = attn_prior
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        energy = collate_1d([s["energy"] for s in samples], 0.0)
        batch.update({"energy": energy})

        max_input_len = max(batch["txt_lengths"])
        max_target_len = max(batch["mel_lengths"])

        attn_prior_padded = torch.FloatTensor(
            len(samples), max_target_len, max_input_len
        )
        attn_prior_padded.zero_()
        for idx, i in enumerate(samples):
            cur_attn_prior = i["attn_prior"]
            if cur_attn_prior is None:
                attn_prior_padded = None
            else:
                attn_prior_padded[
                    idx, : cur_attn_prior.size(0), : cur_attn_prior.size(1)
                ] = cur_attn_prior
        f0 = collate_1d_or_2d([s["f0"] for s in samples], 0.0)
        pitch = collate_1d_or_2d([s["pitch"] for s in samples])
        uv = collate_1d_or_2d([s["uv"] for s in samples])
        batch.update(
            {"pitch": pitch, "f0": f0, "uv": uv, "attn_prior": attn_prior_padded}
        )
        return batch
