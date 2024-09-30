import librosa
import numpy as np
import pyloudnorm as pyln

from utils.audio.vad import trim_long_silences


def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    """compute right padding (final frame) or both sides padding (first and final frames)"""
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return 10.0 ** (x * 0.05)


def normalize(S, min_level_db):
    return (S - min_level_db) / -min_level_db


def denormalize(D, min_level_db):
    return (D * -min_level_db) + min_level_db


def librosa_wav2spec(
    wav_path,
    fft_size=1024,
    hop_size=256,
    win_length=1024,
    window="hann",
    num_mels=80,
    fmin=80,
    fmax=-1,
    eps=1e-6,
    sample_rate=22050,
    loud_norm=False,
    trim_long_sil=False,
):
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    # get amplitude spectrogram
    x_stft = librosa.stft(
        wav,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="constant",
    )
    linear_spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax
    )

    # calculate mel spec
    mel = mel_basis @ linear_spc
    mel = np.log(np.maximum(eps, mel))  # (n_mel_bins, T)
    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode="constant", constant_values=0.0)
    wav = wav[: mel.shape[1] * hop_size]

    # log linear spec
    linear_spc = np.log(np.maximum(eps, linear_spc))
    return {"wav": wav, "mel": mel.T, "linear": linear_spc.T, "mel_basis": mel_basis}


import torch
from librosa.util import normalize as librosa_normalize


from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    linear = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], linear)
    spec = spectral_normalize_torch(spec)
    return spec, linear


def librosa_wav2spec_bigvgan(
    wav_path,
    fft_size=1024,
    hop_size=256,
    win_length=1024,
    window="hann",
    num_mels=80,
    fmin=80,
    fmax=-1,
    eps=1e-6,
    sample_rate=22050,
    loud_norm=False,
    trim_long_sil=False,
):
    if isinstance(wav_path, str):
        if trim_long_sil:
            wav, _, _ = trim_long_silences(wav_path, sample_rate)
        else:
            wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()
    wav = librosa_normalize(wav) * 0.95
    wav = torch.FloatTensor(wav)
    wav = wav.unsqueeze(0)
    mel, linear_spc = mel_spectrogram(
        wav,
        fft_size,
        num_mels,
        sample_rate,
        hop_size,
        win_length,
        fmin,
        fmax,
        center=False,
    )
    return {
        "wav": wav.squeeze(0).numpy(),
        "mel": mel.squeeze(0).T.numpy(),
        "linear": linear_spc.squeeze(0).T.numpy(),
        "mel_basis": mel_basis,
    }


import torchaudio as T
import torchaudio.transforms as TT
import torch.nn.functional as F


def diffwave_wav2spec(
    wav_path,
    fft_size=1024,
    hop_size=256,
    win_length=1024,
    window="hann",
    num_mels=80,
    fmin=80,
    fmax=-1,
    eps=1e-6,
    sample_rate=22050,
    loud_norm=False,
    trim_long_sil=False,
):
    audio, sr = T.load(wav_path)
    audio = torch.clamp(audio[0], -1.0, 1.0)
    if loud_norm:
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()
    mel_args = {
        "sample_rate": sr,
        "win_length": win_length,
        "hop_length": hop_size,
        "n_fft": fft_size,
        "f_min": 20.0,  # 20
        "f_max": sr / 2.0,  # 11025
        "n_mels": num_mels,
        "power": 1.0,
        "normalized": True,
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)
    with torch.no_grad():
        spectrogram = mel_spec_transform(audio)
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    l_pad, r_pad = librosa_pad_lr(audio, fft_size, hop_size, 1)
    audio = F.pad(audio, (l_pad, r_pad), mode="constant", value=0.0)
    audio = audio[: spectrogram.shape[1] * hop_size]
    return {"wav": audio, "mel": spectrogram}
