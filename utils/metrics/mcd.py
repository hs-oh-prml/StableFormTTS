# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np

DEFAULT_V_MIN = np.log(1e-5)


def antidiag_indices(offset, min_i=0, max_i=None, min_j=0, max_j=None):
    """
    for a (3, 4) matrix with min_i=1, max_i=3, min_j=1, max_j=4, outputs

    offset=2 (1, 1),
    offset=3 (2, 1), (1, 2)
    offset=4 (2, 2), (1, 3)
    offset=5 (2, 3)

    constraints:
        i + j = offset
        min_j <= j < max_j
        min_i <= offset - j < max_i
    """
    if max_i is None:
        max_i = offset + 1
    if max_j is None:
        max_j = offset + 1
    min_j = max(min_j, offset - max_i + 1, 0)
    max_j = min(max_j, offset - min_i + 1, offset + 1)
    j = torch.arange(min_j, max_j)
    i = offset - j
    return torch.stack([i, j])


def batch_dynamic_time_warping(distance, shapes=None):
    """full batched DTW without any constraints

    distance:  (batchsize, max_M, max_N) matrix
    shapes: (batchsize,) vector specifying (M, N) for each entry
    """
    # ptr: 0=left, 1=up-left, 2=up
    ptr2dij = {0: (0, -1), 1: (-1, -1), 2: (-1, 0)}

    bsz, m, n = distance.size()
    cumdist = torch.zeros_like(distance)
    backptr = torch.zeros_like(distance).type(torch.int32) - 1

    # initialize
    cumdist[:, 0, :] = distance[:, 0, :].cumsum(dim=-1)
    cumdist[:, :, 0] = distance[:, :, 0].cumsum(dim=-1)
    backptr[:, 0, :] = 0
    backptr[:, :, 0] = 2

    # DP with optimized anti-diagonal parallelization, O(M+N) steps
    for offset in range(2, m + n - 1):
        ind = antidiag_indices(offset, 1, m, 1, n)
        c = torch.stack(
            [
                cumdist[:, ind[0], ind[1] - 1],
                cumdist[:, ind[0] - 1, ind[1] - 1],
                cumdist[:, ind[0] - 1, ind[1]],
            ],
            dim=2,
        )
        v, b = c.min(axis=-1)
        backptr[:, ind[0], ind[1]] = b.int()
        cumdist[:, ind[0], ind[1]] = v + distance[:, ind[0], ind[1]]

    # backtrace
    pathmap = torch.zeros_like(backptr)
    for b in range(bsz):
        i = m - 1 if shapes is None else (shapes[b][0] - 1).item()
        j = n - 1 if shapes is None else (shapes[b][1] - 1).item()
        dtwpath = [(i, j)]
        while (i != 0 or j != 0) and len(dtwpath) < 10000:
            assert i >= 0 and j >= 0
            di, dj = ptr2dij[backptr[b, i, j].item()]
            i, j = i + di, j + dj
            dtwpath.append((i, j))
        dtwpath = dtwpath[::-1]
        indices = torch.from_numpy(np.array(dtwpath))
        pathmap[b, indices[:, 0], indices[:, 1]] = 1

    return cumdist, backptr, pathmap


def compute_l2_dist(x1, x2):
    """compute an (m, n) L2 distance matrix from (m, d) and (n, d) matrices"""
    return torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0), p=2).squeeze(0).pow(2)


def compute_rms_dist(x1, x2):
    l2_dist = compute_l2_dist(x1, x2)
    return (l2_dist / x1.size(1)).pow(0.5)


def get_divisor(pathmap, normalize_type):
    if normalize_type is None:
        return 1
    elif normalize_type == "len1":
        return pathmap.size(0)
    elif normalize_type == "len2":
        return pathmap.size(1)
    elif normalize_type == "path":
        return pathmap.sum().item()
    else:
        raise ValueError(f"normalize_type {normalize_type} not supported")


def batch_compute_distortion(y1, y2, sr, feat_fn, dist_fn, normalize_type):
    d, s, x1, x2 = [], [], [], []
    for cur_y1, cur_y2 in zip(y1, y2):
        assert cur_y1.ndim == 1 and cur_y2.ndim == 1
        cur_x1 = feat_fn(cur_y1)
        cur_x2 = feat_fn(cur_y2)
        x1.append(cur_x1)
        x2.append(cur_x2)

        cur_d = dist_fn(cur_x1, cur_x2)
        d.append(cur_d)
        s.append(d[-1].size())
    max_m = max(ss[0] for ss in s)
    max_n = max(ss[1] for ss in s)
    d = torch.stack(
        [F.pad(dd, (0, max_n - dd.size(1), 0, max_m - dd.size(0))) for dd in d]
    )
    s = torch.LongTensor(s).to(d.device)
    cumdists, backptrs, pathmaps = batch_dynamic_time_warping(d, s)

    rets = []
    itr = zip(s, x1, x2, d, cumdists, backptrs, pathmaps)
    for (m, n), cur_x1, cur_x2, dist, cumdist, backptr, pathmap in itr:
        cumdist = cumdist[:m, :n]
        backptr = backptr[:m, :n]
        pathmap = pathmap[:m, :n]
        divisor = get_divisor(pathmap, normalize_type)

        distortion = cumdist[-1, -1] / divisor
        ret = distortion, (cur_x1, cur_x2, dist, cumdist, backptr, pathmap)
        rets.append(ret)
    return rets


def batch_mel_cepstral_distortion(y1, y2, sr, normalize_type="path", mfcc_fn=None):
    """
    https://arxiv.org/pdf/2011.03568.pdf

    The root mean squared error computed on 13-dimensional MFCC using DTW for
    alignment. MFCC features are computed from an 80-channel log-mel
    spectrogram using a 50ms Hann window and hop of 12.5ms.

    y1: list of waveforms
    y2: list of waveforms
    sr: sampling rate
    """

    try:
        import torchaudio
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    if mfcc_fn is None or mfcc_fn.sample_rate != sr:
        melkwargs = {
            "n_fft": int(0.05 * sr),
            "win_length": int(0.05 * sr),
            "hop_length": int(0.0125 * sr),
            "f_min": 20,
            "n_mels": 80,
            "window_fn": torch.hann_window,
        }
        mfcc_fn = torchaudio.transforms.MFCC(
            sr, n_mfcc=13, log_mels=True, melkwargs=melkwargs
        ).to(y1[0].device)
    return batch_compute_distortion(
        y1,
        y2,
        sr,
        lambda y: mfcc_fn(y).transpose(-1, -2),
        compute_rms_dist,
        normalize_type,
    )
