# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import torch
from torch import nn
from .common import ConvNorm
from .common import ConvLSTMLinear


class AttributeProcessing:
    def __init__(self, take_log_of_input=False):
        super(AttributeProcessing).__init__()
        self.take_log_of_input = take_log_of_input

    def normalize(self, x):
        if self.take_log_of_input:
            x = torch.log(x + 1)
        return x

    def denormalize(self, x):
        if self.take_log_of_input:
            x = torch.exp(x) - 1
        return x


class BottleneckLayerLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        reduction_factor,
        norm="weightnorm",
        non_linearity="relu",
        kernel_size=3,
        use_partial_padding=False,
    ):
        super(BottleneckLayerLayer, self).__init__()

        self.reduction_factor = reduction_factor
        reduced_dim = int(in_dim / reduction_factor)
        self.out_dim = reduced_dim
        if self.reduction_factor > 1:
            fn = ConvNorm(
                in_dim,
                reduced_dim,
                kernel_size=kernel_size,
                use_weight_norm=(norm == "weightnorm"),
            )
            if norm == "instancenorm":
                fn = nn.Sequential(fn, nn.InstanceNorm1d(reduced_dim, affine=True))

            self.projection_fn = fn
            self.non_linearity = nn.ReLU()
            if non_linearity == "leakyrelu":
                self.non_linearity = nn.LeakyReLU()

    def forward(self, x):
        if self.reduction_factor > 1:
            x = self.projection_fn(x)
            x = self.non_linearity(x)
        return x


class DAP(nn.Module):
    def __init__(
        self,
        n_speaker_dim,
        bottleneck_hparams,
        take_log_of_input,
        arch_hparams,
    ):
        super(DAP, self).__init__()
        self.attribute_processing = AttributeProcessing(take_log_of_input)
        self.bottleneck_layer = BottleneckLayerLayer(**bottleneck_hparams)

        arch_hparams["in_dim"] = self.bottleneck_layer.out_dim + n_speaker_dim
        self.feat_pred_fn = ConvLSTMLinear(**arch_hparams)

    def forward(self, txt_enc, spk_emb, x, lens):
        if x is not None:
            x = self.attribute_processing.normalize(x)

        txt_enc = self.bottleneck_layer(txt_enc)
        spk_emb_expanded = spk_emb[..., None].expand(-1, -1, txt_enc.shape[2])
        context = torch.cat((txt_enc, spk_emb_expanded), 1)

        x_hat = self.feat_pred_fn(context, lens)

        outputs = {"x_hat": x_hat, "x": x}
        return outputs

    def infer(self, z, txt_enc, spk_emb, lens=None):
        x_hat = self.forward(txt_enc, spk_emb, x=None, lens=lens)["x_hat"]
        x_hat = self.attribute_processing.denormalize(x_hat)
        return x_hat
