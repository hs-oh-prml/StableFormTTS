import math
import torch
from torch import nn
from torch.nn import Parameter, Linear
from models.commons.layers import Embedding
from utils.nn.seq_utils import (
    get_incremental_state,
    set_incremental_state,
    make_positions,
)
from .saln import StyleAdaptiveLayerNorm
import torch.nn.functional as F

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self, input, incremental_state=None, timestep=None, positions=None, **kwargs
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = (
            make_positions(input, self.padding_idx) if positions is None else positions
        )
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class TransformerFFNLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        filter_size,
        padding="SAME",
        kernel_size=1,
        dropout=0.1,
        act="gelu",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        if padding == "SAME":
            self.ffn_1 = nn.Conv1d(
                hidden_size, filter_size, kernel_size, padding=kernel_size // 2
            )
        elif padding == "LEFT":
            self.ffn_1 = nn.Sequential(
                nn.ConstantPad1d((kernel_size - 1, 0), 0.0),
                nn.Conv1d(hidden_size, filter_size, kernel_size),
            )
        self.ffn_2 = Linear(filter_size, hidden_size)

    def forward(self, x, incremental_state=None):
        # x: T x B x C
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_input" in saved_state:
                prev_input = saved_state["prev_input"]
                x = torch.cat((prev_input, x), dim=0)
            x = x[-self.kernel_size :]
            saved_state["prev_input"] = x
            self._set_input_buffer(incremental_state, saved_state)

        x = self.ffn_1(x.permute(1, 2, 0)).permute(2, 0, 1)
        x = x * self.kernel_size**-0.5

        if incremental_state is not None:
            x = x[-1:]
        if self.act == "gelu":
            x = F.gelu(x)
        if self.act == "relu":
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x

    def _get_input_buffer(self, incremental_state):
        return (
            get_incremental_state(
                self,
                incremental_state,
                "f",
            )
            or {}
        )

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(
            self,
            incremental_state,
            "f",
            buffer,
        )

    def clear_buffer(self, incremental_state):
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_input" in saved_state:
                del saved_state["prev_input"]
            self._set_input_buffer(incremental_state, saved_state)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.1,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.last_attn_probs = None

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
        enc_dec_attn_constraint_mask=None,
        reset_attn_weight=None,
    ):
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        return F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            key_padding_mask,
            need_weights,
            attn_mask,
        )


class EncSALayer(nn.Module):
    def __init__(
        self,
        c,
        num_heads,
        dropout,
        attention_dropout=0.1,
        relu_dropout=0.1,
        kernel_size=9,
        padding="SAME",
        act="gelu",
        saln_dim=64,
    ):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            # self.layer_norm1 = LayerNorm(c)
            self.layer_norm1 = StyleAdaptiveLayerNorm(c, saln_dim)
            self.self_attn = MultiheadAttention(
                self.c,
                num_heads,
                self_attention=True,
                dropout=attention_dropout,
                bias=False,
            )
        # self.layer_norm2 = LayerNorm(c)
        self.layer_norm2 = StyleAdaptiveLayerNorm(c, saln_dim)
        self.ffn = TransformerFFNLayer(
            c,
            4 * c,
            kernel_size=kernel_size,
            dropout=relu_dropout,
            padding=padding,
            act=act,
        )

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get("layer_norm_training", None)
        style_vector = kwargs.get("style_vector", None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x, style_vector)

            (
                x,
                _,
            ) = self.self_attn(
                query=x, key=x, value=x, key_padding_mask=encoder_padding_mask
            )
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x, style_vector)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout, kernel_size=9, num_heads=2, saln_dim=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.op = EncSALayer(
            hidden_size,
            num_heads,
            dropout=dropout,
            attention_dropout=0.1,
            relu_dropout=dropout,
            kernel_size=kernel_size,
            saln_dim=saln_dim,
        )

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class FFTBlocks(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        ffn_kernel_size=9,
        dropout=0.1,
        num_heads=2,
        use_pos_embed=True,
        use_last_norm=True,
        use_pos_embed_alpha=True,
        saln_dim=64,
    ):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.use_last_norm = use_last_norm
        if use_pos_embed:
            self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
            self.padding_idx = 0
            self.pos_embed_alpha = (
                nn.Parameter(torch.Tensor([1])) if use_pos_embed_alpha else 1
            )
            self.embed_positions = SinusoidalPositionalEmbedding(
                embed_dim,
                self.padding_idx,
                init_size=DEFAULT_MAX_TARGET_POSITIONS,
            )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    self.hidden_size,
                    self.dropout,
                    kernel_size=ffn_kernel_size,
                    num_heads=num_heads,
                    saln_dim=saln_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self,
        x,
        padding_mask=None,
        attn_mask=None,
        return_hiddens=False,
        style_vector=None,
    ):
        """
        :param x: [B, T, C]
        :param padding_mask: [B, T]
        :return: [B, T, C] or [L, B, T, C]
        """
        padding_mask = (
            x.abs().sum(-1).eq(0).data if padding_mask is None else padding_mask
        )
        nonpadding_mask_TB = (
            1 - padding_mask.transpose(0, 1).float()[:, :, None]
        )  # [T, B, 1]
        if self.use_pos_embed:
            positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
            x = x + positions
            x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = (
                layer(
                    x,
                    encoder_padding_mask=padding_mask,
                    attn_mask=attn_mask,
                    style_vector=style_vector,
                )
                * nonpadding_mask_TB
            )
            hiddens.append(x)
        if self.use_last_norm:
            x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = torch.stack(hiddens, 0)  # [L, T, B, C]
            x = x.transpose(1, 2)  # [L, B, T, C]
        else:
            x = x.transpose(0, 1)  # [B, T, C]
        return x


class FastSpeechEncoder(FFTBlocks):
    def __init__(
        self,
        dict_size,
        hidden_size=256,
        num_layers=4,
        kernel_size=9,
        num_heads=2,
        dropout=0.1,
        saln_dim=64,
    ):
        super().__init__(
            hidden_size,
            num_layers,
            kernel_size,
            num_heads=num_heads,
            use_pos_embed=False,
            dropout=dropout,
            saln_dim=saln_dim,
        )  # use_pos_embed_alpha for compatibility
        self.embed_tokens = Embedding(dict_size, hidden_size, 0)
        self.embed_scale = math.sqrt(hidden_size)
        self.padding_idx = 0
        self.embed_positions = SinusoidalPositionalEmbedding(
            hidden_size,
            self.padding_idx,
            init_size=DEFAULT_MAX_TARGET_POSITIONS,
        )

    def forward(
        self,
        txt_tokens,
        attn_mask=None,
        style_vector=None,
    ):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [B x T x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(txt_tokens)  # [B, T, H]
        if self.num_layers > 0:
            x = super(FastSpeechEncoder, self).forward(
                x, encoder_padding_mask, attn_mask=attn_mask, style_vector=style_vector
            )
        return x

    def forward_embedding(self, txt_tokens):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        positions = self.embed_positions(txt_tokens)
        x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class FastSpeechDecoder(FFTBlocks):
    def __init__(
        self, hidden_size=256, num_layers=4, kernel_size=9, num_heads=2, saln_dim=64
    ):
        super().__init__(
            hidden_size, num_layers, kernel_size, num_heads=num_heads, saln_dim=saln_dim
        )
