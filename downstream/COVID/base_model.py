import torch
import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from typing import List, Optional, Tuple, Union
from torch import nn
from sklearn.metrics import accuracy_score
from monai.networks.blocks.mlp import MLPBlock
from timm.models.layers import trunc_normal_

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa


class WCELoss(nn.Module):
    def __init__(self, weight=None, num_classes=2):
        super(WCELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.num_classes = num_classes

    def forward(self, pred, target):
        total_loss = 0
        loss = self.ce_loss(pred, target)
        for i in range(self.num_classes):
            fg = (target == i)
            fg_num = fg.sum().item()
            fg_loss = loss[fg]
            if fg_num > 0:
                total_loss += fg_loss.sum() / fg_num
                # total_loss += fg_loss.sum()
        total_loss = total_loss / self.num_classes
        return total_loss


class SinCosPosEmbed(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, t:int, h: int, w: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 3 == 0, embed_dim
        grid_t = torch.arange(t).float()
        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()
        grid = torch.meshgrid(grid_t, grid_h, grid_w)
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([3, 1, t, h, w])

        emb_t = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
        pos_embed = torch.concatenate([emb_t, emb_h, emb_w], dim=1)  # (H*W, D)

        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(
        embed_dim: int, pos: torch.Tensor
    ) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2).float()
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 64,
        patch_size: Union[int, Tuple[int, int, int]] = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        img_size = (
            (img_size, img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )

        self.img_size, self.embed_dim = img_size, embed_dim
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def patchify(self, x):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        print(x.shape, self.patch_size, self.img_size)
        pt, ph, pw = self.patch_size
        t, h, w = self.grid_size
        x = x.reshape(shape=(x.shape[0], 1, t, pt, h, ph, w, pw))
        x = torch.einsum('nctphqwr->nthwpqrc', x)
        x = x.reshape(shape=(x.shape[0], t * h * w, pt * ph * pw * 1))

        return x

    def forward(self, x: torch.Tensor):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLPBlock(config.hidden_size, config.intermediate_size, 0.0)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        return outputs


class BaseModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.pos_type = config.pos_type
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.embed_tokens = nn.Embedding(4, config.hidden_size)
        self.patchifier = PatchEmbed(embed_dim=config.hidden_size,
                                     in_chans=1,
                                     img_size=(self.img_size[0], self.img_size[1], self.img_size[2]),
                                     patch_size=(self.patch_size[0], self.patch_size[1], self.patch_size[2]))

        if self.pos_type == 'sincos3d':
            self.pos_embed = SinCosPosEmbed()
        elif self.pos_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(self.img_size[0] * self.img_size[1] * self.img_size[2] //
                                                            self.patch_size[0] // self.patch_size[1] // self.patch_size[2] + 1, config.hidden_size))
            trunc_normal_(self.pos_embed, std=.02, a=-.02, b=.02)

        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        # self.init_proj()
        self.apply(self._init_weights)

    def init_proj(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d) following MAE
        if hasattr(self.patchifier, "proj"):
            w = self.patchifier.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_patch_embedding(self, input_image):
        batch_size, seq_length = input_image.shape[0], 1 + (self.img_size[0] // self.patch_size[0]) * \
                                 (self.img_size[1] // self.patch_size[1]) * (self.img_size[2] // self.patch_size[2])

        input_ids = torch.empty(batch_size, 1, dtype=torch.int64, device=input_image.device).fill_(1)
        image_embeds = self.patchifier(input_image)
        t, h, w = self.patchifier.grid_size
        embed_dim = self.patchifier.embed_dim
        starts_embeds = self.embed_tokens(input_ids[..., :1])
        image_embeds = torch.cat((starts_embeds, image_embeds), 1)

        if self.pos_type == 'sincos3d':
            pos_embed = self.pos_embed(t, h, w, embed_dim)
            pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
        elif self.pos_type == 'learnable':
            pos_embed = self.pos_embed

        pos_embed = pos_embed.to(image_embeds.device)
        inputs_embeds = image_embeds + pos_embed[None, ...]

        attention_mask = torch.ones(batch_size, 1, t * h * w + 1, t * h * w + 1, dtype=torch.bool).to(image_embeds.device)
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            0,
        )
        return attention_mask, inputs_embeds

    def forward(
        self,
        input_image,
    ):
        attention_mask, inputs_embeds = self.forward_patch_embedding(input_image)

        # embed positions
        hidden_states = inputs_embeds
        # decoder layers
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)

        return hidden_states


class ClassificationModel(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.score = nn.Linear(768, num_classes, bias=False)
        self.loss_func = WCELoss()

    def forward(self, data):
        x = data['data']
        hidden_states = self.model(x)
        logits = self.score(hidden_states[:, 1:, :])
        pooled_logits = torch.mean(logits, dim=1)
        cls_loss = self.loss_func(pooled_logits, data["labels"])

        if self.cal_acc:
            cls_logits_argmax = torch.argmax(pooled_logits, dim=1)
            sklearn_accuracy = accuracy_score(data["labels"].cpu().numpy(), cls_logits_argmax.cpu().numpy())
            return sklearn_accuracy, F.softmax(pooled_logits, dim=1)
        else:
            return cls_loss
