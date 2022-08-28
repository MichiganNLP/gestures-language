# based on the implementation here: https://github.com/revantteotia/clip-training
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import XLMRobertaModel

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class FirstAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.reshape = nn.Linear(d_model, 768)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        normalized = self.ln_2(x)
        x = x + self.mlp(normalized)
        return x


class PoseTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        first_layer = [FirstAttentionBlock(width, heads, attn_mask)]
        modified_layers =  first_layer + [ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers - 1)]
        self.resblocks = nn.Sequential(*modified_layers)

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class GestureCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        
        super().__init__()
        self.context_length = context_length
        
        self.poseal = PoseTransformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask())
        
        self.transformer_width = transformer_width
        self.transformer_layers = transformer_layers
        self.transformer = XLMRobertaModel.from_pretrained('xlm-roberta-base')

        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, 768))
        self.ln_final = LayerNorm(transformer_width)
        
        self.text_projection = nn.Parameter(torch.empty(768, embed_dim))
        self.pose_projection = nn.Parameter(torch.empty(768, embed_dim))
        
        self.for_better_shape = nn.Parameter(torch.empty(16, 768))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)


        proj_std = (self.transformer_width ** -0.5) * ((2 * self.transformer_layers) ** -0.5)
        attn_std = self.transformer_width ** -0.5
        fc_std = (2 * self.transformer_width) ** -0.5

        for block in self.poseal.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)
            
        if self.pose_projection is not None:
            nn.init.normal_(self.pose_projection, std=16 ** -0.5)
            
        if self.for_better_shape is not None:
            nn.init.normal_(self.for_better_shape, std=16 ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_pose(self, pose):
        pose = pose @ self.for_better_shape
        x = pose + self.positional_embedding # .type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.poseal(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape: [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), self.context_length-1, :] @ self.pose_projection

        return  x

    def encode_text(self, text):
        outputs = self.transformer(**text)
        last_hidden_states = outputs[1]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x =  last_hidden_states @ self.text_projection

        return x

    def forward(self, pose, text):
        image_features = self.encode_pose(pose)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text