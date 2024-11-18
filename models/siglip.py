from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SiglipVisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: Optional[int] = None


class SiglipVisionEmbeddings(nn.Module):
    """
    Encoding the images as the Patches
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        B, C, H, W = pixel_values.size()

        # B, Embed_dim, Num_Patches, Num_Patches
        patch_embeds = self.patch_embedding(pixel_values)

        # B, Embed_dim, Num_Patches ** 2
        embeddings = patch_embeds.flatten(2)

        # B, Num_Patches ** 2  , Embed_dim
        embeddings = embeddings.transpose(1, 2)

        # B, Num_Patches ** 2  , Embed_dim
        embeddings = embeddings + self.position_embeddings(self.position_ids)

        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, S, _ = hidden_state.size()

        qkv = self.qkv_proj(hidden_state)
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if scores.size() != (B, self.num_heads, S, S):
            raise ValueError(
                f"Attention weights should be of size {(B, self.num_heads, S, S)}, but is"
                f" {scores.size()}"
            )

        scores = F.softmax(scores, dim=-1).to(q.dtype)
        scores = F.dropout(scores, p=self.dropout, training=self.training)

        out = torch.matmul(scores, v)

        out = (
            out.transpose(1, 2)
            .contiguous()
            .reshape(B, -1, self.head_dim * self.num_heads)
        )

        out = self.out_proj(out)
        return out, scores


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)

        hidden_state = F.gelu(hidden_state, approximate="tanh")

        hidden_state = self.fc2(hidden_state)

        return hidden_state


def sub_connection(x: torch.Tensor, layer) -> torch.Tensor:
    return x + layer(x)


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_nrom1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = sub_connection(
            hidden_states, lambda x: self.self_attn(self.layer_nrom1(x))[0]
        )

        hidden_states = sub_connection(
            hidden_states, lambda x: self.mlp(self.layer_norm2(x))
        )

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds

        for l in self.layers:
            hidden_states = l(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.poster_layernrom = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)

        last_hidden_states = self.encoder(hidden_states)

        last_hidden_states = self.poster_layernrom(hidden_states)

        return last_hidden_states


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values)


if __name__ == "__main__":
    config = SiglipVisionConfig()
    x = torch.randn((4, config.num_channels, config.image_size, config.image_size))
    m = SiglipVisionModel(config)
    o = m(x)
    assert (o.shape) == (
        4,
        (config.image_size / config.patch_size) ** 2,
        config.hidden_size,
    )
    print("Test pass")
