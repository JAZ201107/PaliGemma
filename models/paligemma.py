import torch
import torch.nn as nn
import torch.nn.functional as F

from gemma import KVCache, GemmaConfig
from siglip import SiglipVisionConfig

from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PaliGemmaConfig:
    vision_config: Optional[Dict] = None
    text_config: Optional[Dict] = None
    ignore_index: int = -100
    image_token_index: int = 256000
    vocab_size: int = 257152
    projection_dim: int = 2048
    hidden_size: int = 2048
    pad_token_id: Optional[int] = None
    is_encoder_decoder: bool = False

    def __post_init__(self):
        # Initialize vision_config and text_config using their respective dataclasses
        self.vision_config = (
            SiglipVisionConfig(**self.vision_config) if self.vision_config else None
        )
        self.text_config = (
            GemmaConfig(**self.text_config, pad_token_id=self.pad_token_id)
            if self.text_config
            else None
        )

        # Update vocab_size and other dynamic properties based on sub-configs
        if self.text_config:
            self.vocab_size = self.text_config.vocab_size
        if self.vision_config:
            self.text_config.num_image_tokens = (
                self.vision_config.image_size // self.vision_config.patch_size
            ) ** 2
            self.vision_config.projection_dim = self.projection_dim


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.config = config

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embed: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        # Extract information from the input
        _, _, embed_dim = image_features.shape  # Image
        batch_size, sequence_length = input_ids.shape  # Text + Image
        dtype, device = inputs_embed.dtype, inputs_embed.device

        # Scale image features
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Create Final embedding, which is the concatio of  image embedding and text embedding
        final_embedding = torch.zeros(
            batch_size, sequence_length, embed_dim, dtype=dtype, device=device
        )

        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # (B, S) -> (B, S, E)
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text  embeddings
        final_embedding = torch.where(
            text_mask_expanded, inputs_embed, final_embedding
        )  # If text mask is 1, copy from the inputs_embed, else keep same
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        final_embedding = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding
        )

        # Create attention mask
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embed.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # KV cache is not created or KV cache  is empty
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            assert (
                q_len == 1
            ), "The  query must be one single token, since we are generating "
            kv_len = kv_cache.num_items() + q_len

            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )
        causal_mask = causal_mask.unsqueeze(
            1
        )  # B, Q, KV_len -> B, Num_Heads, Q, KV_len

        # Generate position
        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        else:
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )
        return final_embedding, causal_mask, position_ids

    def forward(self):
        pass
