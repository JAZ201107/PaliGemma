import torch
import torch.nn as nn
import torch.nn.functional as F

from gemma import KVCache, GemmaConfig, GemmaForCausalLM
from siglip import SiglipVisionConfig, SiglipVisionModel

from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PaliGemmaConfig:
    vision_config: Optional[Dict] = None
    text_config: Optional[Dict] = None
    ignore_index: int = -100
    image_token_index: int = 256000
    vocab_size: int = 257152
    projection_dim: int = 2048
    hidden_size: int = 2048 (
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


class PaliGemmaMultiModelProjector(nn.Module):
    """Projecting Image Embedding to Text Embedding Space

    Args:
        nn (_type_): _description_
    """

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.linear = nn.Linear(
            config.vision_config, config.vision_config.projection_dim, bias=True
        )

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.config = config
        self.vision_tower = SiglipVisionModel(config=config.vision_config)
        self.multi_model_projector = PaliGemmaMultiModelProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_ids = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        # Share weight in the decoder layer
        return self.language_model.tie_weight()

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

        # Create Final embedding, which is the concat of  image embedding and text embedding
        final_embedding = torch.zeros(
            batch_size, sequence_length, embed_dim, dtype=dtype, device=device
        )

        # ==== Create Mask =====
        # Tell which is the placeholder token for image
        # which is the text token
        # which is the padding token(also we don't have the pad token)
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )  # Text token is not image token and not padding token
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # (B, S) -> (B, S, E)
        # Expand the mask to the same shape as the final_embedding
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        # ==== End Create Mask =====

        # ==== Apply Mask =====
        final_embedding = torch.where(
            text_mask_expanded, inputs_embed, final_embedding
        )  # If text mask is 1, copy from the inputs_embed, else keep same
        # image_mask_expanded shape: (B, S, E)
        # scaled_image_features shape: (B, num_image_tokens( less than S) ,E)
        # So we need `masked_scatter``
        
        num_true_values = image_mask_expanded.sum().item()
        num_source_elements = scaled_image_features.numel()
        assert num_true_values == num_source_elements, "Mismatch between mask and source tensor sizes"
        
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embedding = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding
        )
        # ==== End Apply Mask =====

        # ==== Create attention mask ==== 
        # min_dtype = torch.finfo(dtype).min
        q_len = inputs_embed.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # KV cache is not created or KV cache  is empty
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
            # Not masking anything during prefilling the cache
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

        # ==== Generate position id ====
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # The <image> tokens in the embedding is not the correct image embeddings
        # Need to be replaced by the actual embedding through Vision Encoder

        # Get the actual image embedding
        selected_image_features = self.vision_tower(
            pixel_values.to(inputs_embeds.dtype)
        )
        image_features = self.multi_model_projector(selected_image_features)

        # Merge Tokens
        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        )

        # Feed the input and mask into the LLM
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
