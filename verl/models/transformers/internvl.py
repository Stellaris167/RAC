# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


def internvl_chat_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    image_flags: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """InternVL forward that is tolerant to text-only requests.

    Upstream InternVL requires `pixel_values` as a mandatory positional argument.
    In vLLM / mixed RL pipelines, pure-text requests can appear and should still be
    handled by the language model branch without crashing.
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None:
        if input_ids is not None and hasattr(self, "img_context_token_id"):
            has_image_token = False
            try:
                cmp = input_ids == self.img_context_token_id
                if torch.is_tensor(cmp):
                    has_image_token = bool(cmp.any().item())
                else:
                    has_image_token = bool(cmp)
            except Exception:
                has_image_token = False
            if has_image_token and not getattr(self, "_warned_missing_pixel_values", False):
                print(
                    "[InternVL] Warning: pixel_values is None while image context token exists in input_ids; "
                    "falling back to text-only forward."
                )
                self._warned_missing_pixel_values = True

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    if image_flags is None:
        image_flags = torch.ones(
            (pixel_values.shape[0], 1),
            dtype=torch.long,
            device=pixel_values.device,
        )

    image_flags = image_flags.squeeze(-1)
    input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

    vit_embeds = self.extract_feature(pixel_values)
    vit_embeds = vit_embeds[image_flags == 1]

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    flat_input_ids = input_ids.reshape(B * N)
    selected = flat_input_ids == self.img_context_token_id
    try:
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
    except Exception:
        vit_embeds = vit_embeds.reshape(-1, C)
        n_token = min(selected.sum(), vit_embeds.size(0))
        input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

    input_embeds = input_embeds.reshape(B, N, C)

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    logits = outputs.logits

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
