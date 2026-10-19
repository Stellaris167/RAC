import argparse
from pathlib import Path

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, Trainer, TrainingArguments

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def _mask_last_token_subsequence(labels, input_ids, token_ids):
    pattern = torch.tensor(token_ids, dtype=input_ids.dtype, device=input_ids.device)
    width = len(pattern)
    starts = [
        start
        for start in range(len(input_ids) - width + 1)
        if torch.equal(input_ids[start : start + width], pattern)
    ]
    assert starts, f'Masked SFT text tokens not found: {token_ids}'
    labels[starts[-1] : starts[-1] + width] = -100


class PrefixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        labels = sample['input_ids'].clone()
        labels[sample['loss_mask'] == 0] = -100
        target_message = self.dataset.messages[index][-1]
        mask_text = target_message.get('loss_mask_text')
        if mask_text:
            mask_ids = self.dataset.tokenizer.encode(mask_text, add_special_tokens=False)
            _mask_last_token_subsequence(labels, sample['input_ids'], mask_ids)
        media = dict(sample.get('multi_modal_inputs', {}))
        processor = self.dataset.processor
        if 'mm_token_type_ids' in processor.model_input_names:
            media['mm_token_type_ids'] = torch.tensor(
                processor.create_mm_token_type_ids([sample['input_ids'].tolist()])[0],
                dtype=torch.long,
            )
        return {
            'input_ids': sample['input_ids'],
            'labels': labels,
            **media,
        }


class PrefixTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            super().create_optimizer()
            self.optimizer = ZeroRedundancyOptimizer(
                self.optimizer.param_groups, optimizer_class=torch.optim.AdamW,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon, fused=True,
            )
        return self.optimizer

    def _wrap_model(self, model, training=True, dataloader=None):
        model = super()._wrap_model(model, training=training, dataloader=dataloader)
        if self.accelerator.ddp_handler is not None:
            self.accelerator.ddp_handler.gradient_as_bucket_view = True
        return model


def collate_single_sample(samples):
    assert len(samples) == 1
    sample = samples[0]
    return {key: value.unsqueeze(0) if key in ('input_ids', 'labels', 'mm_token_type_ids') else value for key, value in sample.items()}


def save_bfloat16_model(trainer, output):
    state = trainer.accelerator.get_state_dict(trainer.model)
    if trainer.is_world_process_zero():
        state = {key: value.to(torch.bfloat16) if value.is_floating_point() else value
                 for key, value in state.items()}
        trainer._save(str(output), state_dict=state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--fsdp', action='store_true')
    parser.add_argument('--save-bf16', action='store_true')
    parser.add_argument('--attn-implementation', default='flash_attention_2')
    parser.add_argument(
        '--fsdp-transformer-layer-cls',
        nargs='+',
        default=['Qwen3_5DecoderLayer', 'Qwen3_5VisionBlock'],
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    processor = AutoProcessor.from_pretrained(args.model)
    dataset = MultiTurnSFTDataset(
        args.data, tokenizer,
        {'max_length': args.max_length, 'pad_mode': 'no_padding', 'truncation': 'error',
         'enable_thinking_default': False if args.disable_thinking else None},
        processor,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=args.output,
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-6,
        warmup_steps=2,
        lr_scheduler_type='cosine',
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=True,
        optim='adamw_torch_fused',
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy='no',
        report_to='none',
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=42,
        fsdp='full_shard auto_wrap' if args.fsdp else '',
        fsdp_config={
            'version': 1,
            'transformer_layer_cls_to_wrap': args.fsdp_transformer_layer_cls,
            'use_orig_params': True,
            'sync_module_states': True,
            'state_dict_type': 'FULL_STATE_DICT',
        } if args.fsdp else None,
    )
    trainer_class = Trainer if args.fsdp else PrefixTrainer
    trainer = trainer_class(
        model=model, args=training_args, train_dataset=PrefixDataset(dataset),
        data_collator=collate_single_sample,
    )
    # Trainer already moved the model; sharded optimizer state must stay local.
    if not args.fsdp:
        trainer.accelerator.device_placement = False
    trainer.train()
    output = Path(args.output) / 'huggingface'
    if args.save_bf16:
        save_bfloat16_model(trainer, output)
    else:
        trainer.save_model(str(output))
    trainer.accelerator.wait_for_everyone()
    trainer.save_state()
    if trainer.is_world_process_zero():
        processor.save_pretrained(output)
    trainer.accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
