import argparse
import json
from pathlib import Path

import pandas as pd
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from verl.tools.build_single_box_sft import load_prompt_messages
from verl.trainer.reward_fn.confidence_dataset import _rewrite_local_media_paths, inject_confidence_prompt
from verl.trainer.reward_fn.confidence_reward import compute_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples-per-condition', type=int, default=8)
    parser.add_argument('--disable-thinking', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=3072)
    parser.add_argument('--max-model-len', type=int, default=16000)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    requests, metadata = [], []
    for benchmark in ('m3cot', 'mathverse', 'mathvision', 'mmmu', 'scienceqa', 'we_math'):
        for severity in ('', '_T0.6'):
            path = Path(args.data_root) / f'{benchmark}_test_processed{severity}' / 'test.parquet'
            frame = pd.read_parquet(path).sample(n=args.samples_per_condition, random_state=42)
            for _, row in frame.iterrows():
                messages = load_prompt_messages(row['message_qwenvl'])
                messages = _rewrite_local_media_paths(messages, args.data_root)
                messages = inject_confidence_prompt(messages, response_schema='boxed_direct' if args.disable_thinking else 'boxed')
                images, videos = process_vision_info(messages)
                media = {}
                if images:
                    media['image'] = images
                if videos:
                    media['video'] = videos
                template_kwargs = {'enable_thinking': False} if args.disable_thinking else {}
                requests.append(dict(prompt=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs), multi_modal_data=media))
                metadata.append(dict(benchmark=benchmark, condition=severity or 'clean', sample_id=str(row['sample_id']), gold=str(row['answer']), is_multiple_choice=bool(row['is_multiple_choice']), choices=list(row['choices'])))
    llm = LLM(model=args.model, dtype='bfloat16', max_model_len=args.max_model_len, gpu_memory_utilization=0.65, enforce_eager=True, seed=42)
    outputs = llm.generate(requests, SamplingParams(temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.1, max_tokens=args.max_tokens))
    records = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as stream:
        for meta, output in zip(metadata, outputs, strict=True):
            response = output.outputs[0].text
            scores = compute_score(meta['benchmark'], response, meta['gold'], extra_info=meta, response_schema='boxed')
            record = {**meta, 'response': response, 'finish_reason': output.outputs[0].finish_reason,
                      'output_tokens': len(output.outputs[0].token_ids), 'prompt_tokens': len(output.prompt_token_ids), **scores}
            record['direct_format_ok'] = bool(scores['format_ok'] and response.strip().startswith(r'\boxed{answer='))
            stream.write(json.dumps(record) + '\n')
            records.append(record)
    frame = pd.DataFrame(records)
    print(frame.groupby(['benchmark', 'condition'])[['format_ok', 'acc']].mean().to_string())
    print(json.dumps(dict(n=len(frame), format_rate=float(frame.format_ok.mean()),
                         direct_format_rate=float(frame.direct_format_ok.mean()),
                         truncation_rate=float((frame.finish_reason == 'length').mean()),
                         accuracy=float(frame.acc.mean()))))


if __name__ == '__main__':
    main()
