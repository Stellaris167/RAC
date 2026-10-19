"""Re-score saved outputs into a separate audit summary, preserving original files."""

import argparse
import json
from collections import Counter
from pathlib import Path

from verl.trainer.reward_fn.confidence_reward import compute_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for path in args.inputs:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        counts = Counter(total=len(rows))
        reasons = Counter()
        examples = []
        for i, row in enumerate(rows):
            result = compute_score('replay', row['output'], row['gts'], extra_info={
                'is_multiple_choice': bool(row['is_multiple_choice']), 'response_schema': 'boxed',
            }, format_reward_warmup_steps=0)
            counts['old_correct'] += row['acc'] == 1
            counts['valid'] += result['accuracy_valid'] == 1
            counts['skipped'] += result['accuracy_valid'] == 0
            counts['correct'] += result['acc'] == 1
            counts['wrong_to_correct'] += row['acc'] == 0 and result['acc'] == 1
            counts['correct_to_wrong'] += row['acc'] == 1 and result['acc'] == 0
            counts['correct_to_skipped'] += row['acc'] == 1 and result['acc'] == -1
            counts['invalid_confidence'] += result['confidence_valid'] == 0
            if result['skip_reason']:
                reasons[result['skip_reason']] += 1
            if row['acc'] != result['acc'] and len(examples) < 15:
                examples.append({'row': i, 'output': row['output'], 'gold': row['gts'],
                    'old_answer': row['answer_text'], 'new_answer': result['answer_text'],
                    'old_acc': row['acc'], 'new_acc': result['acc'], 'reason': result['skip_reason']})
        report = {'counts': dict(counts), 'skip_reasons': dict(reasons),
            'accuracy_valid_only': counts['correct'] / counts['valid'] if counts['valid'] else None,
            'coverage': counts['valid'] / len(rows), 'examples': examples}
        reports[str(path)] = report
        print(str(path), json.dumps({key: value for key, value in report.items() if key != 'examples'}))
    args.output.write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
