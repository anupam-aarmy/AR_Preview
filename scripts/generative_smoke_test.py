import json
import os
import subprocess
import sys
from datetime import datetime

"""Simple smoke test runner for generative pipeline.
Runs two configurations:
 1. Fast mode baseline (no depth)
 2. Fast mode + depth (if depth model available)
Aggregates delta metrics if present.
Usage (from repo root):
  python scripts/generative_smoke_test.py --room assets/room_wall.png --product TV --steps 20
"""
import argparse

def run(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)


def find_delta_report(output_root='output/task2_generative'):
    path = os.path.join(output_root, 'delta_report.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def summarize_delta(delta):
    if not delta or 'entries' not in delta:
        return 'No delta report'
    lines = []
    for e in delta['entries']:
        lines.append(f"{e['size_key']}: ssim={e['ssim']:.4f} mse={e['mse']:.1f} changed={e['changed_ratio']:.3f} fallback={e['fallback_applied']}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--room', default='assets/room_wall.png', help='Room image path')
    ap.add_argument('--product', default='TV', choices=['TV','Painting','Both'])
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--depth-model', default='lllyasviel/control_v11f1p_sd15_depth')
    args = ap.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('output','smoke_runs')
    os.makedirs(log_dir, exist_ok=True)

    configs = [
        {
            'name': 'fast-baseline',
            'cmd': [sys.executable, 'generative_pipeline.py', '--fast', '--product-type', args.product, '--room-image', args.room, '--steps', str(args.steps)]
        },
        {
            'name': 'fast-depth',
            'cmd': [sys.executable, 'generative_pipeline.py', '--fast', '--use-depth', '--depth-model', args.depth_model, '--product-type', args.product, '--room-image', args.room, '--steps', str(args.steps)]
        }
    ]

    summary = {}
    for cfg in configs:
        try:
            run(cfg['cmd'])
            delta = find_delta_report()
            summary[cfg['name']] = delta
        except Exception as e:
            summary[cfg['name']] = {'error': str(e)}

    # Write consolidated summary
    summary_path = os.path.join(log_dir, f'smoke_summary_{timestamp}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\n=== Smoke Test Summary ===')
    for name, delta in summary.items():
        if isinstance(delta, dict) and 'entries' in delta:
            print(f"[{name}]\n{summarize_delta(delta)}")
        else:
            print(f"[{name}] {delta}")
    print(f"\nSaved summary: {summary_path}")

if __name__ == '__main__':
    main()
