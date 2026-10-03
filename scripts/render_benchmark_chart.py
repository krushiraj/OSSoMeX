"""Render aggregate scores without archived workstation paths or model inference."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scores', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = json.loads(args.scores.read_text())
datasets = ['fresh30', 'exposed50', 'reserve27', 'regression7', 'softcite-gold-paragraphs']
fig, ax = plt.subplots(figsize=(10, 5))
for offset, (arm, policy, label) in enumerate([
    ('ossomex012', 'native', 'OSSoMeX 012'),
    ('wapiti', 'explicit', 'Softcite Wapiti'),
    ('scibert', 'explicit', 'Softcite SciBERT'),
]):
    values = [100 * next(r['f1'] for r in rows if r['dataset'] == ds and r['arm'] == arm
                        and r['policy'] == policy) for ds in datasets]
    bars = ax.bar([i + (offset - 1) * .25 for i in range(len(datasets))], values, .25, label=label)
    ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
ax.set_xticks(range(len(datasets)), ['fresh30', 'exposed50', 'reserve27', 'regression7', 'Softcite gold'])
ax.set_ylabel('Exact software-name F1 (%)')
ax.set_ylim(0, 105)
ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.17))
fig.text(.08, .03, 'Local snippets: provisional and development-exposed. Gold: recovered paragraphs.\nSoftcite projection includes languages and excludes implicit names. Do not pool overlapping evaluations.', fontsize=8)
fig.subplots_adjust(bottom=.2, top=.82)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180)
