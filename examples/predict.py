"""Run from the repository root after installing the complete model bundle."""
import argparse
import json
from pathlib import Path
from research.training.full_label import FullLabelPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model-assets/checkpoints/scibert-full-label-006')
parser.add_argument('--device', choices=('cpu', 'mps', 'auto'), default='cpu')
args = parser.parse_args()
pipeline = FullLabelPipeline(Path(args.model), device=args.device)
result = pipeline.predict({'document_id': 'synthetic-example-001',
                           'text': 'We used NumPy 1.24.3 and MATLAB for numerical analysis.'})
print(json.dumps(result, ensure_ascii=False, indent=2))
