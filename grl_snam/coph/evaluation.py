"""Frozen-manifest development evaluations with evaluator-only truth access."""
# grl_snam.coph: trimmed to load and dump. The frozen-manifest single-agent evaluation suite is
# not vendored (VENDORED.md).
from pathlib import Path
import gzip,json


def load(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)


def dump(path,data):
    path=Path(path)
    with (gzip.open(path,'wt') if path.suffix=='.gz' else path.open('w')) as f:json.dump(data,f,allow_nan=False)
