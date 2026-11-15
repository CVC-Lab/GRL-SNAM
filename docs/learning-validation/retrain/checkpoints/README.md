# The retrain's saved policies

`seed{0..4}/coef_sdf.step{000000..004000}.pt`: every checkpoint the five retrain histories beside
this directory (`../seed{0..4}.history.jsonl.gz`, §2.3 of
[`docs/learning-validation.md`](../../../learning-validation.md)) name as their `blob`, 41 a seed,
saved every 100 steps by `grl-snam train ... --checkpoint-every 100`. They are the files the
histories were written with, from the same five runs whose picks (`coef_sdf.best_heldout.pt`,
not here) have the sha256 §7 lists.

The Pages site's scrub page deploys them (`pages/assemble.py --scrub`, which looks for a history's
policies in `checkpoints/<run>/`), and so can you:

```bash
grl-snam ariadne scrub docs/learning-validation/retrain/seed0.history.jsonl.gz \
  --blobs docs/learning-validation/retrain/checkpoints/seed0 --cache /tmp/seed0.scrub
```
