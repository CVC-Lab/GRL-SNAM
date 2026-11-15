# grl-snam-learning-checkpoints: provenance

Every checkpoint of the learning check's three SDF retrains (GRL-SNAM's
`docs/learning-validation.md`), 41 a seed, five seeds each, as torch checkpoints
(`coef_sdf.stepNNNNNN.pt`, load with `grl_snam` / `sdf_nav.CoefMLP`):

| directory | run | trainer |
|---|---|---|
| `retrain/seed{0..4}/` | §2.3, the retrain that did not descend its own loss | before the C2 fix (today's `--bptt-window 0 --no-col-grad`) |
| `fixed/retrain/seed{0..4}/` | §8.3, the retrain after the fix | `--grad-terms-every 10` |
| `world-units/retrain/seed{0..4}/` | §9.4, the sim in world units with an objective that is the task (since revision 2) | `--grad-terms-every 10`, §9's defaults (`--metres-per-unit 20 --objective task --leg-ticks 1000 --pick clean-reach --val-ticks 1000`) |

Each is from `grl-snam train <austin_south/nav_sdf.npz> -o coef_sdf.pt --steps 4000 --seed $s
--threads 1 --device cpu --history history.jsonl --checkpoint-every 100 --val-frac 0.25
--val-seed 0 --val-pairs 512 --val-ticks 200` with the flags above, and is the file its run's
learning history names as the checkpoint's `blob`, byte for byte. `SHA256SUMS` has every file's
sha256; the histories are committed in GRL-SNAM, with these sha256 (of the `.jsonl.gz`):

| history (`docs/learning-validation/`) | sha256 |
|---|---|
| `retrain/seed0.history.jsonl.gz` | `72f4077e932d99eeaddbeb654c05d4d4c078b66734debd551cb7af27d9de3641` |
| `retrain/seed1.history.jsonl.gz` | `70f7d19e54bcdd2db6a378bdb83786fb406a29802a849d93d4aeb4dd875ca2cc` |
| `retrain/seed2.history.jsonl.gz` | `93b797ea0248fc0515524b0c05e72cd985036ffc7a121af4cad7aae7d0550027` |
| `retrain/seed3.history.jsonl.gz` | `cf5d4bc5b60e0c75d7695808cfc5aad78c887d045559942769bd62c550fc5492` |
| `retrain/seed4.history.jsonl.gz` | `7b8030974924cd41f20e7fbe491fe0b178de409f1eadec2f873f7353caf5cc0c` |
| `fixed/retrain/seed0.history.jsonl.gz` | `0b2f541a204a90af947e0177ae43b23886582a6348256df05b08ec6ca284f889` |
| `fixed/retrain/seed1.history.jsonl.gz` | `b6679b72168a8fbd343094b7c45fb5f419374d82be8833a231acf75162923a9f` |
| `fixed/retrain/seed2.history.jsonl.gz` | `2fcc577fb9c649d59fe57219791f1716b736c59ebd50d2a2bc75e965da840210` |
| `fixed/retrain/seed3.history.jsonl.gz` | `675ec5da31d52fadf68affee2e10ba6df5939275d282b775d5eea75dad5a3ad3` |
| `fixed/retrain/seed4.history.jsonl.gz` | `84fdf18526080609314027c8d34f5bf6138f1b02c83c76dcbc15e7f38dcb442f` |
| `world-units/retrain/seed0.history.jsonl.gz` | `c14d14a950c3cd90b8871dc952acc293ac7244992f86d6f3aa9ea6807c26247f` |
| `world-units/retrain/seed1.history.jsonl.gz` | `db31e72427832d83d9280b08b43c1e5b2e00bcecde95ff64a3c61613700e8282` |
| `world-units/retrain/seed2.history.jsonl.gz` | `c34ca74df32e32568f82edd51bbebb2941d95d4ce98d1124be000dc4a2da2544` |
| `world-units/retrain/seed3.history.jsonl.gz` | `20062bb6f05088e73b516b14dc4bb9ef090446b4a2eb475572655b4614df62c2` |
| `world-units/retrain/seed4.history.jsonl.gz` | `c61ccf06a79f250b956b5851630c06bbfda47e0facefdcdfb0386be494044a0a` |

Deploy every checkpoint of a run (`grl-snam ariadne scrub`; the Pages site's scrub page is built
this way by `pages/assemble.py --scrub`):

```bash
P=<prefix>/share/grl-snam-learning-checkpoints
grl-snam ariadne scrub docs/learning-validation/retrain/seed0.history.jsonl.gz --blobs $P/retrain/seed0
```

Packed from the training machine's run directories (`GRL_SNAM_LEARNING_RUNS`, the layout above);
`build.sh` refuses any file whose sha256 differs from `SHA256SUMS`. Bump `cvc_revision` in
`recipe.yaml` whenever these bytes change.
