"""Export a trained ``CoefMLP`` to the versioned ``.cvcnav`` weight file that the
torch-free C++ policy (``cvc::nav::coef_mlp``) loads.

The same blob feeds torch (training / reference), the CPU C++ forward, and a
future CUDA forward — see docs/CVCNAV_CPP_PORT_ROADMAP.md §4. Wire
:func:`write_coef_mlp` into the training / pipeline so the deployed weights never
drift from the checkpoint; the C++ loader verifies an ``arch_hash`` so a
hidden-size change bumps the file rather than loading silently wrong.

    python -m grl_snam.tools.coef_export <checkpoint.pt> <out.cvcnav>
    python -m grl_snam.tools.coef_export --constant alpha=1.2,beta=3,gamma=4 <out.cvcnav | out.pt>

``--constant`` writes a constant policy, the constant arm of a learning comparison
(:func:`grl_snam.learning.arms.constant_coef_mlp`): every weight zero, and the output's bias
the inverse of the output activation at each constant, so the C++ ``coef_mlp`` returns those
coefficients wherever it is asked. :func:`read_coef_mlp` reads any ``.cvcnav`` back into a
torch ``CoefMLP`` (a run's step-0 blob is the untrained arm).

Format (little-endian):
    char magic[4] = "CVNV"
    u32 format_version, u32 flags, u32 in, u32 out, u32 num_layers, u64 arch_hash
    per layer: u32 rows, u32 cols, u32 act(0=identity,1=SiLU), f32 w[rows*cols], f32 b[rows]
    u32 out_bias_len, f32 out_bias[...]        (raw bias; log(expm1) folded at load)
    [v2 only, FLAG_LAM_SIGMOID] f32 lam_soft_max, f32 lam_hard_max   (sigmoid ceilings)
    u32 meta_len, char meta[meta_len]          (optional provenance)

A two-head sigmoid net (CoefMLP.lam_sigmoid, i.e. use_lam_hard) writes format v2: out_bias_len
is 3 (the abg columns only; the lam heads' init lives in the last Linear bias), the two sigmoid
ceilings follow the out_bias block, and FLAG_LAM_SIGMOID is set so the C++ loader reads the lam
columns as lam_max*sigmoid(raw). A plain / single-softplus-lam net stays v1, byte-unchanged.
"""

from __future__ import annotations

import struct
import sys

import numpy as np

FORMAT_VERSION = 1
FORMAT_VERSION_V2 = 2  # two-head sigmoid reroute (lam_soft, lam_hard) — see FLAG_LAM_SIGMOID
FLAG_SOFTPLUS_LOG_EXPM1 = 1 << 0
# Input-feature-layout flags (must match cvc::nav::coef_mlp kFlagFeatMu/kFlagFeatRisk): a
# 6-input net is ambiguous (grip mu vs terrain-risk), so the C++ drive reads the layout from
# these flags rather than in_features(). Set from CoefMLP.use_mu / use_risk.
FLAG_FEAT_MU = 1 << 1
FLAG_FEAT_RISK = 1 << 2
# The lam OUTPUT columns (>= 3) are lam_max*sigmoid(raw), not softplus (matches
# cvc::nav::coef_mlp kFlagLamSigmoid). Set from CoefMLP.lam_sigmoid (use_lam_hard).
FLAG_LAM_SIGMOID = 1 << 3
_ACT_IDENTITY, _ACT_SILU = 0, 1
_U64 = (1 << 64) - 1


def _fnv1a(values):
    """FNV-1a over a sequence of u64 (byte order matches the C++ compute_arch_hash)."""
    h = 1469598103934665603
    for v in values:
        v &= _U64
        for i in range(8):
            h ^= (v >> (8 * i)) & 0xFF
            h = (h * 1099511628211) & _U64
    return h


def arch_hash(in_features, out_features, shape_act):
    """``shape_act`` is the flat ``[rows, cols, act, ...]`` list, matching the C++."""
    return _fnv1a([in_features, out_features, len(shape_act), *shape_act])


def _layers(model):
    """Extract (weight[out,in], bias[out], act-after) from ``model.net``."""
    import torch

    seq = list(model.net)
    out = []
    for i, m in enumerate(seq):
        if isinstance(m, torch.nn.Linear):
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            act = _ACT_SILU if isinstance(nxt, torch.nn.SiLU) else _ACT_IDENTITY
            w = m.weight.detach().cpu().numpy().astype(np.float32)  # [out, in]
            b = m.bias.detach().cpu().numpy().astype(np.float32)  # [out]
            out.append((w, b, act))
    if not out:
        raise ValueError("coef_export: model.net has no Linear layers")
    return out


def write_coef_mlp(model, path, meta: bytes = b""):
    """Serialize a :class:`sdf_nav.CoefMLP` to ``path`` in the ``.cvcnav`` format."""
    model = model.eval()
    layers = _layers(model)
    in_f = int(layers[0][0].shape[1])
    out_f = int(layers[-1][0].shape[0])
    # A two-head sigmoid net is format v2: the lam columns are lam_max*sigmoid(raw), not softplus.
    sigmoid_lam = bool(getattr(model, "lam_sigmoid", False))
    # The RAW pre-softplus bias. For v1 this is one entry per output (abg + lam_init) so the whole
    # stack is a single uniform softplus. For v2 full_out_bias() returns the 3 abg biases only —
    # the lam heads' init lives in the last Linear bias, and the C++ folds log(expm1) for abg only.
    _ob = model.full_out_bias() if hasattr(model, "full_out_bias") else model.bias
    out_bias = _ob.detach().cpu().numpy().astype(np.float32)
    shape_act = []
    for w, _b, act in layers:
        shape_act += [int(w.shape[0]), int(w.shape[1]), int(act)]
    ah = arch_hash(in_f, out_f, shape_act)
    # Encode the input-feature layout so the C++ drive builds the right columns for a widened
    # net (a bare 6-in net is grip; the risk flag disambiguates it). out_f already signals the
    # lam head (out>=4), so it needs no flag; FLAG_LAM_SIGMOID marks the sigmoid two-head form.
    flags = FLAG_SOFTPLUS_LOG_EXPM1
    if getattr(model, "use_mu", False):
        flags |= FLAG_FEAT_MU
    if getattr(model, "use_risk", False):
        flags |= FLAG_FEAT_RISK
    if sigmoid_lam:
        flags |= FLAG_LAM_SIGMOID
    version = FORMAT_VERSION_V2 if sigmoid_lam else FORMAT_VERSION

    with open(path, "wb") as f:
        f.write(b"CVNV")
        f.write(struct.pack("<IIIII", version, flags, in_f, out_f, len(layers)))
        f.write(struct.pack("<Q", ah))
        for w, b, act in layers:
            f.write(struct.pack("<III", int(w.shape[0]), int(w.shape[1]), int(act)))
            f.write(np.ascontiguousarray(w, np.float32).tobytes())
            f.write(np.ascontiguousarray(b, np.float32).tobytes())
        f.write(struct.pack("<I", out_bias.shape[0]))
        f.write(np.ascontiguousarray(out_bias, np.float32).tobytes())
        if sigmoid_lam:
            # v2 trailer: the sigmoid ceilings, right after the out_bias block (before meta).
            f.write(struct.pack("<ff", float(model.lam_soft_max), float(model.lam_hard_max)))
        f.write(struct.pack("<I", len(meta)))
        f.write(meta)
    return path


def read_coef_mlp(path):
    """The torch ``sdf_nav.CoefMLP`` a ``.cvcnav`` holds (the inverse of :func:`write_coef_mlp`):
    the same forward, and :func:`write_coef_mlp` writes it back byte for byte (bar ``meta``).
    Refuses a blob that is not a CoefMLP (three Linear layers, SiLU between) or whose arch_hash
    does not match its layers."""
    import torch

    import grl_snam.sdf_nav as sdf_nav

    with open(path, "rb") as f:
        raw = f.read()
    off = 0

    def take(fmt):
        nonlocal off
        vals = struct.unpack_from(fmt, raw, off)
        off += struct.calcsize(fmt)
        return vals

    def floats(n):
        nonlocal off
        a = np.frombuffer(raw, np.float32, n, off).copy()
        off += 4 * n
        return a

    if raw[:4] != b"CVNV":
        raise ValueError(f"{path}: not a .cvcnav (magic {raw[:4]!r})")
    off = 4
    version, flags, in_f, out_f, n_layers = take("<IIIII")
    (ah,) = take("<Q")
    if version not in (FORMAT_VERSION, FORMAT_VERSION_V2):
        raise ValueError(f"{path}: format version {version} (this reader knows 1 and 2)")
    layers, shape_act = [], []
    for _ in range(n_layers):
        rows, cols, act = take("<III")
        layers.append((floats(rows * cols).reshape(rows, cols), floats(rows), act))
        shape_act += [rows, cols, act]
    if arch_hash(in_f, out_f, shape_act) != ah:
        raise ValueError(f"{path}: arch_hash does not match the layers")
    if [a for _w, _b, a in layers] != [_ACT_SILU, _ACT_SILU, _ACT_IDENTITY]:
        raise ValueError(f"{path}: not a CoefMLP (three Linear layers, SiLU between)")
    (n_bias,) = take("<I")
    out_bias = floats(n_bias)
    sigmoid = bool(flags & FLAG_LAM_SIGMOID)
    lam_soft_max, lam_hard_max = take("<ff") if sigmoid else (5.0, 10.0)
    model = sdf_nav.CoefMLP(
        hidden=layers[0][0].shape[0],
        bias=tuple(float(v) for v in out_bias[:3]),
        in_dim=in_f,
        use_mu=bool(flags & FLAG_FEAT_MU),
        use_risk=bool(flags & FLAG_FEAT_RISK),
        use_lam=out_f >= 4,
        lam_init=float(out_bias[3]) if out_f >= 4 and not sigmoid else 0.5 * lam_soft_max,
        use_lam_hard=out_f >= 5,
        lam_hard_init=0.5 * lam_hard_max,
        lam_sigmoid=sigmoid,
        lam_soft_max=lam_soft_max,
        lam_hard_max=lam_hard_max,
    )
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, (w, b, _act) in zip(linears, layers, strict=True):
            lin.weight.copy_(torch.from_numpy(w))
            lin.bias.copy_(torch.from_numpy(b))
    return model.eval()


def main(argv=None):
    import argparse

    import torch

    import grl_snam.sdf_nav as sdf_nav

    ap = argparse.ArgumentParser(
        prog="python -m grl_snam.tools.coef_export",
        description="Export a CoefMLP checkpoint, or a constant policy, as a .cvcnav.",
    )
    ap.add_argument("paths", nargs="+", metavar="PATH", help="<checkpoint.pt> <out.cvcnav>")
    ap.add_argument(
        "--constant",
        metavar="SPEC",
        help="write a constant policy instead of a checkpoint: alpha,beta,gamma[,lam_soft"
        "[,lam_hard]] or alpha=..,beta=.. or bias or hist=history.jsonl (the trained run's head "
        "means); the paths are then just <out.cvcnav>",
    )
    ap.add_argument(
        "--like",
        metavar="POLICY",
        help="--constant: take the input layout (features, hidden size) from this .cvcnav / .pt",
    )
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.constant is not None:
        from grl_snam.learning.arms import (
            ArmSpec,
            constant_coef_mlp,
            load_arm_model,
            parse_constant,
        )

        if len(args.paths) != 1:
            ap.error("--constant takes one path, the .cvcnav to write")
        like = load_arm_model(ArmSpec("like", args.like)) if args.like else None
        values = parse_constant(args.constant)
        model = constant_coef_mlp(values, like=like)
        out = args.paths[0]
        meta = "constant policy " + ",".join(f"{k}={v:g}" for k, v in values.items())
        if out.endswith(".pt"):
            # The torch checkpoint (tools.scorecard_eval.load_model reads it) instead of the blob.
            info = {"kind": "constant", "coefficients": values}
            torch.save({"model_state_dict": model.state_dict(), "meta": info}, out)
        else:
            write_coef_mlp(model, out, meta.encode())
        print(f"wrote {out} ({meta})")
        return 0
    if len(args.paths) != 2:
        raise SystemExit("usage: python -m grl_snam.tools.coef_export <checkpoint.pt> <out.cvcnav>")
    ckpt, out = args.paths
    model = sdf_nav.CoefMLP()
    state = torch.load(ckpt, map_location="cpu")
    # Unwrap the state dict from whichever checkpoint format we were given:
    #   * `grl-snam train` (tools/train.py) saves {"model_state_dict", "meta"}
    #   * older checkpoints saved {"model": ...}
    #   * a bare state_dict has neither key.
    # The old code only knew "model", so it fed the whole {"model_state_dict",
    # "meta"} dict to load_state_dict and crashed on every `grl-snam train` output.
    if isinstance(state, dict):
        sd = state.get("model_state_dict") or state.get("model") or state
    else:
        sd = state
    model.load_state_dict(sd)
    write_coef_mlp(model, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    main()
