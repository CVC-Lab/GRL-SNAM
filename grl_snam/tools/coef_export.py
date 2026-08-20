"""Export a trained ``CoefMLP`` to the versioned ``.cvcnav`` weight file that the
torch-free C++ policy (``cvc::nav::coef_mlp``) loads.

The same blob feeds torch (training / reference), the CPU C++ forward, and a
future CUDA forward — see docs/CVCNAV_CPP_PORT_ROADMAP.md §4. Wire
:func:`write_coef_mlp` into the training / pipeline so the deployed weights never
drift from the checkpoint; the C++ loader verifies an ``arch_hash`` so a
hidden-size change bumps the file rather than loading silently wrong.

    python -m grl_snam.tools.coef_export <checkpoint.pt> <out.cvcnav>

Format (little-endian):
    char magic[4] = "CVNV"
    u32 format_version, u32 flags, u32 in, u32 out, u32 num_layers, u64 arch_hash
    per layer: u32 rows, u32 cols, u32 act(0=identity,1=SiLU), f32 w[rows*cols], f32 b[rows]
    u32 out_bias_len, f32 out_bias[...]        (raw bias; log(expm1) folded at load)
    u32 meta_len, char meta[meta_len]          (optional provenance)
"""

from __future__ import annotations

import struct
import sys

import numpy as np

FORMAT_VERSION = 1
FLAG_SOFTPLUS_LOG_EXPM1 = 1 << 0
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
    out_bias = model.bias.detach().cpu().numpy().astype(np.float32)  # the (1,3,4) buffer
    shape_act = []
    for w, _b, act in layers:
        shape_act += [int(w.shape[0]), int(w.shape[1]), int(act)]
    ah = arch_hash(in_f, out_f, shape_act)

    with open(path, "wb") as f:
        f.write(b"CVNV")
        f.write(
            struct.pack("<IIIII", FORMAT_VERSION, FLAG_SOFTPLUS_LOG_EXPM1, in_f, out_f, len(layers))
        )
        f.write(struct.pack("<Q", ah))
        for w, b, act in layers:
            f.write(struct.pack("<III", int(w.shape[0]), int(w.shape[1]), int(act)))
            f.write(np.ascontiguousarray(w, np.float32).tobytes())
            f.write(np.ascontiguousarray(b, np.float32).tobytes())
        f.write(struct.pack("<I", out_bias.shape[0]))
        f.write(np.ascontiguousarray(out_bias, np.float32).tobytes())
        f.write(struct.pack("<I", len(meta)))
        f.write(meta)
    return path


def main(argv=None):
    import torch

    import sdf_nav

    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        raise SystemExit("usage: python -m grl_snam.tools.coef_export <checkpoint.pt> <out.cvcnav>")
    ckpt, out = argv
    model = sdf_nav.CoefMLP()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
    write_coef_mlp(model, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
