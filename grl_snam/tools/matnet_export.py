"""``.cvcnm`` weight container for ``CoefEnergyNetMaterial`` + a numpy reference.

The learned material coefficient network (``material_nav.CoefEnergyNetMaterial``
— a transformer over obstacle/goal tokens plus a CNN risk-patch encoder) cannot
be expressed by the ``.cvcnav`` format (a linear chain of ``Linear`` layers).
This module defines a sibling container ``.cvcnm`` (magic ``CVNM``) and a
**numpy** forward that reads it — the torch-free reference the C++ port
(``cvc::nav::coef_energy_net``) is validated against.

Why a numpy reference and not "just compare to torch": torch's
``TransformerEncoder`` in eval/no-grad may dispatch to a fused attention fast
path (BetterTransformer / flash / mem-efficient SDPA / NestedTensor) that
*skips* padded tokens and rounds differently from naive math attention. The
numpy reference here is the explicit **math path** — post-norm layers, ReLU
FFN, no final encoder norm, masked-softmax with row-max — that both torch (with
the fast path disabled) and the C++ twin must match. It is the parity oracle.

Format (little-endian, float32 row-major):
    "CVNM"                     4 bytes magic
    format_version  u32        = 1
    arch_hash       u64        FNV-1a over sorted (name, shape) descriptors
    d_tok u32, nhead u32, num_layers u32, d_risk u32, patch_size u32
    lam_soft_max f32, lam_hard_max f32, mu_lat_max f32, eps f32
    n_tensors u32
    per tensor: name_len u32, name utf-8, ndim u32, dims[ndim] u32, f32 data
    meta_len u32, meta utf-8
"""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"CVNM"
FORMAT_VERSION = 1
EPS = 1e-5


def _fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def arch_hash(named_shapes) -> int:
    """FNV-1a over sorted ``name|d0,d1,...`` descriptors — pins the topology so
    a shape/layer change invalidates old files (the ``.cvcnav`` discipline)."""
    desc = ";".join(f"{n}|{','.join(map(str, s))}" for n, s in sorted(named_shapes))
    return _fnv1a64(desc.encode("utf-8"))


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_matnet(model, path: str, *, meta: str = "") -> str:
    """Serialize a ``CoefEnergyNetMaterial`` to ``path`` as ``.cvcnm``.

    Reads ``model.state_dict()`` — every parameter by name, float32 row-major
    (torch ``Linear`` weight is ``[out, in]``; ``Conv2d`` weight
    ``[Cout, Cin, kH, kW]``; ``in_proj_weight`` ``[3*d, d]``)."""
    sd = model.state_dict()
    tensors = [(k, v.detach().cpu().numpy().astype(np.float32)) for k, v in sd.items()]
    named_shapes = [(k, tuple(v.shape)) for k, v in tensors]

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<Q", arch_hash(named_shapes)))
        f.write(
            struct.pack(
                "<IIIII",
                64,  # d_tok
                4,  # nhead
                2,  # num_layers
                64,  # d_risk
                int(getattr(model, "patch_size", 32)) if hasattr(model, "patch_size") else 32,
            )
        )
        f.write(
            struct.pack(
                "<ffff",
                float(model.lam_soft_max),
                float(model.lam_hard_max),
                float(model.mu_lat_max),
                EPS,
            )
        )
        f.write(struct.pack("<I", len(tensors)))
        for name, arr in tensors:
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", int(d)))
            f.write(np.ascontiguousarray(arr, np.float32).tobytes())
        mb = meta.encode("utf-8")
        f.write(struct.pack("<I", len(mb)))
        f.write(mb)
    return path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class MatNet:
    """Parsed ``.cvcnm``: tensors by name + the scalar hyperparameters."""

    def __init__(self, tensors, d_tok, nhead, num_layers, d_risk, patch_size, caps, arch):
        self.t = tensors
        self.d_tok = d_tok
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_risk = d_risk
        self.patch_size = patch_size
        self.lam_soft_max, self.lam_hard_max, self.mu_lat_max, self.eps = caps
        self.arch_hash = arch


def read_matnet(path: str) -> MatNet:
    with open(path, "rb") as f:
        buf = f.read()
    off = 0

    def take(fmt):
        nonlocal off
        sz = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, buf, off)
        off += sz
        return vals

    assert buf[:4] == MAGIC, "not a .cvcnm file"
    off = 4
    (ver,) = take("<I")
    assert ver == FORMAT_VERSION, f"unsupported .cvcnm version {ver}"
    (arch,) = take("<Q")
    d_tok, nhead, num_layers, d_risk, patch_size = take("<IIIII")
    caps = take("<ffff")
    (n_tensors,) = take("<I")
    tensors = {}
    for _ in range(n_tensors):
        (nl,) = take("<I")
        name = buf[off : off + nl].decode("utf-8")
        off += nl
        (ndim,) = take("<I")
        dims = take("<" + "I" * ndim)
        count = int(np.prod(dims)) if dims else 1
        arr = np.frombuffer(buf, np.float32, count, off).reshape(dims).copy()
        off += count * 4
        tensors[name] = arr
    return MatNet(tensors, d_tok, nhead, num_layers, d_risk, patch_size, caps, arch)


# ---------------------------------------------------------------------------
# Numpy reference forward — the explicit math path (the parity oracle)
# ---------------------------------------------------------------------------


def _linear(x, w, b):
    return x @ w.T + b


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    # matches torch F.softplus (beta=1, threshold=20)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def _layernorm(x, g, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)  # biased, torch convention
    return (x - mu) / np.sqrt(var + eps) * g + b


def _mha(x, pad, w_in, b_in, w_out, b_out, nhead):
    """Multi-head self-attention, one sequence (T, d). ``pad[j]`` True = ignore
    key j. Math path: split QKV, scaled dot, mask to -inf, row-max softmax,
    weighted V, merge heads, out projection."""
    T, d = x.shape
    hd = d // nhead
    qkv = _linear(x, w_in, b_in)  # (T, 3d)
    q, k, v = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
    q = q.reshape(T, nhead, hd).transpose(1, 0, 2)  # (H, T, hd)
    k = k.reshape(T, nhead, hd).transpose(1, 0, 2)
    v = v.reshape(T, nhead, hd).transpose(1, 0, 2)
    scores = (q @ k.transpose(0, 2, 1)) * (1.0 / np.sqrt(hd))  # (H, T, T)
    if pad is not None and pad.any():
        scores = scores + np.where(pad[None, None, :], -np.inf, 0.0)
    scores = scores - scores.max(axis=-1, keepdims=True)  # row-max stable
    e = np.exp(scores)
    p = e / e.sum(axis=-1, keepdims=True)
    ctx = p @ v  # (H, T, hd)
    ctx = ctx.transpose(1, 0, 2).reshape(T, d)  # merge heads
    return _linear(ctx, w_out, b_out)


def _conv2d(x, w, b, stride, pad):
    """Cross-correlation conv (torch Conv2d). x (Cin,H,W), w (Cout,Cin,kH,kW)."""
    Cout, Cin, kH, kW = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Hp, Wp = xp.shape[1], xp.shape[2]
    Ho = (Hp - kH) // stride + 1
    Wo = (Wp - kW) // stride + 1
    out = np.empty((Cout, Ho, Wo), np.float64)
    for oy in range(Ho):
        for ox in range(Wo):
            patch = xp[:, oy * stride : oy * stride + kH, ox * stride : ox * stride + kW]
            out[:, oy, ox] = (w * patch[None]).sum(axis=(1, 2, 3)) + b
    return out


def _adaptive_avg_pool(x, out_hw):
    """torch AdaptiveAvgPool2d: bin i spans [floor(i*H/o), ceil((i+1)*H/o))."""
    C, H, W = x.shape
    oh, ow = out_hw, out_hw
    out = np.empty((C, oh, ow), np.float64)
    for i in range(oh):
        r0, r1 = (i * H) // oh, -(-(i + 1) * H // oh)
        for j in range(ow):
            c0, c1 = (j * W) // ow, -(-(j + 1) * W // ow)
            out[:, i, j] = x[:, r0:r1, c0:c1].mean(axis=(1, 2))
    return out


def _risk_enc(net, patch):
    """RiskPatchEncoder over one (2,P,P) patch -> (d_risk,). Keys net.0/2/4/8."""
    x = patch.astype(np.float64)
    x = _relu(_conv2d(x, net["risk_enc.net.0.weight"], net["risk_enc.net.0.bias"], 1, 1))
    x = _relu(_conv2d(x, net["risk_enc.net.2.weight"], net["risk_enc.net.2.bias"], 2, 1))
    x = _relu(_conv2d(x, net["risk_enc.net.4.weight"], net["risk_enc.net.4.bias"], 2, 1))
    x = _adaptive_avg_pool(x, 4)
    x = x.reshape(-1)  # flatten (C,4,4) row-major, matching torch Flatten
    x = _relu(_linear(x, net["risk_enc.net.8.weight"], net["risk_enc.net.8.bias"]))
    return x


def matnet_forward_numpy(mn: MatNet, obs_feats, obs_mask, goal_feats, risk_patch):
    """The reference forward for one batch item. Shapes: obs_feats (N,6),
    obs_mask (N,) bool, goal_feats (4,), risk_patch (2,P,P). Returns
    (alphas (N,), beta, gamma, lam_soft, lam_hard, mu_lat) — float64.

    This is the explicit math path the C++ twin reproduces (post-norm
    transformer, ReLU FFN, no final encoder norm, masked-softmax)."""
    t = mn.t
    obs_feats = np.asarray(obs_feats, np.float64).reshape(-1, 6)
    N = obs_feats.shape[0]
    goal_feats = np.asarray(goal_feats, np.float64).reshape(4)
    obs_mask = np.asarray(obs_mask, bool).reshape(-1) if N else np.zeros(0, bool)

    # goal token
    zg = _relu(_linear(goal_feats, t["goal_enc.0.weight"], t["goal_enc.0.bias"]))
    zg = _linear(zg, t["goal_enc.2.weight"], t["goal_enc.2.bias"])  # (d,)

    if N > 0:
        zo = _relu(_linear(obs_feats, t["obs_enc.0.weight"], t["obs_enc.0.bias"]))
        zo = _linear(zo, t["obs_enc.2.weight"], t["obs_enc.2.bias"])  # (N,d)
        tokens = np.vstack([zg[None, :], zo])  # (1+N, d)
        pad = np.concatenate([[False], ~obs_mask])  # goal token always valid
    else:
        tokens = zg[None, :]
        pad = np.array([False])

    x = tokens
    for L in range(mn.num_layers):
        pre = f"fuser.layers.{L}."
        attn = _mha(
            x,
            pad,
            t[pre + "self_attn.in_proj_weight"],
            t[pre + "self_attn.in_proj_bias"],
            t[pre + "self_attn.out_proj.weight"],
            t[pre + "self_attn.out_proj.bias"],
            mn.nhead,
        )
        x = _layernorm(x + attn, t[pre + "norm1.weight"], t[pre + "norm1.bias"], mn.eps)
        ff = _linear(
            _relu(_linear(x, t[pre + "linear1.weight"], t[pre + "linear1.bias"])),
            t[pre + "linear2.weight"],
            t[pre + "linear2.bias"],
        )
        x = _layernorm(x + ff, t[pre + "norm2.weight"], t[pre + "norm2.bias"], mn.eps)

    ctx = x[0]  # goal context token
    if N > 0:
        a = _softplus(
            _linear(
                _relu(_linear(x[1:], t["alpha_head.0.weight"], t["alpha_head.0.bias"])),
                t["alpha_head.2.weight"],
                t["alpha_head.2.bias"],
            ).reshape(-1)
        )
        alphas = np.where(obs_mask, a, 0.0)
    else:
        alphas = np.zeros(0)

    def head(prefix, z):
        return _linear(
            _relu(_linear(z, t[prefix + ".0.weight"], t[prefix + ".0.bias"])),
            t[prefix + ".2.weight"],
            t[prefix + ".2.bias"],
        ).reshape(())

    beta = _softplus(head("beta_head", ctx))
    gamma = _softplus(head("gamma_head", ctx))

    risk_ctx = _risk_enc(t, np.asarray(risk_patch, np.float64))
    mat_feats = np.concatenate([risk_ctx, ctx])
    lam_soft = mn.lam_soft_max * _sigmoid(head("lam_soft_head", mat_feats))
    lam_hard = mn.lam_hard_max * _sigmoid(head("lam_hard_head", mat_feats))
    mu_lat = mn.mu_lat_max * _sigmoid(head("mu_lat_head", mat_feats))
    return alphas, float(beta), float(gamma), float(lam_soft), float(lam_hard), float(mu_lat)


__all__ = ["write_matnet", "read_matnet", "matnet_forward_numpy", "MatNet", "arch_hash", "MAGIC"]
