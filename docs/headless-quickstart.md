# GRL-SNAM on a headless server — quickstart

Goal: run GRL-SNAM on a server with no display and produce a demo video (mp4).

## Why `pip install grl-snam` isn't enough

GRL-SNAM's graphics (`pycvc_gl`, VTK) are **not on PyPI**. They are cvcpkg
packages. A pip install gives you only the pure-Python part, which is why
`pycvc_gl` is missing. The fix is to install GRL-SNAM *from cvcpkg* into its own
prefix. The prefix carries its own Python 3.11, torch, `pycvc_gl`, VTK, a software OpenGL
(Mesa), a virtual X server (Xvfb) and an `ffmpeg`, so it should need **no
system GL or X packages and no sudo** (see "What was tested" at the end). Don't
reuse your pip venv.

Requirements: Linux x86-64, glibc ≥ 2.35 (Ubuntu 22.04+ or equivalent).

---

## 1. Install cvcpkg into a custom directory

```bash
curl -fsSL https://cvcpkg.org/install.sh | sh -s -- --install-dir $HOME/cvcpkg/bin
export PATH=$HOME/cvcpkg/bin:$PATH      # add to ~/.bashrc to keep it
cvcpkg --version
```

No `curl | sh`? Use pip instead: `python3 -m venv ~/cvcpkg-venv && ~/cvcpkg-venv/bin/pip install cvcpkg`.

## 2. Install GRL-SNAM, its data, OpenGL and ffmpeg into a prefix

```bash
cvcpkg install grl-snam-cp311 grl-snam-weights scene-austin-south \
  ffmpeg-cli mesa xvfb --prefix $HOME/nav-env
. $HOME/nav-env/bin/activate

python3.11 -c "import grl_snam, pycvc_gl, torch; print('ok')"   # python3.12 / python3.13 for those columns
grl-snam selftest
ffmpeg -hide_banner -encoders | grep libx264     # H.264 encoder present
```

| package | what it is |
|---|---|
| `grl-snam-cp311` | GRL-SNAM + `pycvc_gl`, VTK, torch, matplotlib, … (`cp311` = Python 3.11) |
| `grl-snam-weights` | pretrained model → `$PREFIX/share/grl-snam-weights/coef_sdf.pt` |
| `scene-austin-south` | Austin map (~110 MB, public, no token) → `$PREFIX/share/cvc-scenes/austin_south/` |
| `ffmpeg-cli` | `ffmpeg` / `ffprobe` with H.264 (x264) → `$PREFIX/bin/` (`capture` encodes the mp4 with it) |
| `mesa` | software OpenGL (llvmpipe), with EGL and GLX → `$PREFIX/lib/` |
| `xvfb` | virtual X server + `xvfb-run` → `$PREFIX/bin/` (only needed for the Xvfb path in section 3) |

The whole prefix is ~3 GB, mostly torch, VTK and Mesa's LLVM.

- Want a lighter scene? Use `scene-austin-south-small` (~10 MB, same layout).
- Prefer a git checkout? Inside the activated prefix run `pip install -e .` in
  your GRL-SNAM clone. Keep the `cvcpkg install` above for the bindings and data.
- Different Python? Swap `cp311` for `cp312` or `cp313`. Don't mix columns.

## 3. Headless rendering — no display needed

The cvcpkg VTK picks its render backend at runtime: X11, then **EGL**, then
OSMesa. On a server with no `DISPLAY` there are two ways to render, and both run
entirely from the prefix. Use **EGL** (simplest). Use Xvfb only if EGL gives you
trouble.

### Option A: EGL (recommended)

```bash
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
```

That is the only setting needed. `mesa` in the prefix provides the EGL driver and
the prefix finds it by itself. Put the line in your `~/.bashrc` or job script.

Smoke test (writes a PNG; no video yet):

```bash
env -u DISPLAY grl-snam lab-demo lab.png
```

**GPU server with the NVIDIA driver installed:** to render on the GPU instead of
the CPU, also point EGL at the system's NVIDIA vendor file:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

(Check the file exists with `ls /usr/share/glvnd/egl_vendor.d/`.)

Notes:
- `libEGL warning: ... driver (null)` and `EGL device index: 0 could not be initialized. Trying other devices...`
  messages are harmless as long as the command finishes and `lab.png` appears.
- Don't set `LIBGL_ALWAYS_SOFTWARE=1` with EGL. It crashes (an upstream Mesa bug
  in the EGL device path). The default already falls back to software.

### Option B: Xvfb (virtual X server)

`xvfb-run` starts a virtual X server from the prefix for the length of one
command, and the OpenGL goes through Mesa's GLX:

```bash
unset VTK_DEFAULT_OPENGL_WINDOW __EGL_VENDOR_LIBRARY_FILENAMES
xvfb-run -a -s "-screen 0 1920x1080x24" grl-snam lab-demo lab.png
```

Prefix every `grl-snam capture ...` command below with
`xvfb-run -a -s "-screen 0 1920x1080x24"` in that case.

## 4. Make a demo video

```bash
BUNDLE=$HOME/nav-env/share/cvc-scenes/austin_south
CKPT=$HOME/nav-env/share/grl-snam-weights/coef_sdf.pt

# short test first
grl-snam capture multigoal $BUNDLE $CKPT --minutes 0.1 -o drive.mp4

# full run: multi-goal free-drive, ~3 min of video (15 fps)
grl-snam capture multigoal $BUNDLE $CKPT --minutes 3 -o multigoal.mp4
```

The video shows the real 3-D Austin scene with a live HUD: the network's
predicted coefficients (α, β, γ), wall clearance, speed and mode. Rendered PNG
frames are kept in `_frames/` next to your `-o` file.

For reference, the 0.1-minute test (480 frames, 960×540) took about 1 minute 45
seconds on CPU (Mesa llvmpipe, EGL or Xvfb) and about 1 minute 15 seconds on a
GTX 1650 through NVIDIA EGL.

Other commands:

```bash
# a single fixed A->B run
grl-snam capture drive $BUNDLE $CKPT --start X Y --goal X Y -o drive.mp4

# whole chain: world model -> SDF -> train -> video
grl-snam pipeline $BUNDLE

# train your own model instead of the pretrained one
grl-snam build-sdf $BUNDLE                          # -> nav_sdf.npz
grl-snam train nav_sdf.npz -o coef_sdf.pt --steps 1500
grl-snam capture multigoal $BUNDLE coef_sdf.pt -o mine.mp4
```

## What does NOT work headless

- `grl-snam demo NAME` (the live interactive window). Use `capture` for video.
- `grl-snam demo lab` needs VolRover3.

## Troubleshooting

| symptom | fix |
|---|---|
| `import pycvc_gl` fails | you're not in the prefix. Run `. $HOME/nav-env/bin/activate` |
| `bad X server connection` / `Unable to find a valid OpenGL 3.2 or later implementation` | the prefix isn't activated, `mesa` wasn't installed, or (EGL) `VTK_DEFAULT_OPENGL_WINDOW` isn't set. With the NVIDIA vendor file, check that the `.json` exists |
| renders but very slow | expected on CPU. Test with `--minutes 0.1`, or use a GPU node with the NVIDIA vendor file |
| `ffmpeg: command not found` | the prefix isn't activated, or you left out `ffmpeg-cli` from the install |
| `xvfb-run: command not found` | the prefix isn't activated, or `xvfb` wasn't installed (only needed for Option B) |
| `cvcpkg install` can't find a package | check the `cpXXX` suffix matches the Python you want |

## What was tested

Verified on Ubuntu with an NVIDIA GTX 1650, with `DISPLAY` unset, installing from
the public catalog into a fresh prefix: `grl-snam-cp311`, `grl-snam-weights`,
`scene-austin-south`, `ffmpeg-cli`, `mesa` and `xvfb`.
- `selftest`, `lab-demo`, and `capture multigoal` (480 frames, 960×540, H.264) all
  work on **EGL with the prefix's Mesa**, and on **Xvfb with the prefix's Mesa GLX**.
  A library-load trace shows only prefix copies of Mesa, LLVM and the GL loaders
  being used, with no system Mesa.
- With the NVIDIA vendor file set, EGL loads NVIDIA's driver instead of Mesa.

Not yet tested: a bare server image (this machine has a desktop OS installed).
glibc 2.35 or newer is required.
