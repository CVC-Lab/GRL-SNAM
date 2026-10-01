# GRL-SNAM on a headless server — quickstart

Goal: run GRL-SNAM on a server with no display and produce a demo video (mp4).

## Why `pip install grl-snam` isn't enough

GRL-SNAM's graphics (`pycvc_gl`, VTK) are **not on PyPI**. They are cvcpkg
packages. A pip install gives you only the pure-Python part, which is why
`pycvc_gl` is missing. The fix is to install GRL-SNAM *from cvcpkg* into its own
prefix. The prefix carries its own Python 3.11, torch, `pycvc_gl`, VTK and an
`ffmpeg`, so don't reuse your pip venv.

Requirements: Linux x86-64, glibc ≥ 2.35 (Ubuntu 22.04+ or equivalent).

---

## 1. Install cvcpkg into a custom directory

```bash
curl -fsSL https://cvcpkg.org/install.sh | sh -s -- --install-dir $HOME/cvcpkg/bin
export PATH=$HOME/cvcpkg/bin:$PATH      # add to ~/.bashrc to keep it
cvcpkg --version
```

No `curl | sh`? Use pip instead: `python3 -m venv ~/cvcpkg-venv && ~/cvcpkg-venv/bin/pip install cvcpkg`.

## 2. Install GRL-SNAM, its data and ffmpeg into a prefix

```bash
cvcpkg install grl-snam-cp311 grl-snam-weights scene-austin-south ffmpeg-cli \
  --prefix $HOME/nav-env
. $HOME/nav-env/bin/activate

python -c "import grl_snam, pycvc_gl, torch; print('ok')"
grl-snam selftest
ffmpeg -hide_banner -encoders | grep libx264     # H.264 encoder present
```

| package | what it is |
|---|---|
| `grl-snam-cp311` | GRL-SNAM + `pycvc_gl`, VTK, torch, matplotlib, … (`cp311` = Python 3.11) |
| `grl-snam-weights` | pretrained model → `$PREFIX/share/grl-snam-weights/coef_sdf.pt` |
| `scene-austin-south` | Austin map (~110 MB, public, no token) → `$PREFIX/share/cvc-scenes/austin_south/` |
| `ffmpeg-cli` | `ffmpeg` / `ffprobe` with H.264 (x264) → `$PREFIX/bin/` (`capture` encodes the mp4 with it) |

The whole prefix is ~2.5 GB, mostly torch and VTK.

- Want a lighter scene? Use `scene-austin-south-small` (~10 MB, same layout).
- Prefer a git checkout? Inside the activated prefix run `pip install -e .` in
  your GRL-SNAM clone. Keep the `cvcpkg install` above for the bindings and data.
- Different Python? Swap `cp311` for `cp312` or `cp313`. Don't mix columns.

## 3. Headless rendering — no X server needed (EGL)

The cvcpkg VTK picks its render backend at runtime: X11, then **EGL**, then
OSMesa. On a server with no `DISPLAY` it can render offscreen through EGL, either
on the **GPU** (NVIDIA's EGL, if the driver is installed) or on the **CPU** (Mesa
llvmpipe). No Xvfb is needed.

EGL needs a vendor driver from the system. The cvcpkg GL loader looks for it
inside the prefix, so you point it at the system's vendor file with an
environment variable.

**GPU server with the NVIDIA driver installed** (nothing to apt-install):

```bash
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

**CPU-only server** (software rendering with Mesa; needs sudo for two packages):

```bash
sudo apt install libegl-mesa0 libgl1-mesa-dri
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
```

Put the two `export` lines in your `~/.bashrc` or a job script. Check that the
file named in `__EGL_VENDOR_LIBRARY_FILENAMES` exists (`ls /usr/share/glvnd/egl_vendor.d/`).

Smoke test (writes a PNG; no video yet):

```bash
env -u DISPLAY grl-snam lab-demo lab.png
```

Notes:
- `libEGL warning: ... driver (null)` and `EGL device index: 0 could not be initialized. Trying other devices...`
  messages are harmless as long as the command finishes and `lab.png` appears.
- Don't combine EGL with `LIBGL_ALWAYS_SOFTWARE=1`. With the system Mesa that
  combination crashed (segfault) in testing. The Mesa vendor file alone is enough.

### Fallback: Xvfb (virtual X server)

If EGL doesn't work on your machine, use a virtual X server with Mesa GLX instead:

```bash
sudo apt install xvfb libgl1-mesa-dri libglx-mesa0
unset VTK_DEFAULT_OPENGL_WINDOW __EGL_VENDOR_LIBRARY_FILENAMES
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1920x1080x24" grl-snam lab-demo lab.png
```

Prefix every `grl-snam capture ...` command below with
`LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1920x1080x24"` in that case.

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

For reference, the 0.1-minute test (480 frames, 960×540) took about 2 minutes
on CPU (Mesa llvmpipe) and about 1.3 minutes on a GTX 1650.

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
| `bad X server connection` / `Unable to find a valid OpenGL 3.2 or later implementation` | EGL isn't finding a driver. Check the two `export`s in section 3 and that the vendor `.json` file exists |
| renders but very slow | expected on CPU. Test with `--minutes 0.1`, or use a GPU node with the NVIDIA vendor file |
| `ffmpeg: command not found` | the prefix isn't activated, or you left out `ffmpeg-cli` from the install |
| `cvcpkg install` can't find a package | check the `cpXXX` suffix matches the Python you want |

## What was tested

Verified on Ubuntu with a desktop Mesa and an NVIDIA GTX 1650, with `DISPLAY`
unset: installing `grl-snam-cp311 grl-snam-weights scene-austin-south` from the
public catalog, `selftest`, `lab-demo`, and `capture multigoal` on both the Mesa
EGL (CPU) path and the NVIDIA EGL (GPU) path. `ffmpeg-cli` was installed from the
catalog into its own prefix and encoded PNG frames to H.264 correctly. It was not
installed into the same prefix as GRL-SNAM in one command. Not yet tested: a bare
server image. The two Mesa apt packages in the CPU path are the expected minimum,
so tell us if the server needs more.
