# GRL-SNAM on a headless server — quickstart

Goal: run GRL-SNAM on a server with no display and produce a demo video (mp4).

## Why `pip install grl-snam` isn't enough

GRL-SNAM's graphics (`pycvc_gl`, VTK) are **not on PyPI**. They are cvcpkg
packages. A pip install gives you only the pure-Python part, which is why
`pycvc_gl` is missing. The fix is to install GRL-SNAM *from cvcpkg* into its own
prefix. The prefix carries its own Python 3.11, torch and `pycvc_gl`, so don't
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

## 2. Install GRL-SNAM + dependencies into a prefix

```bash
cvcpkg install grl-snam-cp311 grl-snam-weights scene-austin-south --prefix $HOME/nav-env
. $HOME/nav-env/bin/activate

python -c "import grl_snam, pycvc_gl, torch; print('ok')"
grl-snam selftest
```

| package | what it is |
|---|---|
| `grl-snam-cp311` | GRL-SNAM + `pycvc_gl`, VTK, torch, matplotlib, … (`cp311` = Python 3.11) |
| `grl-snam-weights` | pretrained model → `$PREFIX/share/grl-snam-weights/coef_sdf.pt` |
| `scene-austin-south` | Austin map (~110 MB, public, no token) → `$PREFIX/share/cvc-scenes/austin_south/` |

- Want a lighter scene? Use `scene-austin-south-small` (~10 MB, same layout).
- Prefer a git checkout? Inside the activated prefix run `pip install -e .` in
  your GRL-SNAM clone. Keep the `cvcpkg install` above for the bindings and data.
- Different Python? Swap `cp311` for `cp312` or `cp313`. Don't mix columns.

## 3. System packages for headless rendering (needs sudo)

The cvcpkg VTK renders through X11/GLX (not EGL/OSMesa), so a server with no
display needs a **virtual X server** (Xvfb), a **software GL driver** (Mesa) and
an **`ffmpeg`** binary (`grl-snam capture` shells out to `ffmpeg -c:v libx264`
to encode the mp4). cvcpkg ships the GL *loader* but not these, so for now they
come from apt:

```bash
sudo apt install xvfb libgl1-mesa-dri libglx-mesa0 ffmpeg
```

Check that ffmpeg has H.264 encoding (Ubuntu's does):

```bash
ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264     # must print a line
```

> We are adding Xvfb, Mesa and an ffmpeg CLI as cvcpkg packages so this whole
> step can move into the prefix. Until then, use apt as above.

Smoke test. This writes a PNG, so no video is involved yet:

```bash
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1920x1080x24" grl-snam lab-demo lab.png
```

If that produces `lab.png`, the graphics stack works.

## 4. Make a demo video

```bash
BUNDLE=$HOME/nav-env/share/cvc-scenes/austin_south
CKPT=$HOME/nav-env/share/grl-snam-weights/coef_sdf.pt
export LIBGL_ALWAYS_SOFTWARE=1

# short test first: software GL is slow
xvfb-run -a -s "-screen 0 1920x1080x24" \
  grl-snam capture multigoal $BUNDLE $CKPT --minutes 0.5 -o drive.mp4

# full run: multi-goal free-drive, ~3 min of video (15 fps)
xvfb-run -a -s "-screen 0 1920x1080x24" \
  grl-snam capture multigoal $BUNDLE $CKPT --minutes 3 -o multigoal.mp4
```

The video shows the real 3-D Austin scene with a live HUD: the network's
predicted coefficients (α, β, γ), wall clearance, speed and mode.

Other commands (wrap each in `xvfb-run` the same way):

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
| GLX / `BadValue` / "cannot open display" | run under `xvfb-run`, and check `xvfb`, `libgl1-mesa-dri` and `libglx-mesa0` are installed |
| `ffmpeg: not found` or no `libx264` | install `ffmpeg` (section 3). Rendered frames are kept in `_frames/` next to your `-o` output, so nothing is lost |
| very slow rendering | expected with software GL. Test with `--minutes 0.5` |
| `cvcpkg install` can't find a package | check the `cpXXX` suffix matches the Python you want |
