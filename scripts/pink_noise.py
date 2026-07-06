"""Pink-noise initial-latent patch for Paint3D's diffusers pipelines.

Method 4 (see .claude/dissertation-plan.md): "It's Never Too Late: Noise
Optimization for Collapse Recovery in Trained Diffusion Models" (Harrington
et al., arXiv 2601.00090) finds that pink noise -- enhanced low-frequency
content relative to standard white Gaussian noise -- gives more diverse
samples than white noise even with NO optimization, before any of the
paper's own noise-optimization step. This module only implements that
noise-space swap: FFT the initial latent noise, reweight each frequency
component by 1/(1+f)^alpha (f = radial frequency, in cycles across the
latent tile), inverse-FFT, and renormalize to unit variance so downstream
code (which expects standard-normal statistics) is unaffected. alpha=0.2
is the paper's chosen operating point (higher alpha increases diversity
further but visibly degrades quality above ~0.3).

Paint3D's three diffusers pipeline wrappers (txt2img / inpaint / img2img
controlnet, in external/paint3d/controlnet/) never pass an explicit
`latents=` tensor -- they let diffusers sample the initial noise internally
via `diffusers.utils.torch_utils.randn_tensor`, called through each
pipeline module's own top-level `randn_tensor` name (bound at import time).
Patching only `diffusers.utils.torch_utils.randn_tensor` would miss those
already-bound references, so `install_pink_noise` patches the name in each
pipeline module directly instead.

This module is invoked from inside Paint3D's own subprocesses (see the
opt-in hook added to external/paint3d/pipeline_paint3d_stage{1,2}.py, guarded
by the PINK_NOISE_ALPHA env var), not imported directly by
scripts/generate_crowd_baseline.py -- the Paint3D stages run as separate
`sys.executable` subprocesses (see run_stage()), so a patch installed in the
driver's own process would never reach them.
"""

import torch
from diffusers.utils.torch_utils import randn_tensor as _orig_randn_tensor


def _pink_filter(white, alpha):
    """Reweight a (B, C, H, W) white-noise tensor's 2D spectrum by
    1/(1+f)^alpha, then renormalize to zero mean / unit std per (B, C)
    slice so it matches the statistics randn_tensor's caller expects."""
    orig_dtype = white.dtype
    x = white.float()  # torch.fft needs float32+; pipelines may run fp16
    _, _, h, w = x.shape

    freq_h = torch.fft.fftfreq(h, device=x.device) * h
    freq_w = torch.fft.fftfreq(w, device=x.device) * w
    uu, vv = torch.meshgrid(freq_h, freq_w, indexing="ij")
    f = torch.sqrt(uu ** 2 + vv ** 2)
    weight = 1.0 / (1.0 + f) ** alpha  # (H, W), broadcasts over (B, C, H, W)

    spectrum = torch.fft.fft2(x, norm="ortho") * weight
    pink = torch.fft.ifft2(spectrum, norm="ortho").real

    mean = pink.mean(dim=(-2, -1), keepdim=True)
    std = pink.std(dim=(-2, -1), keepdim=True)
    pink = (pink - mean) / std.clamp_min(1e-8)
    return pink.to(orig_dtype)


def make_pink_randn_tensor(alpha):
    def _pink_randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
        white = _orig_randn_tensor(shape, generator=generator, device=device, dtype=dtype, layout=layout)
        if len(shape) != 4:
            return white  # only the (B, C, H, W) latent case is pink-filtered
        return _pink_filter(white, alpha)
    return _pink_randn_tensor


def install_pink_noise(alpha):
    """Monkeypatch randn_tensor in-place in every diffusers pipeline module
    Paint3D uses, so its unmodified controlnet wrappers draw pink instead of
    white initial-latent noise. Safe to call once per process; idempotent
    (re-patches with the same replacement each time)."""
    patched = make_pink_randn_tensor(alpha)
    import diffusers.pipelines.controlnet.pipeline_controlnet as _txt2img_mod
    import diffusers.pipelines.controlnet.pipeline_controlnet_inpaint as _inpaint_mod
    import diffusers.pipelines.controlnet.pipeline_controlnet_img2img as _img2img_mod
    _txt2img_mod.randn_tensor = patched
    _inpaint_mod.randn_tensor = patched
    _img2img_mod.randn_tensor = patched
    print(f"[pink_noise] installed pink-noise initial-latent patch (alpha={alpha})", flush=True)
