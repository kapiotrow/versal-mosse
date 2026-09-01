"""Layer-1 banks for the offline screen: 7x7 stride-2, 32 channels.

Both are returned as (w_float[32,3,7,7], b_fold[32]) so they go through the
SAME quantize()/conv_features() integer path as the shipping bank -- the
comparison has to be between banks, not between code paths.
"""
import os

import numpy as np

K, STRIDE, NCH = 7, 2, 32


_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'build', 'l1_resnet18_pca32.npz')


def resnet18_conv1_pca(n_out=NCH):
    """resnet18 conv1 is 7x7 stride 2, 64 filters -- Danelljan's Layer-1 geometry
    exactly (VGG-M conv1 is 7x7/2/96; Table 1's 224->109 fixes it). BatchNorm is
    folded as export_weights.py does. Reduced 64 -> 32 by PCA over the FOLDED
    WEIGHT matrix.

    HONEST LABEL: this is not Danelljan's PCA. He compresses ACTIVATIONS (96->40);
    projecting weights composes with the conv exactly only while the map stays
    LINEAR, so for the ReLU arm this is "a 32-filter bank spanning the principal
    directions of resnet18 conv1", not an activation PCA.
    """
    # Cached: 62 parallel workers each instantiating resnet18 is minutes of
    # pure waste, and the bank is a deterministic function of the checkpoint.
    # The cache MUST be keyed on n_out -- an unkeyed one silently returns the
    # 32-channel bank for a 16-channel arm, and the arm then measures nothing
    # it claims to.
    cache = _CACHE.replace('pca32', f'pca{n_out}')
    if os.path.exists(cache):
        z = np.load(cache)
        return z['w'], z['b']
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    w = m.conv1.weight.detach().numpy().astype(np.float64)      # [64,3,7,7]
    bn = m.bn1
    g = bn.weight.detach().numpy().astype(np.float64)
    b = bn.bias.detach().numpy().astype(np.float64)
    mu = bn.running_mean.detach().numpy().astype(np.float64)
    var = bn.running_var.detach().numpy().astype(np.float64)
    sd = np.sqrt(var + float(bn.eps))
    wf = w * (g / sd)[:, None, None, None]
    bf = b - g * mu / sd
    flat = wf.reshape(wf.shape[0], -1)                          # [64,147]
    mean = flat.mean(axis=0)
    u, sv, vt = np.linalg.svd(flat - mean, full_matrices=False)
    comp = vt[:n_out]                                           # [32,147]
    # Scale each component to the mean folded-filter norm so out_shift and
    # bias_acc see a comparable grid (the -rand arms' rule).
    comp *= np.linalg.norm(flat, axis=1).mean() / np.linalg.norm(comp, axis=1)[:, None]
    w_out, b_out = comp.reshape(n_out, 3, K, K), np.zeros(n_out)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, w=w_out, b=b_out)
    return w_out, b_out


def gabor_bank():
    """16 analytic filters + their NEGATIONS = 32 channels.

    12 oriented luminance Gabors (6 orientations x 2 phases) + 4 colour/blob
    filters. Rectified quadrature pairs ARE orientation energy, which is HOG's
    core computation done by convolution -- the mechanism the literature credits,
    without a donor network. The negations make the rectifier lossless: with
    ReLU, (max(v,0), max(-v,0)) spans {v, |v|}.
    """
    LUM = np.array([0.2989, 0.5870, 0.1140])
    y, x = np.mgrid[-(K // 2):K // 2 + 1, -(K // 2):K // 2 + 1].astype(np.float64)
    sig, lam = 2.0, 4.0
    filt = []
    for th in np.arange(6) * np.pi / 6.0:
        xr = x * np.cos(th) + y * np.sin(th)
        yr = -x * np.sin(th) + y * np.cos(th)
        env = np.exp(-(xr ** 2 + 0.7 * yr ** 2) / (2 * sig ** 2))
        for ph in (0.0, np.pi / 2):
            g = env * np.cos(2 * np.pi * xr / lam + ph)
            filt.append(('lum', g - g.mean()))
    lp = np.exp(-(x ** 2 + y ** 2) / (2 * 2.0 ** 2)); lp /= lp.sum()
    log = np.exp(-(x ** 2 + y ** 2) / (2 * 1.2 ** 2))
    log = log / log.sum() - lp
    filt += [('rg', lp), ('by', lp), ('lum', log), ('lum', lp - lp.mean())]
    w = np.zeros((16, 3, K, K))
    for i, (kind, g) in enumerate(filt):
        if kind == 'lum':   w[i] = g[None] * LUM[:, None, None]
        elif kind == 'rg':  w[i] = g[None] * np.array([1.0, -1.0, 0.0])[:, None, None]
        else:               w[i] = g[None] * np.array([-0.5, -0.5, 1.0])[:, None, None]
    n = np.linalg.norm(w.reshape(16, -1), axis=1)
    w /= np.where(n > 0, n, 1.0)[:, None, None, None]
    return np.concatenate([w, -w], axis=0), np.zeros(32)


def vgg16_conv1_pca(n_out=16):
    """VGG16 conv1: 3x3, 64 filters, stride 1 -- a stand-in for Danilowicz &
    Kryjak 2022's stem, which is VGG11 conv1 INCLUDING ReLU AND 2x2 MAXPOOL
    (their sec.4.1). vgg11 is not in the local torch cache; vgg16_bn is, and its
    conv1 plays the identical role (3x3/64 first layer of the same family).
    Label it as the stand-in it is.

    THEIR TABLE 1, VOT2015, 128x128 ROI with a 64x64 filter -- i.e. exactly the
    res64 geometry:

        32 channels  A 0.494  R 1.92   EAO 0.184
         8 channels  A 0.491  R 2.082  EAO 0.183
        16 channels  A 0.487  R 1.975  EAO 0.174
         4 channels  A 0.456  R 2.611  EAO 0.145
        DSST (HOG)   A 0.54   R 2.56   EAO 0.17
        KCF  (HOG)   A 0.48   R 2.17   EAO 0.17

    **8 channels ties 32.** The collapse is at 4. So widening the bank is not
    where their result comes from, and a 16-channel Layer-1 arm costs this
    project NOTHING downstream -- N_CHANNELS is unchanged, so FFT, cmul, IFFT
    and the whole APU tail stay exactly as they are and only conv2d moves.
    """
    import torch
    from torchvision.models import vgg16_bn, VGG16_BN_Weights
    cache = _CACHE.replace('resnet18_pca32', f'vgg16_pca{n_out}')
    if os.path.exists(cache):
        z = np.load(cache)
        return z['w'], z['b']
    m = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).eval()
    conv, bn = m.features[0], m.features[1]
    w = conv.weight.detach().numpy().astype(np.float64)          # [64,3,3,3]
    g = bn.weight.detach().numpy().astype(np.float64)
    b = bn.bias.detach().numpy().astype(np.float64)
    mu = bn.running_mean.detach().numpy().astype(np.float64)
    sd = np.sqrt(bn.running_var.detach().numpy().astype(np.float64) + float(bn.eps))
    wf = w * (g / sd)[:, None, None, None]
    flat = wf.reshape(wf.shape[0], -1)
    u, sv, vt = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
    comp = vt[:n_out]
    comp *= np.linalg.norm(flat, axis=1).mean() / np.linalg.norm(comp, axis=1)[:, None]
    w_out, b_out = comp.reshape(n_out, 3, 3, 3), np.zeros(n_out)
    np.savez(cache, w=w_out, b=b_out)
    return w_out, b_out


def resnet18_conv1_5x5(n_out=NCH):
    """resnet18 conv1 CENTRE-CROPPED to 5x5, for a stride-1 arm at the 128x128 map.

    WHY 5x5 AND WHY STRIDE 1. Hardware measured the 64x64 map as the WORSE
    geometry at matched sigma/target (-0.0222 R, proposed_build_res64.md sec.25),
    so every arm in the Layer-1 screen ran at the losing end. Stride 1 keeps the
    128x128 map, and 5x5 is the largest kernel that stays FRAME-RATE NEUTRAL
    there: conv2d = S + t*taps with S = 2.31 ms and t = 0.255 ms/tap from the two
    measured points (rgb.md: 4.60 ms at 9 taps, 9.19 at 27), so 75 taps is
    ~21.4 ms against a 24.55 ms host tail and still hides. 7x7 stride 1 is
    ~39.8 ms and would flip the frame to AIE-bound.

    The crop discards the outer ring of a learned 7x7. That changes the filters
    and it is not free -- but they stay LEARNED, which the Gabor diagnostic
    showed is the property that matters: an analytic oriented bank LOSES when
    rectified (R 0.3518 -> 0.3292) where both learned banks gain.
    """
    w, b = resnet18_conv1_pca(n_out)          # [n,3,7,7], already folded + PCA
    c = w[:, :, 1:6, 1:6].copy()              # central 5x5
    n_old = np.linalg.norm(w.reshape(w.shape[0], -1), axis=1)
    n_new = np.linalg.norm(c.reshape(c.shape[0], -1), axis=1)
    c *= (n_old / np.where(n_new > 0, n_new, 1.0))[:, None, None, None]
    return c, b
