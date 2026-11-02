#!/usr/bin/env python
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from media_utils import rms, pick_segment, mix_at_snr, windows16

rng = np.random.default_rng(0)
fg = 0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
bg = rng.normal(0, 0.05, 32000).astype(np.float32)

seg = pick_segment(bg, 16000, rng)
assert seg.shape == (16000,)
short = pick_segment(bg[:1000], 16000, rng)          # tiling path
assert short.shape == (16000,)

m0 = mix_at_snr(fg, bg, 0.0)
m20 = mix_at_snr(fg, bg, 20.0)
assert m0.shape == fg.shape and np.abs(m0).max() <= 0.95 + 1e-6
# residual after projecting out fg = the added noise; 0 dB noise is ~10x the +20 dB noise
lvl0 = rms(m0 - fg * (np.dot(m0, fg) / np.dot(fg, fg)))
lvl20 = rms(m20 - fg * (np.dot(m20, fg) / np.dot(fg, fg)))
assert lvl0 > 3 * lvl20, (lvl0, lvl20)

emb = rng.normal(size=(50, 96)).astype(np.float32)
w = windows16(emb, stride=8)
assert w.shape == ((50 - 16) // 8 + 1, 16, 96)
assert np.allclose(w[1], emb[8:24])
print("PASS test_media_utils")
