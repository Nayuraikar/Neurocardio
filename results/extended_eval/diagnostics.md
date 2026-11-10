# Noise / XAI diagnostics

Clean: macroF1 0.4727 Vrec 0.8922 Srec 0.0965 acc 0.9061

## 1. Post-z-score noise, high SNR
| SNR | macroF1 | V recall |
|---|---|---|
| 60 | 0.4731 | 0.8934 |
| 40 | 0.4756 | 0.8875 |
| 30 | 0.2701 | 0.0672 |

## 2. snr_24 detail
- noise sd used 0.06310; expected 10**(-24/20) = 0.06310; empirical std of noise 0.06310; mean per-window std of clean 1.0000 (std-of-stds 7.62e-08)
- clean windows: mean 3.30e-11, std 1.0000; noisy pre-renorm: mean -2.75e-06, std 1.0020; after renorm: mean -2.67e-11, std 1.0000
- empirical SNR 24.00 dB
- pred histogram clean [42020, 503, 4555, 440], snr_24 [4, 11263, 0, 36251], true [42149, 1782, 3200, 387]

## 3. Post-z-score vs pipeline-faithful (noise on raw, then 0.5-40 Hz bandpass, window, z-score)
SNR vs raw record power (std of raw signal incl. baseline offset/drift).

| SNR | post: macroF1 | post: V rec | raw-faithful: macroF1 | raw-faithful: V rec | raw-faithful: S rec |
|---|---|---|---|---|---|
| 24 | 0.0484 | 0.0000 | 0.4750 | 0.8925 | 0.0932 |
| 18 | 0.0234 | 0.0000 | 0.4817 | 0.8847 | 0.0976 |
| 12 | 0.0125 | 0.0000 | 0.4839 | 0.8622 | 0.0909 |
| 6 | 0.0040 | 0.0000 | 0.4535 | 0.7419 | 0.0769 |

## 4. Training augmentation noise
- gaussian_noise_sd_on_zscored_window = 0.03 on unit-variance windows -> SNR 30.5 dB (3% of signal std). Eval SNR 24 dB = sd 0.063 (2.1x train); 6 dB = sd 0.501 (17x train).

## 5. XAI deletion: mean prob drop (n=200, clean mean P 0.9421)
Zero = replace with 0 (window mean); Interp = linear interpolation across deleted samples. Contig = random contiguous block of equal length; Scatter = original random-k.

| % | CAM zero | Contig zero | CAM interp | Contig interp | Scatter interp |
|---|---|---|---|---|---|
| 5 | 0.2062 | 0.0195 | 0.0233 | 0.0195 | -0.0001 |
| 10 | 0.3520 | 0.0430 | 0.0439 | 0.0338 | -0.0007 |
| 20 | 0.4741 | 0.1000 | 0.0522 | 0.1346 | 0.0007 |
| 30 | 0.5919 | 0.0895 | 0.0487 | 0.0960 | 0.0005 |
| 50 | 0.7369 | 0.0729 | 0.0558 | 0.0840 | 0.0020 |

## 6. MoRF vs LeRF (interp replacement, attribution smoothed 15 samples, n=200)
Mean prob drop. MoRF = delete highest-attribution first; LeRF = lowest first. Faithful => MoRF >> LeRF.

| % | CAM MoRF | CAM LeRF | IG MoRF | IG LeRF |
|---|---|---|---|---|
| 5 | 0.0351 | 0.0005 | 0.1164 | -0.0140 |
| 10 | 0.0445 | 0.0002 | 0.1679 | -0.0205 |
| 20 | 0.0599 | 0.0040 | 0.2365 | -0.0251 |
| 30 | 0.0660 | 0.0171 | 0.2576 | -0.0236 |
| 50 | 0.0739 | 0.0283 | 0.2849 | -0.0232 |

## 7. Augmentation (from preprocessing_config.json)
- Classes augmented: S (2 copies incl. original), F (4 copies incl. original); N, V none. Gaussian sd 0.03 on z-scored window; also amp scale U(0.9,1.1), baseline wander amp U(0,0.1) 0.15-0.5 Hz, circular shift +-15 samples. Whether the noise is applied to every copy or the original: NOT FOUND.
