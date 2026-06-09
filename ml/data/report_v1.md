# ML Model Validation Report — v1

**Generated:** 2026-05-17T08:11:29.053742+00:00

## Summary

| Metric | Value |
|--------|-------|
| Correct | 42/45 |
| Accuracy | 93.3% |
| False Positives | 7 |
| Missed | 3 |

## Model Info

| Property | Value |
|----------|-------|
| Mode | finetune |
| Epochs | 80 |
| Best Epoch | 80 |
| Val Accuracy | 1.0 |
| Classes | idle, jab, cross, hook, uppercut, walking |
| Dataset Size | 1384 windows |

## Per-Video Results

| Video | Expected | Detected | Correct | FP | Missed | Status |
|-------|----------|----------|---------|-----|--------|--------|
| jab.mp4 | 17 jab | 16 | 15 | 1 | 2 | PASS (misclassified: {'uppercut': 1}) |
| cross.mp4 | 9 cross | 12 | 9 | 3 | 0 | PASS (misclassified: {'uppercut': 1, 'hook': 1}) |
| hook.mp4 | 10 hook | 13 | 10 | 3 | 0 | FAIL |
| uppercut.mp4 | 9 uppercut | 8 | 8 | 0 | 1 | PASS |
| walking_back_forth.mp4 | 0 walking | 0 | 0 | 0 | 0 | PASS |
| idling_notwalking.mp4 | 0 idle | 0 | 0 | 0 | 0 | PASS |

## Detection Details

### jab.mp4 (jab)
Expected: 17, Detected: 16, Breakdown: {'jab': 15, 'uppercut': 1}

| Time | Move | Confidence | Velocity | Reason |
|------|------|------------|----------|--------|
| 1.40s | JAB | 1.00 | 0.1221 | ML |
| 2.03s | JAB | 1.00 | 0.0720 | ML |
| 2.60s | JAB | 1.00 | 0.1948 | ML |
| 3.70s | JAB | 1.00 | 0.1245 | ML |
| 4.37s | UPPERCUT | 0.75 | 0.1057 | VEL:y-dominant |
| 5.10s | JAB | 1.00 | 0.1266 | ML |
| 5.83s | JAB | 1.00 | 0.1498 | ML |
| 6.60s | JAB | 1.00 | 0.1452 | ML |
| 7.23s | JAB | 1.00 | 0.1167 | ML |
| 7.70s | JAB | 1.00 | 0.1383 | ML |
| 8.20s | JAB | 1.00 | 0.2079 | ML |
| 9.00s | JAB | 1.00 | 0.1412 | ML |
| 9.60s | JAB | 1.00 | 0.0731 | ML |
| 10.43s | JAB | 1.00 | 0.1856 | ML |
| 11.43s | JAB | 1.00 | 0.1186 | ML |
| 12.83s | JAB | 1.00 | 0.0425 | ML |

### cross.mp4 (cross)
Expected: 9, Detected: 12, Breakdown: {'cross': 10, 'uppercut': 1, 'hook': 1}

| Time | Move | Confidence | Velocity | Reason |
|------|------|------------|----------|--------|
| 0.13s | CROSS | 1.00 | 0.1298 | ML |
| 2.03s | CROSS | 1.00 | 0.0710 | ML |
| 2.80s | CROSS | 1.00 | 0.1553 | ML |
| 3.83s | UPPERCUT | 0.75 | 0.1241 | VEL:y-dominant |
| 5.03s | HOOK | 0.80 | 0.1789 | VEL:z-dominant |
| 6.47s | CROSS | 1.00 | 0.1823 | ML |
| 7.33s | CROSS | 1.00 | 0.1327 | ML |
| 7.77s | CROSS | 1.00 | 0.1183 | ML |
| 8.63s | CROSS | 1.00 | 0.0961 | ML |
| 9.87s | CROSS | 1.00 | 0.0769 | ML |
| 10.80s | CROSS | 1.00 | 0.1385 | ML |
| 12.07s | CROSS | 1.00 | 0.2279 | ML |

### hook.mp4 (hook)
Expected: 10, Detected: 13, Breakdown: {'hook': 13}

| Time | Move | Confidence | Velocity | Reason |
|------|------|------------|----------|--------|
| 0.60s | HOOK | 1.00 | 0.0982 | ML |
| 2.77s | HOOK | 1.00 | 0.1591 | ML |
| 4.10s | HOOK | 1.00 | 0.1421 | ML |
| 5.63s | HOOK | 1.00 | 0.1274 | ML |
| 7.20s | HOOK | 1.00 | 0.1485 | ML |
| 8.97s | HOOK | 1.00 | 0.1692 | ML |
| 10.23s | HOOK | 1.00 | 0.2354 | ML |
| 11.87s | HOOK | 1.00 | 0.1866 | ML |
| 13.70s | HOOK | 1.00 | 0.1946 | ML |
| 15.03s | HOOK | 1.00 | 0.1867 | ML |
| 16.43s | HOOK | 1.00 | 0.0928 | ML |
| 17.90s | HOOK | 1.00 | 0.1795 | ML |
| 19.13s | HOOK | 1.00 | 0.0990 | ML |

### uppercut.mp4 (uppercut)
Expected: 9, Detected: 8, Breakdown: {'uppercut': 8}

| Time | Move | Confidence | Velocity | Reason |
|------|------|------------|----------|--------|
| 2.37s | UPPERCUT | 1.00 | 0.1644 | VEL:y-dominant |
| 4.03s | UPPERCUT | 1.00 | 0.2211 | ML:default |
| 6.43s | UPPERCUT | 1.00 | 0.1645 | ML:default |
| 9.03s | UPPERCUT | 1.00 | 0.1527 | ML:default |
| 10.47s | UPPERCUT | 1.00 | 0.1896 | ML:default |
| 11.90s | UPPERCUT | 1.00 | 0.1869 | ML:default |
| 13.67s | UPPERCUT | 1.00 | 0.0548 | ML:default |
| 15.13s | UPPERCUT | 1.00 | 0.0815 | ML:default |

### walking_back_forth.mp4 (walking)
Expected: 0, Detected: 0, Breakdown: {}

| Time | Move | Confidence | Velocity | Reason |
|------|------|------------|----------|--------|

### idling_notwalking.mp4 (idle)
Expected: 0, Detected: 0, Breakdown: {}

| Time | Move | Confidence | Velocity | Reason |
|------|------|------------|----------|--------|
