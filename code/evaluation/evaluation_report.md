# ClaimAI System Evaluation Report

Evaluation against `dataset/sample_claims.csv` using **real predictions** (no label cache — predictions generated fresh for every row).

## 1. Accuracy Metrics

| Strategy | Field | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|
| Heuristic (Rule-based) | `claim_status` | 60.0% | 32.7% | 44.9% | 37.8% |
| Heuristic (Rule-based) | `issue_type` | 45.0% | 34.7% | 37.5% | 33.3% |
| Heuristic (Rule-based) | `object_part` | 70.0% | 62.0% | 63.9% | 61.9% |

## 2. Per-Class Breakdown (Heuristic strategy)

### `claim_status`

| Class | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| contradicted | 5 | 0 | 0 | 5 | 0.0% | 0.0% | 0.0% |
| not_enough_information | 2 | 1 | 2 | 1 | 33.3% | 50.0% | 40.0% |
| supported | 13 | 11 | 6 | 2 | 64.7% | 84.6% | 73.3% |

Misclassified (8):
- row 4 (user_005): true=`contradicted` pred=`supported`
- row 5 (user_006): true=`not_enough_information` pred=`supported`
- row 7 (user_008): true=`contradicted` pred=`supported`
- row 13 (user_020): true=`contradicted` pred=`supported`
- row 15 (user_030): true=`supported` pred=`not_enough_information`
- row 16 (user_031): true=`supported` pred=`not_enough_information`
- row 18 (user_033): true=`contradicted` pred=`supported`
- row 19 (user_034): true=`contradicted` pred=`supported`

### `issue_type`

| Class | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| broken_part | 3 | 1 | 0 | 2 | 100.0% | 33.3% | 50.0% |
| crack | 3 | 2 | 1 | 1 | 66.7% | 66.7% | 66.7% |
| crushed_packaging | 1 | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| dent | 3 | 3 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| glass_shatter | 0 | 0 | 1 | 0 | 0.0% | 0.0% | 0.0% |
| missing_part | 0 | 0 | 2 | 0 | 0.0% | 0.0% | 0.0% |
| none | 2 | 0 | 0 | 2 | 0.0% | 0.0% | 0.0% |
| scratch | 2 | 1 | 1 | 1 | 50.0% | 50.0% | 50.0% |
| stain | 1 | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| torn_packaging | 1 | 0 | 1 | 1 | 0.0% | 0.0% | 0.0% |
| unknown | 3 | 0 | 3 | 3 | 0.0% | 0.0% | 0.0% |
| water_damage | 1 | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |

Misclassified (11):
- row 3 (user_007): true=`broken_part` pred=`unknown`
- row 4 (user_005): true=`scratch` pred=`unknown`
- row 5 (user_006): true=`unknown` pred=`crack`
- row 7 (user_008): true=`broken_part` pred=`scratch`
- row 10 (user_011): true=`stain` pred=`water_damage`
- row 12 (user_018): true=`crack` pred=`glass_shatter`
- row 13 (user_020): true=`none` pred=`unknown`
- row 15 (user_030): true=`torn_packaging` pred=`missing_part`
- row 17 (user_032): true=`unknown` pred=`missing_part`
- row 18 (user_033): true=`unknown` pred=`crushed_packaging`
- row 19 (user_034): true=`none` pred=`torn_packaging`

### `object_part`

| Class | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| box | 0 | 0 | 2 | 0 | 0.0% | 0.0% | 0.0% |
| contents | 1 | 1 | 1 | 0 | 50.0% | 100.0% | 66.7% |
| corner | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| door | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| front_bumper | 2 | 1 | 0 | 1 | 100.0% | 50.0% | 66.7% |
| headlight | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| hinge | 1 | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| hood | 0 | 0 | 1 | 0 | 0.0% | 0.0% | 0.0% |
| keyboard | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| package_corner | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| package_side | 1 | 0 | 1 | 1 | 0.0% | 0.0% | 0.0% |
| rear_bumper | 2 | 2 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| screen | 2 | 2 | 1 | 0 | 66.7% | 100.0% | 80.0% |
| seal | 2 | 0 | 0 | 2 | 0.0% | 0.0% | 0.0% |
| side_mirror | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| trackpad | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| unknown | 1 | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% |
| windshield | 1 | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% |

Misclassified (6):
- row 7 (user_008): true=`front_bumper` pred=`hood`
- row 9 (user_010): true=`hinge` pred=`screen`
- row 15 (user_030): true=`seal` pred=`contents`
- row 16 (user_031): true=`package_side` pred=`box`
- row 18 (user_033): true=`unknown` pred=`package_side`
- row 19 (user_034): true=`seal` pred=`box`

## 3. Operational Analysis

### Execution Latency

- **Heuristic (Rule-based):** 8562.0 ms total (428.10 ms / claim)
- Real GPT-4o API latency (estimated): ~1.5 – 3 s / claim

### Model Calls and Token Usage (GPT-4o)

| Run | Claims | Images | API calls | Est. input tokens | Est. output tokens | Est. total cost |
|---|---|---|---|---|---|---|
| Sample | 20 | 29 | 20 | ~35,300 | ~6,000 | ~$0.267 |
| Test   | 45   | 72   | 45   | ~79,425 | ~13,500 | ~$0.600 |

Pricing assumptions: GPT-4o at $5.00 / 1M input tokens, $15.00 / 1M output tokens.

### Rate Limits and Resilience

- GPT-4o tier-1 limits: 500 RPM / 30,000 TPM (free) up to 10,000 RPM / 2M TPM (paid).
- Strategy: sequential calls with exponential backoff on HTTP 429.
- Caching: identical (user_id + claim text) pairs de-duplicated before calling the API.
- Images are base64-encoded inline (no external URLs), avoiding storage side-effects.
- Heuristic strategy is always available as a zero-cost, zero-latency fallback.

### Batching and Throughput

- All images for a single claim are sent in one API call (multi-image content array).
- For large runs, an asyncio semaphore (5 concurrent) is recommended.
- At 5 RPM concurrent, the 45-row test set completes in ~14 seconds wall-clock.
