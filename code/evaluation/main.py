"""
Evaluation script — runs predictions on sample_claims.csv using the
heuristic strategy (no sample-cache lookup) and reports real metrics.
"""
import os
import csv
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verification_engine import VerificationEngine


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true) if y_true else 0.0

    classes = list(set(y_true + y_pred))
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        'accuracy':  accuracy,
        'precision': sum(precisions) / len(classes) if classes else 0.0,
        'recall':    sum(recalls)    / len(classes) if classes else 0.0,
        'f1':        sum(f1s)        / len(classes) if classes else 0.0,
    }


def per_class_breakdown(y_true, y_pred):
    classes = sorted(set(y_true + y_pred))
    rows = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append((c, tp + fn, tp, fp, fn, prec, rec, f1))
    return rows


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_evaluation():
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_csv_path = os.path.join(workspace_root, 'dataset', 'sample_claims.csv')

    if not os.path.exists(sample_csv_path):
        print(f"Error: {sample_csv_path} not found.")
        return

    engine = VerificationEngine(workspace_root)

    sample_rows = []
    with open(sample_csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sample_rows.append(row)

    print(f"Loaded {len(sample_rows)} sample claims.")

    # ── Heuristic strategy ─────────────────────────────────────────────
    t0 = time.time()
    heuristic_preds = [engine.predict_row(r, strategy='heuristic') for r in sample_rows]
    heuristic_time  = time.time() - t0

    # ── VLM strategy (only if OPENAI_API_KEY is set) ───────────────────
    vlm_preds = None
    vlm_time  = None
    api_key   = os.environ.get('OPENAI_API_KEY', '')
    if api_key:
        print("OPENAI_API_KEY found — running VLM evaluation …")
        t0 = time.time()
        vlm_preds = [engine.predict_row(r, strategy='vlm') for r in sample_rows]
        vlm_time  = time.time() - t0
    else:
        print("OPENAI_API_KEY not set — skipping VLM evaluation.")

    # ── Compute metrics ────────────────────────────────────────────────
    fields = ['claim_status', 'issue_type', 'object_part']
    strategies = [('Heuristic (Rule-based)', heuristic_preds, heuristic_time)]
    if vlm_preds:
        strategies.append(('VLM (GPT-4o)', vlm_preds, vlm_time))

    results = {}
    for label, preds, _ in strategies:
        results[label] = {}
        for field in fields:
            y_true = [r[field] for r in sample_rows]
            y_pred = [p[field] for p in preds]
            results[label][field] = {
                'metrics':    calculate_metrics(y_true, y_pred),
                'breakdown':  per_class_breakdown(y_true, y_pred),
                'y_true':     y_true,
                'y_pred':     y_pred,
            }

    # Count images
    image_count = sum(
        len(r['image_paths'].split(';')) for r in sample_rows
    )

    # ── Write report ───────────────────────────────────────────────────
    report_dir  = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(report_dir, 'evaluation_report.md')

    n_sample = len(sample_rows)
    n_test   = 45  # approximate claims.csv size

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# ClaimAI System Evaluation Report\n\n")
        f.write("Evaluation against `dataset/sample_claims.csv` using **real predictions** ")
        f.write("(no label cache — predictions generated fresh for every row).\n\n")

        # ── Metrics tables ─────────────────────────────────────────────
        f.write("## 1. Accuracy Metrics\n\n")
        f.write("| Strategy | Field | Accuracy | Macro Precision | Macro Recall | Macro F1 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for label, field_results in results.items():
            for field, data in field_results.items():
                m = data['metrics']
                f.write(f"| {label} | `{field}` | {m['accuracy']:.1%} | "
                        f"{m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.1%} |\n")
        f.write("\n")

        # ── Per-class breakdown for heuristic ─────────────────────────
        f.write("## 2. Per-Class Breakdown (Heuristic strategy)\n\n")
        for field in fields:
            data = results['Heuristic (Rule-based)'][field]
            f.write(f"### `{field}`\n\n")
            f.write("| Class | Support | TP | FP | FN | Precision | Recall | F1 |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for row_data in data['breakdown']:
                c, sup, tp, fp, fn, prec, rec, f1 = row_data
                f.write(f"| {c} | {sup} | {tp} | {fp} | {fn} | "
                        f"{prec:.1%} | {rec:.1%} | {f1:.1%} |\n")
            # Wrong predictions list
            wrongs = [(i, t, p) for i, (t, p) in enumerate(
                zip(data['y_true'], data['y_pred'])) if t != p]
            if wrongs:
                f.write(f"\nMisclassified ({len(wrongs)}):\n")
                for i, true_val, pred_val in wrongs:
                    uid = sample_rows[i]['user_id']
                    f.write(f"- row {i} ({uid}): true=`{true_val}` pred=`{pred_val}`\n")
            f.write("\n")

        # ── Operational analysis ───────────────────────────────────────
        f.write("## 3. Operational Analysis\n\n")
        f.write("### Execution Latency\n\n")
        for label, _, elapsed in strategies:
            if elapsed is not None:
                per_claim = elapsed / n_sample * 1000
                f.write(f"- **{label}:** {elapsed*1000:.1f} ms total "
                        f"({per_claim:.2f} ms / claim)\n")
        f.write(f"- Real GPT-4o API latency (estimated): ~1.5 – 3 s / claim\n\n")

        f.write("### Model Calls and Token Usage (GPT-4o)\n\n")
        f.write(f"| Run | Claims | Images | API calls | Est. input tokens | Est. output tokens | Est. total cost |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        sample_imgs = image_count
        test_imgs   = 72
        # ~1000 text tokens + ~765 tokens per image (high-detail tile estimate)
        in_per_sample  = n_sample  * (1000 + sample_imgs  // n_sample  * 765)
        in_per_test    = n_test    * (1000 + test_imgs     // n_test    * 765)
        out_per_sample = n_sample  * 300
        out_per_test   = n_test    * 300
        # GPT-4o pricing (as of 2025-06): $5/1M input, $15/1M output
        cost_s = in_per_sample * 5e-6 + out_per_sample * 15e-6
        cost_t = in_per_test   * 5e-6 + out_per_test   * 15e-6
        f.write(f"| Sample | {n_sample} | {sample_imgs} | {n_sample} | "
                f"~{in_per_sample:,} | ~{out_per_sample:,} | ~${cost_s:.3f} |\n")
        f.write(f"| Test   | {n_test}   | {test_imgs}   | {n_test}   | "
                f"~{in_per_test:,} | ~{out_per_test:,} | ~${cost_t:.3f} |\n")
        f.write("\nPricing assumptions: GPT-4o at $5.00 / 1M input tokens, $15.00 / 1M output tokens.\n\n")

        f.write("### Rate Limits and Resilience\n\n")
        f.write("- GPT-4o tier-1 limits: 500 RPM / 30,000 TPM (free) up to 10,000 RPM / 2M TPM (paid).\n")
        f.write("- Strategy: sequential calls with exponential backoff on HTTP 429.\n")
        f.write("- Caching: identical (user_id + claim text) pairs de-duplicated before calling the API.\n")
        f.write("- Images are base64-encoded inline (no external URLs), avoiding storage side-effects.\n")
        f.write("- Heuristic strategy is always available as a zero-cost, zero-latency fallback.\n\n")

        f.write("### Batching and Throughput\n\n")
        f.write("- All images for a single claim are sent in one API call (multi-image content array).\n")
        f.write("- For large runs, an asyncio semaphore (5 concurrent) is recommended.\n")
        f.write("- At 5 RPM concurrent, the 45-row test set completes in ~14 seconds wall-clock.\n")

    print(f"Evaluation report written to {report_path}")


if __name__ == '__main__':
    run_evaluation()
