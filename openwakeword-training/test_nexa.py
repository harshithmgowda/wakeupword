import json
import os
import sys
import time
import numpy as np
import onnxruntime as ort

def run_evaluation():
    print("=" * 60)
    print("Starting Comprehensive Evaluation for 'Hey Nexa' Model")
    print("=" * 60)

    # 1. Load ONNX model
    onnx_path = "my_custom_model/hey_nexa.onnx"
    if not os.path.exists(onnx_path):
        print(f"Error: {onnx_path} not found!")
        sys.exit(1)

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX Model: {onnx_path}")
    print(f"Input Name: {input_name}, Shape: {session.get_inputs()[0].shape}")
    print(f"Output Name: {output_name}, Shape: {session.get_outputs()[0].shape}")

    # 2. Load held-out test datasets
    pos_test_path = "my_custom_model/hey_nexa/positive_features_test.npy"
    neg_test_path = "my_custom_model/hey_nexa/negative_features_test.npy"
    val_fp_path = "validation_set_features.npy"

    pos_data = np.load(pos_test_path)
    neg_data = np.load(neg_test_path)
    print(f"Loaded {pos_data.shape[0]} positive test examples, shape {pos_data.shape}")
    print(f"Loaded {neg_data.shape[0]} negative test examples, shape {neg_data.shape}")

    # 3. Predict on positive test set
    t0 = time.perf_counter()
    pos_preds = session.run([output_name], {input_name: pos_data.astype(np.float32)})[0].squeeze()
    pos_latency = (time.perf_counter() - t0) / len(pos_data) * 1000

    # Predict on negative test set
    neg_preds = session.run([output_name], {input_name: neg_data.astype(np.float32)})[0].squeeze()

    # 4. Predict on false-positive validation set (11.3 hours)
    val_fp_hours = 11.3
    val_preds = None
    if os.path.exists(val_fp_path):
        val_data = np.load(val_fp_path)
        window = pos_data.shape[1]  # 16
        num_windows = val_data.shape[0] - window
        print(f"Evaluating on {num_windows} validation windows (~{val_fp_hours:.1f} hours of continuous background audio)...")
        val_preds_list = []
        batch_sz = 8192
        for start_idx in range(0, num_windows, batch_sz):
            end_idx = min(start_idx + batch_sz, num_windows)
            batch = np.array([val_data[i:i + window] for i in range(start_idx, end_idx)], dtype=np.float32)
            preds = session.run([output_name], {input_name: batch})[0].squeeze()
            if preds.ndim == 0:
                preds = np.array([preds])
            val_preds_list.append(preds)
        val_preds = np.concatenate(val_preds_list)

    # 5. Evaluate across thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    eval_results = []
    print("\n" + "-" * 85)
    print(f"{'Threshold':<10}{'TP':<6}{'FN (Miss)':<10}{'FP':<6}{'TN':<6}{'Recall':<10}{'Precision':<12}{'FA/Hour (11.3h)':<15}")
    print("-" * 85)

    for th in thresholds:
        tp = int(np.sum(pos_preds >= th))
        fn = int(np.sum(pos_preds < th))
        fp = int(np.sum(neg_preds >= th))
        tn = int(np.sum(neg_preds < th))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if val_preds is not None:
            val_fps = int(np.sum(val_preds >= th))
            fa_per_hr = val_fps / val_fp_hours
        else:
            fa_per_hr = None

        eval_results.append({
            "threshold": th,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "recall": round(recall, 4),
            "precision": round(prec, 4),
            "false_activations_per_hour": round(fa_per_hr, 2) if fa_per_hr is not None else "N/A"
        })
        fa_str = f"{fa_per_hr:.2f}" if fa_per_hr is not None else "N/A"
        print(f"{th:<10.2f}{tp:<6}{fn:<10}{fp:<6}{tn:<6}{recall:<10.2%}{prec:<12.2%}{fa_str:<15}")

    print("-" * 85)
    print(f"Single-frame desktop inference latency: {pos_latency:.3f} ms / frame (Real-time budget: 80.0 ms)")

    out_metrics = {
        "model_name": "hey_nexa",
        "positive_test_count": len(pos_data),
        "negative_test_count": len(neg_data),
        "validation_hours": val_fp_hours,
        "single_frame_latency_ms": round(pos_latency, 3),
        "threshold_evaluations": eval_results
    }
    with open("evaluation.json", "w") as f:
        json.dump(out_metrics, f, indent=2)
    print("Saved metrics to evaluation.json")

if __name__ == '__main__':
    run_evaluation()

