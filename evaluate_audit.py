"""
Observational Audit and Discovery Pipeline
===========================================
Research Thesis: "Integration of Deep Learning Models for Detection and 
                  Characterization of Exoplanets via Gravitational Microlensing"
Author: Gedion Melese Ashebo
Institution: Adama Science and Technology University (ASTU)

This script performs the scientific audit on archival observational data 
(OGLE-IV and KMTNet) using the trained PINN architecture.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import medfilt

# Set TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==============================================================================
# 1. PARSING & PRE-PROCESSING
# ==============================================================================
def robust_parse(file_content):
    """
    Parses raw telescope photometric time-series.
    Converts observational magnitudes to linear flux: F = 10^(-0.4 * m).
    """
    if isinstance(file_content, bytes):
        lines = file_content.decode('utf-8', errors='ignore').split('\n')
    else:
        lines = file_content.split('\n')

    t_list, f_list = [], []
    for line in lines:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(nums) >= 2:
            try:
                t_list.append(float(nums[0]))
                # Convert magnitude to linear flux
                f_list.append(10.0 ** (-0.4 * float(nums[1])))
            except ValueError:
                continue

    return np.array(t_list), np.array(f_list)


def preprocess_light_curve(t, flux):
    """
    Applies rank-order median filtering (k=5), min-max standardization,
    and linear interpolation to a uniform 1024-point grid.
    """
    # 1. Non-linear median filter to remove impulsive CCD outliers
    flux_filtered = medfilt(flux, kernel_size=5)

    # 2. Min-max normalization (centering to training manifold)
    f_min, f_max = np.min(flux_filtered), np.max(flux_filtered)
    f_std = (flux_filtered - f_min) / (f_max - f_min + 1e-8)

    # 3. Time normalization and uniform grid interpolation (1024 points)
    t_norm = (t - np.min(t)) / (np.max(t) - np.min(t)) * 100.0
    f_input = np.interp(np.linspace(0.0, 100.0, 1024), t_norm, f_std)

    return f_input


# ==============================================================================
# 2. SCIENTIFIC AUDIT ENGINE
# ==============================================================================
def run_scientific_audit(data_dir, model_path, output_csv):
    """
    Evaluates archival light curves against the trained PINN model and
    applies the two-tiered discovery gate to prioritize discovery purity.
    """
    print(f" Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please train the model first using train.py or specify a valid path.")
        return

    model = tf.keras.models.load_model(model_path, compile=False)
    results = []

    # Get list of target files
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"⚠️ Warning: Data directory '{data_dir}' is empty or not found.")
        print("Creating and testing on 1 synthetic demonstration event...")
        os.makedirs(data_dir, exist_ok=True)
        demo_file = os.path.join(data_dir, "P01_DEMO_KMT_2019_0842.txt")
        t_dummy = np.linspace(8600, 8660, 200)
        m_dummy = 18.0 - 2.5 * np.exp(-((t_dummy - 8626) / 5)**2)
        with open(demo_file, "w") as f:
            for t_val, m_val in zip(t_dummy, m_dummy):
                f.write(f"{t_val:.4f} {m_val:.4f}\n")

    file_list = sorted(os.listdir(data_dir))
    print(f"🚀 Running scientific audit on {len(file_list)} targets...\n")

    for f_name in file_list:
        file_path = os.path.join(data_dir, f_name)
        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            t, flux = robust_parse(content)
            if len(t) < 10:
                continue

            f_input = preprocess_light_curve(t, flux)

            # Neural network inference
            pd_prob, pr_phys = model.predict(f_input.reshape(1, 1024, 1), verbose=0)
            discovery_prob = float(pd_prob[0, 0])

            # Physical parameter decoding (consistent with training manifold)
            # Mass Ratio (q) and Projected Separation (s)
            q_inferred = 10.0 ** (float(pr_phys[0, 2]) * 2.0 - 4.5)
            s_inferred = float(pr_phys[0, 3]) * 0.6 + 0.7
            peakiness = float(np.max(f_input) / (np.mean(f_input) + 1e-8))

            # Principled Tiered Discovery Gate (High Purity)
            is_detected = (discovery_prob > 0.48) or (discovery_prob > 0.44 and peakiness > 20.0)

            # Ground-truth evaluation based on target label prefix (P = Planet, S = Stellar)
            is_planetary_ground_truth = f_name.upper().startswith("P")
            verdict = "CORRECT" if (is_detected == is_planetary_ground_truth) else "ERROR"

            results.append({
                "Target": f_name,
                "Prob": f"{discovery_prob * 100.0:.2f}%",
                "Mass_q": f"{q_inferred:.2e}",
                "Sep_s": f"{s_inferred:.3f}",
                "Peak": round(peakiness, 2),
                "Status": "PLANET" if is_detected else "STELLAR",
                "Verdict": verdict
            })
        except Exception as e:
            print(f"Skipping {f_name} due to read error: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        print("=" * 85)
        print("               TARGET ANALYSIS RESULTS (TABLE 4.2)")
        print("=" * 85)
        print(df.to_string(index=False))
        print("=" * 85)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Audit complete. Results saved to: {output_csv}")
    else:
        print("No valid light curves could be processed.")


# ==============================================================================
# 3. MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Scientific Audit on Archival Microlensing Data")
    parser.add_argument("--data_dir", type=str, default="./data/audit_sample", help="Directory containing target light curves")
    parser.add_argument("--model_path", type=str, default="./results/models/pinn_microlensing_final.h5", help="Path to trained model")
    parser.add_argument("--output_csv", type=str, default="./results/audit/target_analysis_results.csv", help="Output CSV path")
    args, _ = parser.parse_known_args()

    run_scientific_audit(args.data_dir, args.model_path, args.output_csv)


if __name__ == "__main__":
    main()
