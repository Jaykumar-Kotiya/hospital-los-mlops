import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

"""
Creates SHAP explainability plots for the trained model.

Outputs (in reports/figures/):
- shap_summary_bar.png
- shap_summary_beeswarm.png
- shap_waterfall_example.png
- shap_force_example.html
- shap_dependence_top1.png
- shap_dependence_top2.png

Also outputs (in reports/):
- shap_top_features.csv   (Top 20 features by mean absolute SHAP)
"""

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OUTDIR = REPORTS / "figures"


def _load_feature_names(n_features: int):
    """
    Load human-readable feature names produced by preprocessing.
    Priority:
      1) feature_names.npy
      2) feature_names.txt
      3) fallback: f0..f{n-1}
    """
    npy_path = PROCESSED / "feature_names.npy"
    txt_path = PROCESSED / "feature_names.txt"

    if npy_path.exists():
        names = np.load(npy_path, allow_pickle=True).tolist()
        names = [str(x) for x in names]
        if len(names) == n_features:
            return names

    if txt_path.exists():
        names = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]
        if len(names) == n_features:
            return names

    # fallback
    return [f"f{i}" for i in range(n_features)]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Load features
    X = np.load(PROCESSED / "X.npy")
    print(f"Loaded X: {X.shape}")

    # Load newest advanced model
    candidates = sorted(MODELS.glob("xgboost_los_model_advanced_v2_*.joblib"))
    if not candidates:
        candidates = sorted(MODELS.glob("xgboost_los_model_advanced_*.joblib"))
    if not candidates:
        raise FileNotFoundError("No advanced joblib model found in models/. Run training first.")

    model_path = candidates[-1]
    print(f"Using model: {model_path.name}")
    model = joblib.load(model_path)

    # Load feature names
    feature_names = _load_feature_names(n_features=X.shape[1])
    if (PROCESSED / "feature_names.txt").exists() or (PROCESSED / "feature_names.npy").exists():
        print(f"Loaded {len(feature_names)} feature names.")
    else:
        print("Feature names not found → using f0..fN fallback.")

    # Sample subset to keep SHAP fast
    n = min(2000, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    X_sample = X[idx]

    # SHAP for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # =============================
    # ✅ NEW: Save top feature table
    # =============================
    mean_abs = np.abs(shap_values).mean(axis=0)  # global importance
    top_k = 20
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    top_table = pd.DataFrame({
        "rank": np.arange(1, top_k + 1),
        "feature": [feature_names[i] for i in top_idx],
        "mean_abs_shap": [float(mean_abs[i]) for i in top_idx],
    })

    csv_path = REPORTS / "shap_top_features.csv"
    top_table.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # 1) Bar summary (global importance)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    bar_path = OUTDIR / "shap_summary_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"Saved: {bar_path}")

    # 2) Beeswarm summary (distribution + direction)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    bee_path = OUTDIR / "shap_summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(bee_path, dpi=200)
    plt.close()
    print(f"Saved: {bee_path}")

    # -----------------------------
    # BONUS: single-patient explanation
    # -----------------------------
    sample_local_idx = 0
    print(f"Creating per-patient plots for X_sample[{sample_local_idx}]...")

    # Waterfall plot
    try:
        exp = shap.Explanation(
            values=shap_values[sample_local_idx],
            base_values=explainer.expected_value,
            data=X_sample[sample_local_idx],
            feature_names=feature_names,
        )
        shap.plots.waterfall(exp, show=False, max_display=15)
        wf_path = OUTDIR / "shap_waterfall_example.png"
        plt.tight_layout()
        plt.savefig(wf_path, dpi=200)
        plt.close()
        print(f"Saved: {wf_path}")
    except Exception as e:
        print(f"Waterfall plot skipped due to: {e}")

    # Force plot (HTML)
    try:
        force = shap.force_plot(
            explainer.expected_value,
            shap_values[sample_local_idx],
            X_sample[sample_local_idx],
            feature_names=feature_names,
            matplotlib=False,
        )
        html_path = OUTDIR / "shap_force_example.html"
        shap.save_html(str(html_path), force)
        print(f"Saved: {html_path}")
    except Exception as e:
        print(f"Force plot HTML skipped due to: {e}")

    # -----------------------------
    # BONUS: dependence plots for top 2 features
    # -----------------------------
    try:
        top2_idx = np.argsort(mean_abs)[::-1][:2]

        for j, fi in enumerate(top2_idx, start=1):
            plt.figure()
            shap.dependence_plot(
                int(fi),
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
            )
            dep_path = OUTDIR / f"shap_dependence_top{j}.png"
            plt.tight_layout()
            plt.savefig(dep_path, dpi=200)
            plt.close()
            print(f"Saved: {dep_path}")
    except Exception as e:
        print(f"Dependence plots skipped due to: {e}")

    print("SHAP explainability done ✅")
    print(f"All outputs saved under: {OUTDIR}")


if __name__ == "__main__":
    main()
