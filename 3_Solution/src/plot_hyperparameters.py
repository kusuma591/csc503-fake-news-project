
import os
import pandas as pd
import matplotlib.pyplot as plt

# Locate results folder
base_dir = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(base_dir, "evaluation", "results")

# Load hyperparameter results
df = pd.read_csv(
    os.path.join(results_dir, "hyperparameter_results.csv")
)

# Create one figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))

# --------------------------------
# (a) SVM hyperparameters
# --------------------------------
svm = df[df["Model"] == "SVM"]

# Different styles make overlapping lines easier to see
styles = {
    "l1": {"linestyle": "-", "marker": "o"},
    "l2": {"linestyle": "--", "marker": "s"}
}

for penalty in ["l1", "l2"]:
    subset = svm[
        svm["Penalty"] == penalty
    ].sort_values("C")

    axes[0].errorbar(
        subset["C"],
        subset["Mean_F1"],
        yerr=subset["Std_F1"],
        linestyle=styles[penalty]["linestyle"],
        marker=styles[penalty]["marker"],
        linewidth=1.5,
        markersize=5,
        capsize=2,
        label=penalty.upper()
    )

axes[0].set_xscale("log")
axes[0].set_xticks([0.001, 0.1, 0.5, 1, 5])
axes[0].set_xticklabels(["0.001", "0.1", "0.5", "1", "5"])
axes[0].set_ylim(0, 1.05)
axes[0].set_xlabel("Regularization parameter C")
axes[0].set_ylabel("Mean F1-score")
axes[0].set_title("(a) Linear SVM")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# --------------------------------
# (b) Naive Bayes hyperparameters
# --------------------------------
nb = df[
    df["Model"] == "Naive Bayes"
].sort_values("Alpha")

axes[1].errorbar(
    nb["Alpha"],
    nb["Mean_F1"],
    yerr=nb["Std_F1"],
    linestyle="-",
    marker="o",
    linewidth=1.5,
    markersize=5,
    capsize=2
)

axes[1].set_xscale("log")
axes[1].set_xticks([0.001, 0.1, 0.5, 1, 5])
axes[1].set_xticklabels(["0.001", "0.1", "0.5", "1", "5"])
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel("Smoothing parameter alpha")
axes[1].set_title("(b) Multinomial Naive Bayes")
axes[1].grid(alpha=0.3)

# --------------------------------
# Save combined figure as PDF only
# --------------------------------
plt.tight_layout()

output_path = os.path.join(
    results_dir,
    "rq2_hyperparameters_combined.pdf"
)

plt.savefig(
    output_path,
    format="pdf",
    bbox_inches="tight"
)

plt.close()

print(f"Graph saved to: {output_path}")
