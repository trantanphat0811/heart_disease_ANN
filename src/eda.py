"""Quick EDA script: outputs summary statistics and basic plots into reports/"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run(data_path: str = "data/synthetic_heart_disease_dataset.csv", out_dir: str = "reports"):
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    # basic info
    desc = df.describe(include="all")
    desc.to_csv(out_dir / "summary_statistics.csv")

    # target distribution
    if "Heart_Disease" in df.columns:
        counts = df["Heart_Disease"].value_counts()
        counts.to_csv(out_dir / "target_counts.csv")
        plt.figure()
        sns.countplot(x=df["Heart_Disease"])
        plt.title("Heart_Disease distribution")
        plt.savefig(out_dir / "target_distribution.png", dpi=150)
        plt.close()

    # numeric histograms (sample a subset if too many columns)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=False)
        plt.title(f"Histogram: {col}")
        plt.savefig(out_dir / f"hist_{col}.png", dpi=150)
        plt.close()

    # correlation heatmap for numeric features
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm")
        plt.title("Numeric feature correlation")
        plt.tight_layout()
        plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

    print(f"EDA outputs written to {out_dir}")


if __name__ == "__main__":
    run()
