# LLM Pandemic Policy DCE Analysis (Extended)
# -----------------------------------------------------------
# Adds multi-factor comparisons (state, role, county_type, demographic_context,
# capacity_context, trade_off_frame, model), preserves missing as "None",
# and splits "good" vs "bad" choices with normalized visuals.

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_palette("husl")

OUTPUT_DIR = "analysis_output"
INPUT_DIR = "extended_county_results"
INPUT_PATTERN = "extended_results_*.csv"

CATEGORICAL_VARS = [
    "state",
    "role",
    "county_type",
    "demographic_context",
    "capacity_context",
    "trade_off_frame",
    "model",
]

NPI_LABELS = {0: "No Closure", 1: "Crowded Closures", 2: "Non-Essential Closures"}

# -----------------------
# Helpers
# -----------------------

def ensure_output_dir(d=OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)
    return d

def read_all_csv(directory=INPUT_DIR, pattern=INPUT_PATTERN):
    # Robust CSV read:
    # - keep_default_na=True allows standard NA parsing
    # - na_values left default for typical NA tokens
    # - dtype=str first, then coerce numerics for numeric columns
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found matching {pattern} in {directory}")

    print(f"Found {len(files)} CSV files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp, keep_default_na=True)  # standard NA parsing [pandas docs]
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def coerce_and_tag_categories(df: pd.DataFrame) -> pd.DataFrame:
    # Map NaN to explicit "None" for all analysis-driving categoricals. [pandas categorical]
    for col in CATEGORICAL_VARS:
        if col in df.columns:
            df[col] = df[col].astype("object")
            df[col] = df[col].where(~df[col].isna(), "None")
            df[col] = df[col].astype("category")

    # numeric coercions
    num_cols = [
        "infection_rate_selected",
        "infection_rate_non_selected",
        "npi_level_selected",
        "npi_level_non_selected",
        "temperature",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # add derived fields
    df["is_bad_choice"] = (
        (df["infection_rate_selected"] > df["infection_rate_non_selected"])
        & (df["npi_level_selected"] >= df["npi_level_non_selected"])
    )
    df["is_good_choice"] = ~df["is_bad_choice"]

    df["infection_diff"] = (
        df["infection_rate_selected"] - df["infection_rate_non_selected"]
    )
    df["npi_diff"] = df["npi_level_selected"] - df["npi_level_non_selected"]

    # human-readable NPI labels for selected/non-selected
    df["npi_label_selected"] = df["npi_level_selected"].map(NPI_LABELS)
    df["npi_label_non_selected"] = df["npi_level_non_selected"].map(NPI_LABELS)

    return df

def summarize_overall(df: pd.DataFrame) -> dict:
    stats = {
        "mean_infection_selected": df["infection_rate_selected"].mean(),
        "mean_infection_non_selected": df["infection_rate_non_selected"].mean(),
        "median_infection_selected": df["infection_rate_selected"].median(),
        "median_infection_non_selected": df["infection_rate_non_selected"].median(),
        "bad_choices_count": int(df["is_bad_choice"].sum()),
        "bad_choices_pct": 100.0 * df["is_bad_choice"].mean(),
        "good_choices_count": int(df["is_good_choice"].sum()),
        "good_choices_pct": 100.0 * df["is_good_choice"].mean(),
        "total_choices": int(len(df)),
    }
    return stats

def summarize_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    # For each category in group_col, compute:
    # - counts
    # - infection mean/median (selected vs non)
    # - deltas
    # - bad/good choice counts and %
    # - distribution of selected NPI levels
    gb = df.groupby(group_col, dropna=False)

    base = gb.agg(
        total_choices=("id", "count"),
        mean_infection_selected=("infection_rate_selected", "mean"),
        mean_infection_non_selected=("infection_rate_non_selected", "mean"),
        median_infection_selected=("infection_rate_selected", "median"),
        median_infection_non_selected=("infection_rate_non_selected", "median"),
        bad_choices_count=("is_bad_choice", "sum"),
        good_choices_count=("is_good_choice", "sum"),
    ).reset_index()

    base["bad_choices_pct"] = 100.0 * base["bad_choices_count"] / base["total_choices"]
    base["good_choices_pct"] = 100.0 * base["good_choices_count"] / base["total_choices"]
    base["infection_diff"] = (
        base["mean_infection_selected"] - base["mean_infection_non_selected"]
    )

    # NPI distribution (selected)
    for npi in [0, 1, 2]:
        pct_col = f"npi_{npi}_selected_pct"
        base[pct_col] = (
            gb.apply(lambda g: (g["npi_level_selected"] == npi).mean() * 100.0)
            .reset_index(level=0, drop=True)
            .reindex(base[group_col])
            .values
        )

    return base

def write_csv(df: pd.DataFrame, name: str):
    ensure_output_dir()
    out = os.path.join(OUTPUT_DIR, name)
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    return out

# -----------------------
# Visuals
# -----------------------

def plot_overall(df: pd.DataFrame):
    ensure_output_dir()
    # Infection distribution (selected vs non-selected)
    melted = pd.DataFrame({
        "infection_rate": pd.concat([df["infection_rate_selected"], df["infection_rate_non_selected"]], ignore_index=True),
        "type": ["Selected"] * len(df) + ["Not Selected"] * len(df),
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=melted, x="infection_rate", hue="type", bins=25, ax=axes[0], stat="count", common_norm=False)
    axes[0].set_title("Infection Rates: Selected vs Not Selected")
    axes[0].set_xlabel("Infections per 100")
    axes[0].set_ylabel("Count")

    means = melted.groupby("type")["infection_rate"].mean()
    sns.barplot(x=means.index, y=means.values, ax=axes[1], palette=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Mean Infection Rate Comparison")
    axes[1].set_ylabel("Mean Infections per 100")
    axes[1].set_xlabel("")
    for i, v in enumerate(means.values):
        axes[1].text(i, v + 0.3, f"{v:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "overall_infection_rate_analysis_ext.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

    # NPI levels (selected vs non-selected)
    npi_long = pd.DataFrame({
        "npi_label": pd.concat([df["npi_label_selected"], df["npi_label_non_selected"]], ignore_index=True),
        "type": ["Selected"] * len(df) + ["Not Selected"] * len(df),
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=npi_long,
        x="npi_label",
        hue="type",
        order=["No Closure", "Crowded Closures", "Non-Essential Closures"],
        ax=ax,
    )
    ax.set_title("NPI Level Distribution (Selected vs Not Selected)")
    ax.set_xlabel("NPI Level")
    ax.set_ylabel("Count")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, "overall_npi_level_analysis_ext.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

def plot_group_comparisons(df_stats: pd.DataFrame, group_col: str, prefix: str):
    ensure_output_dir()

    # Sort by infection_diff for ranking
    stats = df_stats.sort_values("infection_diff", ascending=True)
    # 1) Mean infection selected vs non-selected by group
    fig, ax = plt.subplots(figsize=(14, max(6, 0.3 * len(stats))))
    x = np.arange(len(stats))
    width = 0.4
    ax.barh(x - width/2, stats["mean_infection_selected"], height=width, label="Selected", color="#2ecc71", alpha=0.85)
    ax.barh(x + width/2, stats["mean_infection_non_selected"], height=width, label="Not Selected", color="#e74c3c", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(stats[group_col].astype(str))
    ax.set_xlabel("Mean Infections per 100")
    ax.set_title(f"Mean Infection Rates by {group_col}")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"{prefix}_{group_col}_mean_infection.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

    # 2) Bad choices % by group
    stats_bad = stats.sort_values("bad_choices_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(14, max(6, 0.3 * len(stats_bad))))
    bars = ax.barh(stats_bad[group_col].astype(str), stats_bad["bad_choices_pct"], color="#e74c3c", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Bad Choices (%)")
    ax.set_title(f'Bad Choices by {group_col} (Higher infection + Equal/More restrictive NPI)')
    ax.grid(axis="x", alpha=0.3)
    for b, pct, cnt in zip(bars, stats_bad["bad_choices_pct"], stats_bad["bad_choices_count"]):
        ax.text(b.get_width(), b.get_y() + b.get_height()/2, f" {pct:.1f}% (n={int(cnt)})", va="center")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"{prefix}_{group_col}_bad_choices.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

    # 3) NPI selected heatmap per group (percentage)
    npi_heat = stats[[group_col, "npi_0_selected_pct", "npi_1_selected_pct", "npi_2_selected_pct"]].copy()
    npi_heat.columns = [group_col, "No Closure", "Crowded Closures", "Non-Essential Closures"]
    npi_heat = npi_heat.set_index(group_col)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(npi_heat))))
    sns.heatmap(npi_heat, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Selected (%)"}, ax=ax)
    ax.set_title(f"NPI Level Distribution (Selected) by {group_col}")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"{prefix}_{group_col}_npi_heatmap.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

    # 4) Infection preference (infection_diff)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(stats))))
    colors = ["#2ecc71" if v < 0 else "#e74c3c" for v in stats["infection_diff"]]
    bars = ax.barh(stats[group_col].astype(str), stats["infection_diff"], color=colors, alpha=0.8, edgecolor="black")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Infection Diff (Selected - Not Selected)")
    ax.set_title(f"Infection Preference by {group_col} (Negative = Lower Infection Preference)")
    ax.grid(axis="x", alpha=0.3)
    for b, diff in zip(bars, stats["infection_diff"]):
        x = b.get_width()
        ax.text(x + (0.1 if x > 0 else -0.1), b.get_y() + b.get_height()/2, f"{diff:.2f}", ha="left" if x > 0 else "right", va="center")
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"{prefix}_{group_col}_infection_preference.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

def plot_good_vs_bad_split(df: pd.DataFrame, group_col: str, prefix: str):
    ensure_output_dir()
    # Per group, compare mean infection, mean NPI, and counts for good vs bad subsets
    sub = df.copy()
    sub["choice_type"] = np.where(sub["is_bad_choice"], "Bad", "Good")

    # Mean infection selected by group & choice_type
    stats_inf = (
        sub.groupby([group_col, "choice_type"])
        .agg(mean_infection_selected=("infection_rate_selected", "mean"),
             mean_infection_non_selected=("infection_rate_non_selected", "mean"),
             mean_npi_selected=("npi_level_selected", "mean"))
        .reset_index()
    )

    # 1) Mean selected infection by choice_type per group
    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(stats_inf[group_col].unique()))))
    sns.barplot(
        data=stats_inf,
        x="mean_infection_selected",
        y=group_col,
        hue="choice_type",
        ax=ax
    )
    ax.set_title(f"Mean Selected Infection by {group_col} and Choice Type")
    ax.set_xlabel("Mean Infections per 100")
    ax.set_ylabel(group_col)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"{prefix}_{group_col}_good_bad_mean_infection_selected.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

    # 2) Mean selected NPI level by choice_type per group
    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(stats_inf[group_col].unique()))))
    sns.barplot(
        data=stats_inf,
        x="mean_npi_selected",
        y=group_col,
        hue="choice_type",
        ax=ax
    )
    ax.set_title(f"Mean Selected NPI Level by {group_col} and Choice Type")
    ax.set_xlabel("Mean NPI Level (0,1,2)")
    ax.set_ylabel(group_col)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"{prefix}_{group_col}_good_bad_mean_npi_selected.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

    # 3) Share of Bad vs Good by group (normalized)
    share = (
        sub.groupby([group_col, "choice_type"])
        .size()
        .groupby(level=0)
        .apply(lambda s: s / s.sum())
        .reset_index(name="share")
    )
    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(share[group_col].unique()))))
    sns.barplot(
        data=share,
        x="share",
        y=group_col,
        hue="choice_type",
        ax=ax
    )
    ax.set_title(f"Share of Bad vs Good Choices by {group_col}")
    ax.set_xlabel("Share")
    ax.set_ylabel(group_col)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, f"{prefix}_{group_col}_good_bad_share.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close()

# -----------------------
# Tables
# -----------------------

def print_group_table(df_stats: pd.DataFrame, group_col: str):
    print("\n" + "="*80)
    print(f"{group_col.upper()} COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Group':<30} {'Sel Mean':>10} {'NonSel Mean':>12} {'Diff':>8} {'Bad%':>8} {'Good%':>8} {'n':>8}")
    print("-"*86)
    for _, r in df_stats.iterrows():
        print(f"{str(r[group_col]):<30} "
              f"{r['mean_infection_selected']:>10.2f} "
              f"{r['mean_infection_non_selected']:>12.2f} "
              f"{r['infection_diff']:>8.2f} "
              f"{r['bad_choices_pct']:>8.1f} "
              f"{r['good_choices_pct']:>8.1f} "
              f"{int(r['total_choices']):>8d}")
    print("\n" + "-"*86)

# -----------------------
# Main
# -----------------------

def main():
    print("Step 1: Combine CSV files...")
    df = read_all_csv()
    ensure_output_dir()
    df.to_csv(os.path.join(OUTPUT_DIR, "all_data_combined.csv"), index=False)
    print("Saved combined dataset.")

    print("\nStep 2: Coerce categories and derive fields...")
    df = coerce_and_tag_categories(df)
    df.to_csv(os.path.join(OUTPUT_DIR, "all_data_typed.csv"), index=False)
    print("Saved typed dataset with derived fields.")

    print("\nStep 3: Overall summary and visuals...")
    overall = summarize_overall(df)
    pd.Series(overall).to_csv(os.path.join(OUTPUT_DIR, "overall_summary.csv"))
    plot_overall(df)

    # Per-variable stats, visuals, and tables
    print("\nStep 4: Grouped comparisons...")
    for col in CATEGORICAL_VARS:
        if col not in df.columns:
            continue
        print(f" - Grouping by {col}...")
        stats = summarize_by(df, col)
        write_csv(stats, f"{col}_comparison_stats.csv")
        print_group_table(stats, col)
        plot_group_comparisons(stats, col, prefix="comparisons")
        plot_good_vs_bad_split(df, col, prefix="comparisons")

    # Save explicit bad/good choices
    bad = df[df["is_bad_choice"]].copy()
    good = df[df["is_good_choice"]].copy()
    if len(bad):
        bad.to_csv(os.path.join(OUTPUT_DIR, "bad_choices_rows.csv"), index=False)
        print("Saved bad choices rows.")
    if len(good):
        good.to_csv(os.path.join(OUTPUT_DIR, "good_choices_rows.csv"), index=False)
        print("Saved good choices rows.")

    print("\n✓ Analysis complete. Check the 'analysis_output' folder.")

if __name__ == "__main__":
    main()
