"""
LLM Pandemic Policy DCE Analysis – Full Extended Version
---------------------------------------------------------
Performs full statistical and visualization analysis for simulation results:
- Overall summary and visualizations
- Model-level analysis
- County-type analysis
- Role-based analysis
- Model × County-type analysis
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")

# === 1. Combine CSV Files ===
def combine_csv_files(directory="county_results", pattern="results_*.csv"):
    csv_files = glob.glob(os.path.join(directory, pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df['temperature'] = pd.to_numeric(df.get('temperature', np.nan), errors='coerce')

    if 'county_type' not in df.columns:
        print("⚠️ Adding placeholder 'county_type' column.")
        df['county_type'] = 'Unknown'

    if 'role' not in df.columns:
        print("⚠️ Adding placeholder 'role' column.")
        df['role'] = 'Individual'

    print(f"✅ Combined {len(csv_files)} files, total {len(df)} rows.")
    return df


# === 2. Basic Utility Functions ===
def compute_bad_choices(df):
    return df[
        (df['infection_rate_selected'] > df['infection_rate_non_selected']) &
        (df['npi_level_selected'] >= df['npi_level_non_selected'])
    ]


# === 3. Summary Analyses ===
def overall_summary(df):
    bad = compute_bad_choices(df)
    stats = {
        'total': len(df),
        'bad_choices': len(bad),
        'bad_pct': len(bad) / len(df) * 100,
        'mean_inf_sel': df['infection_rate_selected'].mean(),
        'mean_inf_non': df['infection_rate_non_selected'].mean(),
        'median_inf_sel': df['infection_rate_selected'].median(),
        'median_inf_non': df['infection_rate_non_selected'].median()
    }
    return stats


def model_summary(df):
    records = []
    for model, sub in df.groupby('model'):
        bad = compute_bad_choices(sub)
        rec = {
            'model': model,
            'total': len(sub),
            'bad_pct': len(bad) / len(sub) * 100,
            'mean_inf_sel': sub['infection_rate_selected'].mean(),
            'mean_inf_non': sub['infection_rate_non_selected'].mean()
        }
        for npi in [0, 1, 2]:
            rec[f'npi_{npi}_sel_pct'] = (sub['npi_level_selected'] == npi).mean() * 100
        records.append(rec)
    return pd.DataFrame(records)


def county_summary(df):
    records = []
    for ctype, sub in df.groupby('county_type'):
        bad = compute_bad_choices(sub)
        rec = {
            'county_type': ctype,
            'total': len(sub),
            'bad_pct': len(bad) / len(sub) * 100,
            'mean_inf_sel': sub['infection_rate_selected'].mean(),
            'mean_inf_non': sub['infection_rate_non_selected'].mean()
        }
        for npi in [0, 1, 2]:
            rec[f'npi_{npi}_sel_pct'] = (sub['npi_level_selected'] == npi).mean() * 100
        records.append(rec)
    return pd.DataFrame(records)


def role_summary(df):
    records = []
    for role, sub in df.groupby('role'):
        bad = compute_bad_choices(sub)
        rec = {
            'role': role,
            'total': len(sub),
            'bad_pct': len(bad) / len(sub) * 100,
            'mean_inf_sel': sub['infection_rate_selected'].mean(),
            'mean_inf_non': sub['infection_rate_non_selected'].mean()
        }
        for npi in [0, 1, 2]:
            rec[f'npi_{npi}_sel_pct'] = (sub['npi_level_selected'] == npi).mean() * 100
        records.append(rec)
    return pd.DataFrame(records)


def model_county_summary(df):
    records = []
    for (model, ctype), sub in df.groupby(['model', 'county_type']):
        bad = compute_bad_choices(sub)
        rec = {
            'model': model,
            'county_type': ctype,
            'total': len(sub),
            'bad_pct': len(bad) / len(sub) * 100,
            'mean_inf_sel': sub['infection_rate_selected'].mean(),
            'mean_inf_non': sub['infection_rate_non_selected'].mean()
        }
        for npi in [0, 1, 2]:
            rec[f'npi_{npi}_sel_pct'] = (sub['npi_level_selected'] == npi).mean() * 100
        records.append(rec)
    return pd.DataFrame(records)


# === 4. Visualizations ===
def make_visualizations(df, model_df, county_df, role_df, model_county_df, outdir="analysis_output"):
    os.makedirs(outdir, exist_ok=True)

    # --- Original-style global charts ---
    plt.figure(figsize=(8,5))
    sns.histplot(df, x='infection_rate_selected', bins=10, color='blue', label='Selected', alpha=0.6)
    sns.histplot(df, x='infection_rate_non_selected', bins=10, color='red', label='Non-Selected', alpha=0.5)
    plt.legend()
    plt.title("Distribution of Infection Rates (Selected vs. Non-Selected)")
    plt.savefig(os.path.join(outdir, "infection_rate_distribution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    sns.countplot(x='npi_level_selected', data=df)
    plt.title("NPI Level Choices (All Models Combined)")
    plt.xticks([0,1,2], ['No Closure', 'Crowded', 'Non-Essential'])
    plt.savefig(os.path.join(outdir, "npi_distribution_all.png"), dpi=300)
    plt.close()

    # --- County-only comparisons ---
    plt.figure(figsize=(8,5))
    sns.barplot(data=county_df, x='county_type', y='mean_inf_sel', color='skyblue')
    plt.title("Mean Infection Rate (Selected) by County Type")
    plt.savefig(os.path.join(outdir, "county_mean_infection_selected.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    sns.barplot(data=county_df, x='county_type', y='bad_pct', color='salmon')
    plt.title("Bad Choice Percentage by County Type")
    plt.savefig(os.path.join(outdir, "county_bad_choices.png"), dpi=300)
    plt.close()

    # --- Role comparisons ---
    plt.figure(figsize=(8,5))
    sns.barplot(data=role_df, x='role', y='mean_inf_sel', color='lightgreen')
    plt.title("Mean Infection Rate (Selected) by Role")
    plt.savefig(os.path.join(outdir, "role_mean_infection_selected.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    sns.barplot(data=role_df, x='role', y='bad_pct', color='orange')
    plt.title("Bad Choice Percentage by Role")
    plt.savefig(os.path.join(outdir, "role_bad_choices.png"), dpi=300)
    plt.close()

    # --- Role × County comparison ---
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='county_type', y='infection_rate_selected', hue='role')
    plt.title("Mean Infection Rate by County Type and Role")
    plt.savefig(os.path.join(outdir, "role_county_infection_rate.png"), dpi=300)
    plt.close()

    # --- Model × County comparison ---
    plt.figure(figsize=(10,6))
    sns.barplot(data=model_county_df, x='county_type', y='mean_inf_sel', hue='model')
    plt.title("Mean Infection Rate (Selected) by Model × County Type")
    plt.savefig(os.path.join(outdir, "model_county_infection_rate_selected.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10,6))
    sns.barplot(data=model_county_df, x='county_type', y='bad_pct', hue='model', palette='Reds')
    plt.title("Bad Choice % by Model × County Type")
    plt.savefig(os.path.join(outdir, "model_county_bad_choices.png"), dpi=300)
    plt.close()

    print(f"✅ Visualizations saved in {outdir}")


# === 5. Run All ===
if __name__ == "__main__":
    combined = combine_csv_files()

    overall = overall_summary(combined)
    print("\n=== Overall Summary ===")
    for k, v in overall.items():
        print(f"{k:25}: {v}")

    model_df = model_summary(combined)
    county_df = county_summary(combined)
    role_df = role_summary(combined)
    model_county_df = model_county_summary(combined)

    os.makedirs("analysis_output", exist_ok=True)
    model_df.to_csv("analysis_output/model_summary.csv", index=False)
    county_df.to_csv("analysis_output/county_summary.csv", index=False)
    role_df.to_csv("analysis_output/role_summary.csv", index=False)
    model_county_df.to_csv("analysis_output/model_county_summary.csv", index=False)

    make_visualizations(combined, model_df, county_df, role_df, model_county_df)
    print("\n✅ All analyses complete!")
