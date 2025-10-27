"""
LLM Pandemic Policy DCE Analysis with Cross-Model Comparison
-------------------------------------------------------------
Combines multiple CSV files from LLM simulations and performs:
1. Overall "simple analysis" across all models
2. Model-specific analysis for cross-model comparison
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# === 1. COMBINE ALL CSV FILES ===
def combine_csv_files(directory="combined_results", pattern="results_*.csv"):
    csv_files = glob.glob(os.path.join(directory, pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {pattern} in {directory}")
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    # Read and combine all CSVs
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} total choices")
    print(f"Models: {combined_df['model'].unique()}")
    print(f"Choices per model:")
    for model, count in combined_df['model'].value_counts().items():
        print(f"  - {model}: {count}")
    
    return combined_df


# === 2. SIMPLE ANALYSIS (Overall) ===
def perform_simple_analysis(df):
    """
    - Summarize infection rates in selected vs non-selected scenarios
    - Summarize NPI levels (business closures) in selected vs non-selected
    - Identify "bad choices"
    """
    # Extract selected and non-selected data
    selected_data = df[['infection_rate_selected', 'npi_level_selected']].copy()
    selected_data.columns = ['infection_rate', 'npi_level']
    selected_data['type'] = 'Selected'
    
    non_selected_data = df[['infection_rate_non_selected', 'npi_level_non_selected']].copy()
    non_selected_data.columns = ['infection_rate', 'npi_level']
    non_selected_data['type'] = 'Not Selected'
    
    # Combine for easy plotting
    comparison_df = pd.concat([selected_data, non_selected_data], ignore_index=True)
    
    # Calculate summary statistics
    stats = {
        'mean_infection_selected': df['infection_rate_selected'].mean(),
        'mean_infection_non_selected': df['infection_rate_non_selected'].mean(),
        'median_infection_selected': df['infection_rate_selected'].median(),
        'median_infection_non_selected': df['infection_rate_non_selected'].median(),
    }
    
    # Business closure frequencies
    npi_labels = {0: 'No Closure', 1: 'Crowded Closures', 2: 'Non-Essential Closures'}
    
    # Count frequencies
    selected_npi_counts = df['npi_level_selected'].value_counts()
    non_selected_npi_counts = df['npi_level_non_selected'].value_counts()
    
    # Convert to percentages
    selected_npi_pct = (selected_npi_counts / len(df) * 100).to_dict()
    non_selected_npi_pct = (non_selected_npi_counts / len(df) * 100).to_dict()
    
    stats['npi_selected'] = selected_npi_pct
    stats['npi_non_selected'] = non_selected_npi_pct
    
    # Identify "bad choices"
    bad_choices = df[
        (df['infection_rate_selected'] > df['infection_rate_non_selected']) &
        (df['npi_level_selected'] >= df['npi_level_non_selected'])
    ]
    
    stats['bad_choices_count'] = len(bad_choices)
    stats['bad_choices_pct'] = (len(bad_choices) / len(df)) * 100
    stats['total_choices'] = len(df)
    
    return comparison_df, stats, bad_choices


# === 3. MODEL-SPECIFIC ANALYSIS ===
def perform_model_comparison(df):
    """
    Performs analysis split by model for cross-model comparison.
    """
    models = df['model'].unique()
    model_stats = []
    
    for model in models:
        model_df = df[df['model'] == model]
        
        # Basic stats
        stats = {
            'model': model,
            'total_choices': len(model_df),
            'mean_infection_selected': model_df['infection_rate_selected'].mean(),
            'mean_infection_non_selected': model_df['infection_rate_non_selected'].mean(),
            'median_infection_selected': model_df['infection_rate_selected'].median(),
            'median_infection_non_selected': model_df['infection_rate_non_selected'].median(),
        }
        
        # NPI preferences
        for npi in [0, 1, 2]:
            selected_count = (model_df['npi_level_selected'] == npi).sum()
            non_selected_count = (model_df['npi_level_non_selected'] == npi).sum()
            stats[f'npi_{npi}_selected_pct'] = (selected_count / len(model_df)) * 100
            stats[f'npi_{npi}_non_selected_pct'] = (non_selected_count / len(model_df)) * 100
        
        # Bad choices
        bad_choices = model_df[
            (model_df['infection_rate_selected'] > model_df['infection_rate_non_selected']) &
            (model_df['npi_level_selected'] >= model_df['npi_level_non_selected'])
        ]
        stats['bad_choices_count'] = len(bad_choices)
        stats['bad_choices_pct'] = (len(bad_choices) / len(model_df)) * 100
        
        model_stats.append(stats)
    
    return pd.DataFrame(model_stats)


# === 4. CROSS-MODEL VISUALIZATIONS ===
def create_model_comparison_visualizations(df, model_stats_df, output_dir="analysis_output"):
    """
    Creates visualizations comparing models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models = df['model'].unique()
    n_models = len(models)
    
    # Figure 1: Mean Infection Rates by Model
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_models)
    width = 0.35
    
    selected_means = model_stats_df['mean_infection_selected'].values
    non_selected_means = model_stats_df['mean_infection_non_selected'].values
    
    bars1 = ax.bar(x - width/2, selected_means, width, label='Selected', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, non_selected_means, width, label='Not Selected', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Infection Rate (per 100)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Infection Rates: Selected vs Not Selected Scenarios by Model', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_stats_df['model'].values, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_infection_rates.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_comparison_infection_rates.png")
    
    # Figure 2: NPI Distribution by Model
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    npi_labels = {0: 'No Closure', 1: 'Crowded\nClosures', 2: 'Non-Essential\nClosures'}
    
    # Selected scenarios
    npi_selected_data = []
    for _, row in model_stats_df.iterrows():
        for npi in [0, 1, 2]:
            npi_selected_data.append({
                'Model': row['model'],
                'NPI Level': npi_labels[npi],
                'Percentage': row[f'npi_{npi}_selected_pct']
            })
    npi_selected_df = pd.DataFrame(npi_selected_data)
    
    sns.barplot(data=npi_selected_df, x='Model', y='Percentage', hue='NPI Level', ax=axes[0])
    axes[0].set_title('NPI Levels in Selected Scenarios by Model', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title='NPI Level', fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Not selected scenarios
    npi_non_selected_data = []
    for _, row in model_stats_df.iterrows():
        for npi in [0, 1, 2]:
            npi_non_selected_data.append({
                'Model': row['model'],
                'NPI Level': npi_labels[npi],
                'Percentage': row[f'npi_{npi}_non_selected_pct']
            })
    npi_non_selected_df = pd.DataFrame(npi_non_selected_data)
    
    sns.barplot(data=npi_non_selected_df, x='Model', y='Percentage', hue='NPI Level', ax=axes[1])
    axes[1].set_title('NPI Levels in Not Selected Scenarios by Model', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='NPI Level', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_npi_levels.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_comparison_npi_levels.png")
    
    # Figure 3: Bad Choices Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(model_stats_df['model'], model_stats_df['bad_choices_pct'], 
                  color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bad Choices (%)', fontsize=12, fontweight='bold')
    ax.set_title('Percentage of "Bad Choices" by Model\n(Higher infection + Equal/More restrictive NPI)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(model_stats_df['model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels and count
    for i, (bar, pct, count) in enumerate(zip(bars, model_stats_df['bad_choices_pct'], 
                                               model_stats_df['bad_choices_count'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%\n(n={int(count)})', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_bad_choices.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_comparison_bad_choices.png")
    
    # Figure 4: Infection Rate Preference (Selected - Not Selected)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    infection_diff = model_stats_df['mean_infection_selected'] - model_stats_df['mean_infection_non_selected']
    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in infection_diff]
    
    bars = ax.bar(model_stats_df['model'], infection_diff, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Infection Rate Difference\n(Selected - Not Selected)', fontsize=12, fontweight='bold')
    ax.set_title('Model Preference for Lower vs Higher Infection Rates\n(Negative = Prefers Lower Infection)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(model_stats_df['model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, diff in zip(bars, infection_diff):
        height = bar.get_height()
        y_pos = height + (0.3 if height > 0 else -0.5)
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{diff:.2f}', ha='center', va='bottom' if height > 0 else 'top', 
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_infection_preference.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_comparison_infection_preference.png")
    
    # Figure 5: Heatmap of NPI preferences (Selected scenarios)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    npi_heatmap_data = model_stats_df[['model', 'npi_0_selected_pct', 
                                        'npi_1_selected_pct', 'npi_2_selected_pct']].copy()
    npi_heatmap_data.columns = ['Model', 'No Closure', 'Crowded Closures', 'Non-Essential Closures']
    npi_heatmap_data = npi_heatmap_data.set_index('Model')
    
    sns.heatmap(npi_heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Percentage (%)'}, ax=ax)
    ax.set_title('NPI Level Distribution in Selected Scenarios by Model', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('NPI Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_npi_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/model_comparison_npi_heatmap.png")
    
    plt.close('all')


# === 5. OVERALL VISUALIZATIONS ===
def create_overall_visualizations(comparison_df, stats, output_dir="analysis_output"):
    """
    Creates visualizations for overall analysis (all models combined).
    """
    npi_labels = {0: 'No Closure', 1: 'Crowded\nClosures', 2: 'Non-Essential\nClosures'}
    comparison_df['npi_label'] = comparison_df['npi_level'].map(npi_labels)
    
    # Figure 1: Overall Infection Rate Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.countplot(data=comparison_df, x='infection_rate', hue='type', ax=axes[0])
    axes[0].set_xlabel('Infection Rate (per 100 people)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Overall Distribution of Infection Rates\nSelected vs Not Selected Scenarios', 
                      fontsize=13, fontweight='bold')
    axes[0].legend(title='Scenario Type', fontsize=10)
    
    means = comparison_df.groupby('type')['infection_rate'].mean()
    sns.barplot(x=means.index, y=means.values, ax=axes[1], palette=['#2ecc71', '#e74c3c'])
    axes[1].set_ylabel('Mean Infection Rate (per 100)', fontsize=12)
    axes[1].set_xlabel('')
    axes[1].set_title('Overall Mean Infection Rate Comparison', fontsize=13, fontweight='bold')
    
    for i, v in enumerate(means.values):
        axes[1].text(i, v + 0.3, f'{v:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_infection_rate_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/overall_infection_rate_analysis.png")
    
    # Figure 2: Overall NPI Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.countplot(data=comparison_df, x='npi_label', hue='type', ax=ax,
                  order=['No Closure', 'Crowded\nClosures', 'Non-Essential\nClosures'])
    ax.set_xlabel('NPI Level (Business Closure)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Overall Distribution of NPI Levels\nSelected vs Not Selected Scenarios', 
                 fontsize=13, fontweight='bold')
    ax.legend(title='Scenario Type', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_npi_level_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/overall_npi_level_analysis.png")
    
    plt.close('all')


# === 6. SUMMARY TABLES ===
def print_model_comparison_table(model_stats_df):
    """
    Prints a formatted table comparing models.
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    print("\n--- Infection Rate Preferences ---")
    print(f"{'Model':<30} {'Selected (Mean)':<20} {'Not Selected (Mean)':<20} {'Difference':<15}")
    print("-" * 85)
    for _, row in model_stats_df.iterrows():
        diff = row['mean_infection_selected'] - row['mean_infection_non_selected']
        print(f"{row['model']:<30} {row['mean_infection_selected']:<20.2f} "
              f"{row['mean_infection_non_selected']:<20.2f} {diff:<15.2f}")
    
    print("\n--- Bad Choices Comparison ---")
    print(f"{'Model':<30} {'Bad Choices Count':<20} {'Bad Choices %':<15}")
    print("-" * 65)
    for _, row in model_stats_df.iterrows():
        print(f"{row['model']:<30} {int(row['bad_choices_count']):<20} {row['bad_choices_pct']:<15.1f}")
    
    print("\n--- NPI Preferences (Selected Scenarios) ---")
    npi_labels = ['No Closure', 'Crowded Closures', 'Non-Essential Closures']
    print(f"{'Model':<30} {npi_labels[0]:<20} {npi_labels[1]:<20} {npi_labels[2]:<20}")
    print("-" * 90)
    for _, row in model_stats_df.iterrows():
        print(f"{row['model']:<30} {row['npi_0_selected_pct']:<20.1f} "
              f"{row['npi_1_selected_pct']:<20.1f} {row['npi_2_selected_pct']:<20.1f}")
    
    print("\n" + "="*80)


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Step 1: Combine all CSV files
    print("Step 1: Combining CSV files...")
    combined_df = combine_csv_files(directory="combined_results", pattern="results_*.csv")
    
    # Save combined dataset
    combined_df.to_csv("combined_results/all_models_combined.csv", index=False)
    print(f"\nSaved combined dataset: combined_results/all_models_combined.csv")
    
    # Step 2: Perform overall analysis
    print("\nStep 2: Performing overall analysis...")
    comparison_df, overall_stats, bad_choices = perform_simple_analysis(combined_df)
    
    # Step 3: Perform model-specific analysis
    print("\nStep 3: Performing model-specific analysis...")
    model_stats_df = perform_model_comparison(combined_df)
    
    # Save model comparison stats
    model_stats_df.to_csv("analysis_output/model_comparison_stats.csv", index=False)
    print(f"Saved: analysis_output/model_comparison_stats.csv")
    
    # Step 4: Create overall visualizations
    print("\nStep 4: Creating overall visualizations...")
    create_overall_visualizations(comparison_df, overall_stats, output_dir="analysis_output")
    
    # Step 5: Create model comparison visualizations
    print("\nStep 5: Creating model comparison visualizations...")
    create_model_comparison_visualizations(combined_df, model_stats_df, output_dir="analysis_output")
    
    # Step 6: Print comparison table
    print_model_comparison_table(model_stats_df)
    
    # Save bad choices for inspection
    if len(bad_choices) > 0:
        bad_choices.to_csv("analysis_output/bad_choices_all_models.csv", index=False)
        print(f"\nSaved bad choices details: analysis_output/bad_choices_all_models.csv")
    
    print("\n Analysis complete! Check the 'analysis_output' folder for all visualizations and stats.")