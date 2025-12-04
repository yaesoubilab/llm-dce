# %% [markdown]
# # Data Ingestion

# %%
import pandas as pd
import os
from pathlib import Path
import glob

# %%
# Dictionary for model name mapping
MODEL_NAME_MAPPING = {
    'x-ai/grok-4-fast': 'Grok',
    'openai/gpt-5-nano': 'ChatGPT',
    'google/gemini-2.5-flash': 'Gemini',
    'deepseek/deepseek-chat-v3.1': 'DeepSeek'
}

# %%
# Define strategy directories
strategies = {
    'strategy_one': 'strategy_one_results',
    'strategy_two': 'strategy_two_results',
    'strategy_three': 'strategy_three_results',
    'strategy_four': 'strategy_four_results'
}


# %%

# List to hold all dataframes
all_dfs = []

# Read files from each strategy directory
for strategy_name, directory in strategies.items():
    dir_path = os.path.join('.', directory)
    
    # Check if directory exists
    if not os.path.exists(dir_path):
        print(f"Warning: Directory '{dir_path}' not found. Skipping...")
        continue
    
    # Find all CSV files matching the pattern results_*.csv
    csv_files = glob.glob(os.path.join(dir_path, 'results_*.csv'))
    
    if not csv_files:
        print(f"Warning: No CSV files found in '{dir_path}'. Skipping...")
        continue
    
    # Read each CSV file in the directory
    for csv_file in csv_files:
        # Extract model name from filename
        filename = os.path.basename(csv_file)
        model_identifier = filename.replace('results_', '').replace('.csv', '')
        
        # Read the CSV
        df = pd.read_csv(csv_file)
        
        # Add strategy column
        df['strategy'] = strategy_name
        
        all_dfs.append(df)
        print(f"✓ Loaded: {csv_file} ({len(df)} rows)")


# %%

# Combine all dataframes
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"Successfully combined {len(all_dfs)} files")
    print(f"Total rows: {len(combined_df)}")
    print(f"{'='*60}")
    results_df = combined_df
else:
    print("No data files were found or loaded.")

# %%
results_df['model'] = results_df['model'].replace(MODEL_NAME_MAPPING)

# %%
results_df['npi_preference'] = results_df['npi_level_selected'] - results_df['npi_level_non_selected']

# %%
results_df['infection_preference'] = results_df['infection_rate_selected'] - results_df['infection_rate_non_selected']

# %%
results_df['irrational_choice'] = (
    ((results_df['infection_rate_selected'] > results_df['infection_rate_non_selected']) & 
     (results_df['npi_level_selected'] >= results_df['npi_level_non_selected'])) |
    ((results_df['npi_level_selected'] > results_df['npi_level_non_selected']) & 
    (results_df['infection_rate_selected'] <= results_df['infection_rate_non_selected']))
)

# %%
results_df.head()

# %% [markdown]
# # Analysis

# %% [markdown]
# ### Figure 1: % Irrational Choice By Strategy

# %% [markdown]
# ##### Pre-processing + Function definition

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# %%

# Set style with larger fonts
sns.set_style("white")  # Changed from "whitegrid" to remove grid
sns.set_context("paper", font_scale=2.8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300


# %%

# Calculate irrational choice frequency by model and strategy
def calculate_irrational_frequency(df):
    """
    Calculate the frequency of irrational choices for each model in each strategy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'model', 'strategy', and 'irrational_choice' columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with irrational choice rates by strategy and model
    """
    # Group by strategy and model, calculate mean of irrational_choice
    freq_data = df.groupby(['strategy', 'model'])['irrational_choice'].agg([
        ('irrational_rate', 'mean'),
        ('total_choices', 'count')
    ]).reset_index()
    
    # Convert to percentage
    freq_data['irrational_percentage'] = freq_data['irrational_rate'] * 100
    
    return freq_data


# %%

def extract_strategy_number(strategy_name):
    """
    Extract the numeric value from a strategy name for proper sorting.
    Handles various formats like 'strategy_one', 'strategy_1', 'Strategy One', etc.
    """
    # Map word numbers to integers
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    strategy_lower = strategy_name.lower()
    
    # First, try to find a direct number
    digits = ''.join(filter(str.isdigit, strategy_lower))
    if digits:
        return int(digits)
    
    # If no digits, look for word numbers
    for word, num in word_to_num.items():
        if word in strategy_lower:
            return num
    
    # Default to 0 if no number found
    return 0



# %%

# Calculate frequencies
irrational_freq = calculate_irrational_frequency(results_df)


# %%

# Individual Strategy Plots
# Create separate publication-ready bar graphs for each strategy.

def plot_individual_strategies(freq_data, save_figures=False):
    """
    Create individual publication-ready bar plots for each strategy using Seaborn.
    Models are ordered by irrational choice rate (highest to lowest).
    
    Parameters:
    -----------
    freq_data : pd.DataFrame
        DataFrame with strategy, model, and irrational_percentage columns
    save_figures : bool
        Whether to save figures to disk
    """
    # Sort strategies numerically using proper extraction
    strategies = sorted(freq_data['strategy'].unique(), 
                       key=extract_strategy_number)
    
    # Define color palette
    models = sorted(freq_data['model'].unique())
    palette = sns.color_palette("Set2", n_colors=len(models))
    model_colors = dict(zip(models, palette))
    
    # Create individual plots for each strategy
    for strategy in strategies:
        strategy_data = freq_data[freq_data['strategy'] == strategy].copy()
        
        # Sort models by irrational percentage (highest to lowest)
        strategy_data = strategy_data.sort_values('irrational_percentage', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create bar plot with Seaborn
        bars = sns.barplot(
            data=strategy_data,
            x='model',
            y='irrational_percentage',
            palette=[model_colors[m] for m in strategy_data['model']],
            edgecolor='black',
            linewidth=1.8,
            ax=ax
        )
        
        # Add value labels on bars
        for i, bar in enumerate(bars.patches):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.5,
                f'{height:.1f}%',
                ha='center', 
                va='bottom',
                fontsize=32,
                fontweight='bold'
            )
        
        # Formatting
        ax.set_xlabel('Model', fontsize=18, fontweight='bold', labelpad=12)
        ax.set_ylabel('Irrational Choice Rate (%)', fontsize=18, fontweight='bold', labelpad=12)
        
        # Format strategy name for title
        strategy_title = strategy.replace('_', ' ').title()
        ax.set_title(f'Irrational Choice Rate by Model\n{strategy_title}', 
                     fontsize=20, fontweight='bold', pad=20)
        
        # Set y-axis limits with padding
        max_val = strategy_data['irrational_percentage'].max()
        ax.set_ylim(0, min(100, max_val * 1.25))
        
        # Adjust tick parameters
        ax.tick_params(axis='x', labelsize=16, rotation=0)
        ax.tick_params(axis='y', labelsize=16)
        
        # Remove all spines for cleaner look
        sns.despine(ax=ax)
        
        plt.tight_layout()
        
        if save_figures:
            plt.savefig(f'irrational_choices_{strategy}.png', dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()



# %%

# Generate individual strategy plots
# plot_individual_strategies(irrational_freq, save_figures=False)



# %%

def plot_combined_strategies(freq_data, save_figure=False):
    """
    Create a combined publication-ready multi-panel plot with all strategies.
    Strategies are displayed in numerical order (Strategy One, Two, Three, Four, etc.).
    Models within each panel are ordered by irrational choice rate (highest to lowest).
    
    Parameters:
    -----------
    freq_data : pd.DataFrame
        DataFrame with strategy, model, and irrational_percentage columns
    save_figure : bool
        Whether to save figure to disk
    """
    # Sort strategies numerically using proper extraction
    strategies = sorted(freq_data['strategy'].unique(), 
                       key=extract_strategy_number)
    n_strategies = len(strategies)
    
    # Determine grid layout (2 columns, dynamic rows)
    n_cols = 2
    n_rows = int(np.ceil(n_strategies / n_cols))
    
    # Define color palette
    models = sorted(freq_data['model'].unique())
    palette = sns.color_palette("Set2", n_colors=len(models))
    model_colors = dict(zip(models, palette))
    
    # Create figure with subplots - scale size based on number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9 * n_rows))
    
    # Flatten axes array for easier iteration
    if n_strategies == 1:
        axes = np.array([axes])
    elif n_strategies <= 2:
        axes = np.array(axes) if isinstance(axes, np.ndarray) else np.array([axes])
        if axes.ndim == 1:
            axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each strategy IN NUMERICAL ORDER
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strategy_data = freq_data[freq_data['strategy'] == strategy].copy()
        
        # Sort models by irrational percentage (highest to lowest) within this strategy
        strategy_data = strategy_data.sort_values('irrational_percentage', ascending=False)
        
        # Create bar plot using ordered data
        bars = ax.bar(
            range(len(strategy_data)),
            strategy_data['irrational_percentage'],
            color=[model_colors[m] for m in strategy_data['model']],
            edgecolor='black',
            linewidth=1.8
        )
        
        # Set x-axis labels to model names
        ax.set_xticks(range(len(strategy_data)))
        ax.set_xticklabels(strategy_data['model'], rotation=0, ha='center') 
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, strategy_data['irrational_percentage'])):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.5,
                f'{val:.1f}%',
                ha='center', 
                va='bottom',
                fontsize=20, # font size adjustment
                fontweight='bold'
            )
        
        # Formatting
        ax.set_xlabel('Model', fontsize=20, fontweight='bold', labelpad=10)# font size adjustment
        ax.set_ylabel('Irrational Choice Rate (%)', fontsize=20, fontweight='bold', labelpad=10)# font size adjustment
        
        # Format strategy name for title (NO panel labels)
        strategy_title = strategy.replace('_', ' ').title()
        ax.set_title(strategy_title, fontsize=24, fontweight='bold', pad=15) # font size adjustment
        
        # Set y-axis limits with padding
        max_val = freq_data['irrational_percentage'].max()
        ax.set_ylim(0, min(100, max_val * 1.25))
        
        # Adjust tick parameters
        ax.tick_params(axis='x', labelsize=20) # font size adjustment
        ax.tick_params(axis='y', labelsize=20) # font size adjustment
        
        # Remove all spines
        sns.despine(ax=ax)
    
    # Hide unused subplots
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)
    
    # Overall title
    fig.suptitle('Irrational Choice Rates by Model and Strategy', 
                 fontsize=32, fontweight='bold', y=0.995) # font size adjustment
    
    plt.tight_layout()
    
    if save_figure:
        plt.savefig('irrational_choices_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()



# %% [markdown]
# ##### Figure Output

# %%
# Generate combined multi-panel plot with proper ordering
plot_combined_strategies(irrational_freq, save_figure=False)

# %% [markdown]
# ### Figure 1: % Irrational Choice By Strategy (context-stratified)

# %% [markdown]
# ##### Pre-Processing + Function Definition

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# %%

# Set publication-ready style
sns.set_style("white")
sns.set_context("paper", font_scale=1.8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300


# %%

def extract_strategy_number(strategy_name):
    """Extract numeric value from strategy name for proper sorting."""
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    strategy_lower = strategy_name.lower()
    digits = ''.join(filter(str.isdigit, strategy_lower))
    if digits:
        return int(digits)
    
    for word, num in word_to_num.items():
        if word in strategy_lower:
            return num
    
    return 0


# %%

def get_context_column(strategy_name):
    """Return the appropriate context column name for each strategy."""
    strategy_num = extract_strategy_number(strategy_name)
    
    context_mapping = {
        2: 'capacity_context',
        3: 'economic_context',
        4: 'user_preference_type'
    }
    
    return context_mapping.get(strategy_num, None)


# %%

def get_legend_title(context_col):
    """Return a formatted legend title based on the context column name."""
    if not context_col:
        return None
    
    title_mapping = {
        'capacity_context': 'Capacity Context',
        'economic_context': 'Economic Context',
        'user_preference_type': 'User Preference'
    }
    
    return title_mapping.get(context_col, context_col.replace('_', ' ').title())


# %%

def calculate_context_breakdown(df):
    """Calculate irrational choice rates broken down by context within each strategy."""
    results = []
    
    strategies = sorted(df['strategy'].unique(), key=extract_strategy_number)
    
    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy].copy()
        context_col = get_context_column(strategy)
        
        if context_col and context_col in strategy_data.columns:
            # Group by model and context
            grouped = strategy_data.groupby(['model', context_col])['irrational_choice'].agg([
                ('irrational_rate', 'mean'),
                ('count', 'count')
            ]).reset_index()
            
            grouped['irrational_percentage'] = grouped['irrational_rate'] * 100
            grouped['strategy'] = strategy
            grouped['context_type'] = context_col
            
            results.append(grouped)
        else:
            # Strategy 1 or missing context - overall rate
            grouped = strategy_data.groupby('model')['irrational_choice'].agg([
                ('irrational_rate', 'mean'),
                ('count', 'count')
            ]).reset_index()
            
            grouped['irrational_percentage'] = grouped['irrational_rate'] * 100
            grouped['strategy'] = strategy
            grouped['context_type'] = 'overall'
            grouped['context'] = 'Overall'
            
            results.append(grouped)
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# %%

# Calculate context breakdowns
context_breakdown = calculate_context_breakdown(results_df)


# %%

def plot_strategy_grouped_by_context(breakdown_data, save_figures=False):
    """
    Create individual plots for each strategy with grouped bars by context.
    Each model has multiple bars (one per context) with different colors.
    """
    strategies = sorted(breakdown_data['strategy'].unique(), key=extract_strategy_number)
    
    for strategy in strategies:
        strategy_data = breakdown_data[breakdown_data['strategy'] == strategy].copy()
        context_col = get_context_column(strategy)
        strategy_num = extract_strategy_number(strategy)
        
        if not context_col:
            context_col = 'context'
        
        # Get unique models and contexts
        models = sorted(strategy_data['model'].unique())
        contexts = sorted(strategy_data[context_col].unique())
        
        # Create color palette for contexts
        palette = sns.color_palette("Set2", n_colors=len(contexts))
        context_colors = dict(zip(contexts, palette))
        
        # Set up the bar positions
        x = np.arange(len(models))
        width = 0.8 / len(contexts)  # Width of each bar
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot bars for each context
        for i, context in enumerate(contexts):
            context_data = strategy_data[strategy_data[context_col] == context].copy()
            context_data = context_data.set_index('model').reindex(models).reset_index()
            
            # Calculate positions for this context's bars
            positions = x + (i - len(contexts)/2 + 0.5) * width
            
            bars = ax.bar(
                positions,
                context_data['irrational_percentage'],
                width,
                label=str(context).replace('_', ' ').title(),
                color=context_colors[context],
                edgecolor='black',
                linewidth=1.5
            )
            
            # Add value labels on bars
            for bar, val in zip(bars, context_data['irrational_percentage']):
                if not pd.isna(val):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.5,
                        f'{val:.1f}%',
                        ha='center', 
                        va='bottom',
                        fontsize=12,
                        fontweight='bold'
                    )
        
        # Formatting
        ax.set_xlabel('Model', fontsize=18, fontweight='bold', labelpad=12)
        ax.set_ylabel('Irrational Choice Rate (%)', fontsize=18, fontweight='bold', labelpad=12)
        
        # Format strategy name for title
        strategy_title = strategy.replace('_', ' ').title()
        ax.set_title(f'{strategy_title}', fontsize=20, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=16)
        
        # Set y-axis limits
        ax.set_ylim(0, min(100, strategy_data['irrational_percentage'].max() * 1.3))
        
        # Adjust ticks
        ax.tick_params(axis='y', labelsize=16)
        
        # Add legend only for strategies 2, 3, 4 (not strategy 1)
        if strategy_num != 1:
            legend_title = get_legend_title(context_col)
            ax.legend(
                title=legend_title,
                loc='upper right', 
                fontsize=14, 
                title_fontsize=15,
                frameon=True, 
                fancybox=True, 
                shadow=True
            )
        
        # Remove spines
        sns.despine(ax=ax)
        
        plt.tight_layout()
        
        if save_figures:
            plt.savefig(f'irrational_choices_{strategy}_grouped.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()

# Generate grouped bar plots for each strategy
# plot_strategy_grouped_by_context(context_breakdown, save_figures=False)


# %%

def plot_combined_grouped_contexts(breakdown_data, save_figure=False):
    """
    Create a combined 4-panel figure with grouped bars by context for each strategy.
    """
    strategies = sorted(breakdown_data['strategy'].unique(), key=extract_strategy_number)
    n_strategies = len(strategies)
    
    # Determine layout
    n_cols = 2
    n_rows = int(np.ceil(n_strategies / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9 * n_rows))
    
    # Flatten axes
    if n_strategies == 1:
        axes = np.array([axes])
    elif n_strategies == 2:
        axes = np.array(axes)
    else:
        axes = axes.flatten()
    
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strategy_data = breakdown_data[breakdown_data['strategy'] == strategy].copy()
        context_col = get_context_column(strategy)
        strategy_num = extract_strategy_number(strategy)
        
        if not context_col:
            context_col = 'context'
        
        # Get unique models and contexts
        models = sorted(strategy_data['model'].unique())
        contexts = sorted(strategy_data[context_col].unique())
        
        # Create color palette for contexts
        palette = sns.color_palette("Set2", n_colors=len(contexts))
        context_colors = dict(zip(contexts, palette))
        
        # Set up the bar positions
        x = np.arange(len(models))
        width = 0.8 / len(contexts)
        
        # Plot bars for each context
        for i, context in enumerate(contexts):
            context_data = strategy_data[strategy_data[context_col] == context].copy()
            context_data = context_data.set_index('model').reindex(models).reset_index()
            
            # Calculate positions for this context's bars
            positions = x + (i - len(contexts)/2 + 0.5) * width
            
            bars = ax.bar(
                positions,
                context_data['irrational_percentage'],
                width,
                label=str(context).replace('_', ' ').title(),
                color=context_colors[context],
                edgecolor='black',
                linewidth=1.5
            )
            
            # Add value labels on bars
            for bar, val in zip(bars, context_data['irrational_percentage']):
                if not pd.isna(val):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.5,
                        f'{val:.1f}%',
                        ha='center', 
                        va='bottom',
                        fontsize=11,
                        fontweight='bold'
                    )
        
        # Formatting
        ax.set_xlabel('Model', fontsize=17, fontweight='bold', labelpad=10)
        ax.set_ylabel('Irrational Choice Rate (%)', fontsize=17, fontweight='bold', labelpad=10)
        
        # Format strategy name for title
        strategy_title = strategy.replace('_', ' ').title()
        ax.set_title(strategy_title, fontsize=18, fontweight='bold', pad=15)
        
        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=15)
        
        # Set y-axis limits
        ax.set_ylim(0, min(100, breakdown_data['irrational_percentage'].max() * 1.3))
        
        # Adjust ticks
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=20)
        
        # Add legend only for strategies 2, 3, 4 (not strategy 1)
        if strategy_num != 1:
            legend_title = get_legend_title(context_col)
            ax.legend(
                title=legend_title,
                loc='upper right', 
                fontsize=16,
                title_fontsize=18,
                frameon=True, 
                fancybox=True, 
                shadow=True
            )
        
        # Remove spines
        sns.despine(ax=ax)
    
    # Hide unused subplots
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)
    
    # Overall title
    fig.suptitle('Irrational Choice Rates by Context - All Strategies', 
                 fontsize=32, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_figure:
        plt.savefig('irrational_choices_all_contexts_grouped.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()



# %% [markdown]
# ##### Figure Output

# %%

# Generate combined 4-panel grouped plot
plot_combined_grouped_contexts(context_breakdown, save_figure=False)

# %% [markdown]
# ### Figure 2: Mean Infection Preference

# %% [markdown]
# ##### Pre-Processing + Function Definition

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from scipy import stats
import pandas as pd
import numpy as np


# %%

# Set publication-ready style
sns.set_style("white")
sns.set_context("paper", font_scale=1.8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300


# %%

def extract_strategy_number(strategy_name):
    """Extract numeric value from strategy name for proper sorting."""
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    strategy_lower = strategy_name.lower()
    digits = ''.join(filter(str.isdigit, strategy_lower))
    if digits:
        return int(digits)
    
    for word, num in word_to_num.items():
        if word in strategy_lower:
            return num
    
    return 0


# %%

def get_context_column(strategy_name):
    """Return the appropriate context column name for each strategy."""
    strategy_num = extract_strategy_number(strategy_name)
    
    context_mapping = {
        2: 'capacity_context',
        3: 'economic_context',
        4: 'user_preference_type'
    }
    
    return context_mapping.get(strategy_num, None)


# %%

def get_legend_title(context_col):
    """Return a formatted legend title based on the context column name."""
    if not context_col:
        return None
    
    title_mapping = {
        'capacity_context': 'Capacity Context',
        'economic_context': 'Economic Context',
        'user_preference_type': 'User Preference'
    }
    
    return title_mapping.get(context_col, context_col.replace('_', ' ').title())


# %%

def wrap_label(text, max_len=20):
    """Wrap text labels for better readability."""
    text = str(text)
    return '\n'.join(textwrap.wrap(text, width=max_len))


# %%

def calculate_errorbar_data(df, response_col, context_cols, use_percentage=False):
    """Calculate means and confidence intervals for errorbar plots."""
    grouped = df.groupby(context_cols)[response_col]
    
    plot_data = []
    for name, group in grouped:
        if len(group) > 0:
            data = group.values
            if use_percentage:
                data = data * 100
            
            mean_val = data.mean()
            ci = stats.t.interval(
                0.95,
                len(data) - 1,
                loc=mean_val,
                scale=stats.sem(data)
            )
            
            record = {}
            if isinstance(name, tuple):
                for i, col in enumerate(context_cols):
                    record[col] = name[i]
            else:
                record[context_cols[0]] = name
            
            record['mean'] = mean_val
            record['ci_lower'] = mean_val - ci[0]
            record['ci_upper'] = ci[1] - mean_val
            record['n'] = len(data)
            
            plot_data.append(record)
    
    return pd.DataFrame(plot_data)


# %%

def plot_individual_errorbar(df, response_col, parameter_col, use_percentage=False, save_figures=False):
    """
    Create individual errorbar plots for each strategy.
    Automatically includes context variables based on strategy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with data
    response_col : str
        Column name for the response variable (e.g., 'npi_preference')
    parameter_col : str
        Single parameter column (e.g., 'infection_preference')
    use_percentage : bool
        Whether to convert to percentage
    save_figures : bool
        Whether to save figures to disk
    """
    strategies = sorted(df['strategy'].unique(), key=extract_strategy_number)
    unit = '%' if use_percentage else ''
    
    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy].copy()
        strategy_num = extract_strategy_number(strategy)
        
        # Get context column for this strategy
        context_col = get_context_column(strategy)
        
        # Determine grouping columns
        if context_col and context_col in strategy_data.columns:
            # Multiple grouping: parameter + context
            group_cols = [parameter_col, context_col]
        else:
            # Single grouping: just parameter (Strategy 1)
            group_cols = [parameter_col]
        
        # Calculate overall mean for this strategy
        overall_mean = strategy_data[response_col].mean()
        if use_percentage:
            overall_mean = overall_mean * 100
        
        # Calculate errorbar data
        plot_df = calculate_errorbar_data(strategy_data, response_col, group_cols, use_percentage)
        
        if len(plot_df) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Single grouping (Strategy 1) - simple horizontal errorbar plot
        if len(group_cols) == 1:
            # Sort by mean
            plot_df = plot_df.sort_values('mean', ascending=True)
            
            y_positions = range(1, len(plot_df) + 1)
            
            ax.errorbar(
                plot_df['mean'],
                y_positions,
                xerr=[plot_df['ci_lower'], plot_df['ci_upper']],
                fmt='o',
                markersize=12,
                markerfacecolor=sns.color_palette("Set2")[0],
                markeredgecolor='black',
                markeredgewidth=1.5,
                ecolor='black',
                elinewidth=2.0,
                capsize=6,
                capthick=2
            )
            
            # Overall mean line
            ax.axvline(
                overall_mean,
                color='red',
                linestyle='--',
                linewidth=2.5,
                alpha=0.7,
                label=f'Mean = {overall_mean:.2f}{unit}'
            )
            
            # Labels
            wrapped_labels = [wrap_label(str(val)) for val in plot_df[parameter_col]]
            ax.set_yticks(list(y_positions))
            ax.set_yticklabels(wrapped_labels, fontsize=16)
            
            ax.set_ylim(0.5, len(plot_df) + 0.5)
            
        # Multiple grouping (Strategies 2, 3, 4) - grouped errorbar plot
        else:
            primary_col = parameter_col
            secondary_col = context_col
            
            primary_values = sorted(
                plot_df[primary_col].unique(),
                key=lambda x: plot_df[plot_df[primary_col] == x]['mean'].mean(),
                reverse=True
            )
            secondary_values = sorted(plot_df[secondary_col].unique())
            
            colors = sns.color_palette("Set2", len(secondary_values))
            
            group_spacing = 1.0
            bar_height = 0.7
            bar_spacing = bar_height / max(1, len(secondary_values))
            
            y_positions = {}
            current_y = 1.0
            for primary_val in primary_values:
                y_positions[primary_val] = current_y
                current_y += group_spacing
            
            for i, secondary_val in enumerate(secondary_values):
                subset = plot_df[plot_df[secondary_col] == secondary_val]
                
                y_coords = []
                means = []
                ci_lowers = []
                ci_uppers = []
                
                for primary_val in primary_values:
                    row = subset[subset[primary_col] == primary_val]
                    if len(row) > 0:
                        y_offset = (i - len(secondary_values) / 2 + 0.5) * bar_spacing
                        y_coords.append(y_positions[primary_val] + y_offset)
                        means.append(row['mean'].values[0])
                        ci_lowers.append(row['ci_lower'].values[0])
                        ci_uppers.append(row['ci_upper'].values[0])
                
                ax.errorbar(
                    means,
                    y_coords,
                    xerr=[ci_lowers, ci_uppers],
                    fmt='o',
                    markersize=10,
                    markerfacecolor=colors[i],
                    markeredgecolor='black',
                    markeredgewidth=1.5,
                    ecolor='black',
                    elinewidth=2.0,
                    capsize=6,
                    capthick=2,
                    label=str(secondary_val).replace('_', ' ').title()
                )
            
            # Overall mean line
            ax.axvline(
                overall_mean,
                color='red',
                linestyle='--',
                linewidth=2.5,
                alpha=0.7,
                label=f'Mean = {overall_mean:.2f}{unit}',
                zorder=0
            )
            
            wrapped_labels = [wrap_label(str(val)) for val in primary_values]
            ax.set_yticks([y_positions[val] for val in primary_values])
            ax.set_yticklabels(wrapped_labels, fontsize=16)
            
            # Add legend with context title
            legend_title = get_legend_title(context_col)
            ax.legend(
                fontsize=14,
                title=legend_title,
                title_fontsize=15,
                loc='best',
                frameon=True,
                fancybox=True,
                shadow=True
            )
            
            ax.set_ylim(0.5, current_y - group_spacing + 0.5)
        
        # Common formatting
        xlabel = f'{response_col} (Mean with 95% CI)'
        if use_percentage:
            xlabel = f'{response_col} (% Mean with 95% CI)'
        ax.set_xlabel(xlabel, fontsize=18, fontweight='bold', labelpad=12)
        
        strategy_title = strategy.replace('_', ' ').title()
        ax.set_title(strategy_title, fontsize=20, fontweight='bold', pad=20)
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        sns.despine(ax=ax)
        
        plt.tight_layout()
        
        if save_figures:
            plt.savefig(f'{response_col}_{strategy}_errorbar.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()


# %%
def plot_combined_errorbar(df, response_col, parameter_col, use_percentage=False, save_figure=False):
    """
    Create a combined 4-panel errorbar plot for all strategies.
    Automatically includes context variables based on strategy.
    Y-axis labels are ordered consistently across all panels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with data
    response_col : str
        Column name for the response variable
    parameter_col : str
        Single parameter column
    use_percentage : bool
        Whether to convert to percentage
    save_figure : bool
        Whether to save figure to disk
    """
    strategies = sorted(df['strategy'].unique(), key=extract_strategy_number)
    n_strategies = len(strategies)
    unit = '%' if use_percentage else ''
    
    # **NEW: Calculate global ordering for parameter_col across all strategies**
    global_param_order = (
        df.groupby(parameter_col)[response_col]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    
    # Determine layout
    n_cols = 2
    n_rows = int(np.ceil(n_strategies / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9 * n_rows))
    
    # Flatten axes
    if n_strategies == 1:
        axes = np.array([axes])
    elif n_strategies == 2:
        axes = np.array(axes)
    else:
        axes = axes.flatten()
    
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strategy_data = df[df['strategy'] == strategy].copy()
        strategy_num = extract_strategy_number(strategy)
        
        # Get context column for this strategy
        context_col = get_context_column(strategy)
        
        # Determine grouping columns
        if context_col and context_col in strategy_data.columns:
            group_cols = [parameter_col, context_col]
        else:
            group_cols = [parameter_col]
        
        # Calculate overall mean
        overall_mean = strategy_data[response_col].mean()
        if use_percentage:
            overall_mean = overall_mean * 100
        
        # Calculate errorbar data
        plot_df = calculate_errorbar_data(strategy_data, response_col, group_cols, use_percentage)
        
        if len(plot_df) == 0:
            continue
        
        # Single grouping (Strategy 1)
        if len(group_cols) == 1:
            # **MODIFIED: Use global ordering instead of local sorting**
            plot_df[parameter_col] = pd.Categorical(
                plot_df[parameter_col], 
                categories=global_param_order, 
                ordered=True
            )
            plot_df = plot_df.sort_values(parameter_col)
            
            y_positions = range(1, len(plot_df) + 1)
            
            ax.errorbar(
                plot_df['mean'],
                y_positions,
                xerr=[plot_df['ci_lower'], plot_df['ci_upper']],
                fmt='o',
                markersize=10,
                markerfacecolor=sns.color_palette("Set2")[0],
                markeredgecolor='black',
                markeredgewidth=1.5,
                ecolor='black',
                elinewidth=1.8,
                capsize=5,
                capthick=1.8
            )
            
            ax.axvline(overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            wrapped_labels = [wrap_label(str(val), max_len=15) for val in plot_df[parameter_col]]
            ax.set_yticks(list(y_positions))
            ax.set_yticklabels(wrapped_labels, fontsize=14)
            ax.set_ylim(0.5, len(plot_df) + 0.5)
            
        # Multiple grouping (Strategies 2, 3, 4)
        else:
            primary_col = parameter_col
            secondary_col = context_col
            
            # **MODIFIED: Use global ordering instead of local sorting**
            primary_values = [val for val in global_param_order 
                            if val in plot_df[primary_col].unique()]
            secondary_values = sorted(plot_df[secondary_col].unique())
            
            colors = sns.color_palette("Set2", len(secondary_values))
            
            group_spacing = 1.0
            bar_height = 0.7
            bar_spacing = bar_height / max(1, len(secondary_values))
            
            y_positions = {}
            current_y = 1.0
            for primary_val in primary_values:
                y_positions[primary_val] = current_y
                current_y += group_spacing
            
            for i, secondary_val in enumerate(secondary_values):
                subset = plot_df[plot_df[secondary_col] == secondary_val]
                
                y_coords = []
                means = []
                ci_lowers = []
                ci_uppers = []
                
                for primary_val in primary_values:
                    row = subset[subset[primary_col] == primary_val]
                    if len(row) > 0:
                        y_offset = (i - len(secondary_values) / 2 + 0.5) * bar_spacing
                        y_coords.append(y_positions[primary_val] + y_offset)
                        means.append(row['mean'].values[0])
                        ci_lowers.append(row['ci_lower'].values[0])
                        ci_uppers.append(row['ci_upper'].values[0])
                
                ax.errorbar(
                    means,
                    y_coords,
                    xerr=[ci_lowers, ci_uppers],
                    fmt='o',
                    markersize=8,
                    markerfacecolor=colors[i],
                    markeredgecolor='black',
                    markeredgewidth=1.2,
                    ecolor='black',
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    label=str(secondary_val).replace('_', ' ').title()
                )
            
            ax.axvline(overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=0)
            
            wrapped_labels = [wrap_label(str(val), max_len=15) for val in primary_values]
            ax.set_yticks([y_positions[val] for val in primary_values])
            ax.set_yticklabels(wrapped_labels, fontsize=20)
            ax.set_ylim(0.5, current_y - group_spacing + 0.5)
            
            # Add legend with context title
            legend_title = get_legend_title(context_col)
            ax.legend(
                fontsize=18,
                title=legend_title,
                title_fontsize=18,
                loc='best',
                frameon=True,
                fancybox=True,
                shadow=True
            )
        
        # Common formatting
        strategy_title = strategy.replace('_', ' ').title()
        ax.set_title(strategy_title, fontsize=20, fontweight='bold', pad=15)
        
        xlabel = f'{response_col.replace("_", " ").title()} (Mean with 95% CI)'
        if use_percentage:
            xlabel = f'{response_col.replace("_", " ").title()} (%)'
        ax.set_xlabel(xlabel, fontsize=20, fontweight='bold', labelpad=10)
        
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        
        sns.despine(ax=ax)
    
    # Hide unused subplots
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)
    
    # Overall title
    fig.suptitle(f'{response_col.replace("_", " ").title()} by {parameter_col.replace("_", " ").title()} - All Strategies', 
                 fontsize=32, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(f'{response_col}_combined_errorbar.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

# %% [markdown]
# ##### Final Output

# %%
plot_combined_errorbar(results_df, 'infection_rate_selected', 'model', use_percentage=False)

# %% [markdown]
# ### Figure 3: Mean NPI Preference

# %% [markdown]
# ##### Final Output

# %%
plot_combined_errorbar(results_df, 'infection_rate_selected', 'model', use_percentage=False)

# %% [markdown]
# ### Figure 4: Two Axis NPI-Infection Plot

# %% [markdown]
# ##### Pre-Processing + Function Definition

# %%
def plot_mean_preferences_by_strategy(df, x_col='npi_level_selected', y_col='infection_rate_selected', save_figure=False):
    """
    Create a publication-ready multi-panel plot showing mean NPI vs mean Infection.
    One panel per strategy, with each model as a labeled point.
    Uses consistent axes across all panels for easy comparison.
    Strategies are numerically ordered in the panels.
    """
    
    # Calculate means for each strategy-model combination
    means = df.groupby(['strategy', 'model']).agg({
        x_col: 'mean',
        y_col: 'mean'
    }).reset_index()
    
    # Calculate global bounds for consistent axes across all panels
    x_min, x_max = means[x_col].min(), means[x_col].max()
    y_min, y_max = means[y_col].min(), means[y_col].max()
    
    # Add padding for readability
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    # Set consistent limits with padding for labels
    xlim = (x_min - x_range * 0.1, x_max + x_range * 0.2)
    ylim = (y_min - y_range * 0.1, y_max + y_range * 0.15)
    
    # **MODIFIED: Sort strategies numerically using extract_strategy_number**
    strategies = sorted(means['strategy'].unique(), key=extract_strategy_number)
    n_strategies = len(strategies)
    
    # Create subplot grid (2x2 for 4 strategies)
    n_cols = 2
    n_rows = int(np.ceil(n_strategies / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(12, 10),
                            sharex=True, sharey=True)
    
    axes = axes.flatten() if n_strategies > 1 else [axes]
    
    # Color palette for models with better distinction
    models = sorted(means['model'].unique())
    palette = sns.color_palette("tab10", n_colors=len(models))
    model_colors = dict(zip(models, palette))
    
    # Plot each strategy
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strategy_data = means[means['strategy'] == strategy]
        
        # Set consistent limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Plot each model as a point
        for _, row in strategy_data.iterrows():
            model = row['model']
            ax.scatter(
                row[x_col],
                row[y_col],
                s=150,
                c=[model_colors[model]],
                alpha=0.8,
                edgecolor='black',
                linewidth=1.5,
                zorder=3,
                label=model
            )
        
        # Format strategy title (e.g., "Strategy 1")
        strategy_title = strategy.replace('_', ' ').title()
        ax.set_title(strategy_title, fontsize=14, fontweight='bold', pad=10)
        
        # Clean spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        # Tick formatting
        ax.tick_params(labelsize=11, width=1.2, length=5)
    
    # Hide unused subplots
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)
    
    # Add axis labels only to edge subplots
    for i in range(n_rows):
        axes[i * n_cols].set_ylabel(
            f'{y_col.replace("_", " ").title()}',
            fontsize=13, fontweight='bold'
        )
    
    for i in range(n_cols):
        if (n_rows - 1) * n_cols + i < n_strategies:
            axes[(n_rows - 1) * n_cols + i].set_xlabel(
                f'{x_col.replace("_", " ").title()}',
                fontsize=13, fontweight='bold'
            )
    
    # Overall title
    fig.suptitle(
        f'{y_col.replace("_", " ").title()} vs. {x_col.replace("_", " ").title()} by Strategy',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               frameon=True, fancybox=True, shadow=False, ncol=1,
               fontsize=11, title='Model', title_fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    if save_figure:
        filename = f'mean_{x_col}_vs_{y_col}_by_strategy.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        
        # Also save as PDF for publications
        filename_pdf = f'mean_{x_col}_vs_{y_col}_by_strategy.pdf'
        plt.savefig(filename_pdf, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename_pdf}")
    
    plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Publication-ready styling
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.dpi'] = 300

def extract_strategy_number(strategy_name):
    """Extract numeric value from strategy name."""
    word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4}
    strategy_lower = str(strategy_name).lower()
    
    for word, num in word_to_num.items():
        if word in strategy_lower:
            return num
    
    # Fallback to extracting digits
    digits = ''.join(filter(str.isdigit, str(strategy_name)))
    return int(digits) if digits else 0

def get_context_info(strategy_num):
    """Get context column and display name for each strategy."""
    context_map = {
        2: ('capacity_context', 'Capacity Context'),
        3: ('economic_context', 'Economic Context'),
        4: ('user_preference_type', 'User Preference')
    }
    return context_map.get(strategy_num, (None, None))

def plot_mean_preferences_with_context(df, x_col='npi_level_selected', y_col='infection_rate_selected', save_figure=False):
    """
    Create a publication-ready multi-panel plot showing mean NPI vs mean Infection.
    One panel per strategy, with each model as a colored point and context as shape.
    Uses consistent axes across all panels for easy comparison.
    """
    
    # Calculate means for each strategy-model combination
    strategies = sorted(df['strategy'].unique(), key=extract_strategy_number)
    n_strategies = len(strategies)
    
    # Calculate global bounds for consistent axes across all panels
    all_means = []
    for strategy in strategies:
        strategy_data = df[df['strategy'] == strategy].copy()
        strategy_num = extract_strategy_number(strategy)
        context_col, _ = get_context_info(strategy_num)
        
        if context_col and context_col in strategy_data.columns:
            means = strategy_data.groupby(['model', context_col]).agg({
                x_col: 'mean',
                y_col: 'mean'
            }).reset_index()
        else:
            means = strategy_data.groupby('model').agg({
                x_col: 'mean',
                y_col: 'mean'
            }).reset_index()
        all_means.append(means)
    
    all_means_df = pd.concat(all_means, ignore_index=True)
    x_min, x_max = all_means_df[x_col].min(), all_means_df[x_col].max()
    y_min, y_max = all_means_df[y_col].min(), all_means_df[y_col].max()
    
    # Add padding for readability
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    # Set consistent limits with extra x-axis padding
    xlim = (x_min - x_range * 0.15, x_max + x_range * 0.25)
    ylim = (y_min - y_range * 0.1, y_max + y_range * 0.15)
    
    # Create subplot grid (2x2 for 4 strategies)
    n_cols = 2
    n_rows = int(np.ceil(n_strategies / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(16, 10),
                            sharex=True, sharey=True)
    
    axes = axes.flatten() if n_strategies > 1 else [axes]
    
    # Color palette for models
    models = sorted(df['model'].unique())
    palette = sns.color_palette("tab10", n_colors=len(models))
    model_colors = dict(zip(models, palette))
    
    # Marker styles for contexts
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Track which legends we've created
    model_legend_created = False
    context_legends_created = {}
    
    # Plot each strategy
    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strategy_data = df[df['strategy'] == strategy].copy()
        strategy_num = extract_strategy_number(strategy)
        
        # Get context information
        context_col, context_title = get_context_info(strategy_num)
        use_context = context_col and context_col in strategy_data.columns
        
        # Set consistent limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if use_context:
            # Calculate means for each strategy-model-context combination
            means = strategy_data.groupby(['model', context_col]).agg({
                x_col: 'mean',
                y_col: 'mean'
            }).reset_index()
            
            # Get unique contexts and assign markers
            contexts = sorted(means[context_col].unique())
            context_markers = dict(zip(contexts, markers[:len(contexts)]))
            
            # Plot each model-context combination
            for _, row in means.iterrows():
                model = row['model']
                context_val = row[context_col]
                
                ax.scatter(
                    row[x_col],
                    row[y_col],
                    s=150,
                    c=[model_colors[model]],
                    marker=context_markers[context_val],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1.5,
                    zorder=3
                )
            
            # Create model legend (color)
            model_legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=model_colors[m], markersize=8,
                           markeredgecolor='black', markeredgewidth=1.5,
                           label=m, alpha=0.8)
                for m in models
            ]
            
            first_legend = ax.legend(
                handles=model_legend_elements,
                title='Model',
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                frameon=True,
                fancybox=True,
                fontsize=10,
                title_fontsize=11,
                borderaxespad=0
            )
            ax.add_artist(first_legend)
            
            # Create context legend (shape)
            context_legend_elements = [
                plt.Line2D([0], [0], marker=context_markers[ctx], color='w',
                           markerfacecolor='gray', markersize=8,
                           markeredgecolor='black', markeredgewidth=1.5,
                           label=str(ctx), alpha=0.8)
                for ctx in contexts
            ]
            
            ax.legend(
                handles=context_legend_elements,
                title=context_title,
                bbox_to_anchor=(1.02, 0.6),
                loc='upper left',
                frameon=True,
                fancybox=True,
                fontsize=10,
                title_fontsize=11,
                borderaxespad=0
            )
        
        else:
            # Strategy 1: No context, simple plot with just model colors
            means = strategy_data.groupby('model').agg({
                x_col: 'mean',
                y_col: 'mean'
            }).reset_index()
            
            for _, row in means.iterrows():
                model = row['model']
                ax.scatter(
                    row[x_col],
                    row[y_col],
                    s=150,
                    c=[model_colors[model]],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=1.5,
                    zorder=3
                )
            
            # Add model legend for Strategy 1
            model_legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=model_colors[m], markersize=8,
                           markeredgecolor='black', markeredgewidth=1.5,
                           label=m, alpha=0.8)
                for m in models
            ]
            
            ax.legend(
                handles=model_legend_elements,
                title='Model',
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                frameon=True,
                fancybox=True,
                fontsize=10,
                title_fontsize=11,
                borderaxespad=0
            )
        
        # Format strategy title
        strategy_title = strategy.replace('_', ' ').title()
        ax.set_title(strategy_title, fontsize=14, fontweight='bold', pad=10)
        
        # Clean spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.2)
        
        # Tick formatting
        ax.tick_params(labelsize=11, width=1.2, length=5)
    
    # Hide unused subplots
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)
    
    # Add axis labels only to edge subplots
    for i in range(n_rows):
        axes[i * n_cols].set_ylabel(
            f'{y_col.replace("_", " ").title()}',
            fontsize=13, fontweight='bold'
        )
    
    for i in range(n_cols):
        if (n_rows - 1) * n_cols + i < n_strategies:
            axes[(n_rows - 1) * n_cols + i].set_xlabel(
                f'{x_col.replace("_", " ").title()}',
                fontsize=13, fontweight='bold'
            )
    
    # Overall title
    fig.suptitle(
        f'{y_col.replace("_", " ").title()} vs. {x_col.replace("_", " ").title()} by Strategy',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    if save_figure:
        filename = f'mean_{x_col}_vs_{y_col}_with_context.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        
        # Also save as PDF for publications
        filename_pdf = f'mean_{x_col}_vs_{y_col}_with_context.pdf'
        plt.savefig(filename_pdf, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename_pdf}")
    
    plt.show()




# %% [markdown]
# ##### Final Output

# %%
# Usage
plot_mean_preferences_by_strategy(
    results_df, 
    x_col='infection_rate_selected', 
    y_col='npi_level_selected', 
    save_figure=False
)

# %%
# Usage
plot_mean_preferences_with_context(
    results_df,
    x_col='infection_rate_selected',
    y_col='npi_level_selected',
    save_figure=False
)


