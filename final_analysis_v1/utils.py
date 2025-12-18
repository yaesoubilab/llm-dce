
import pandas as pd
import os
import glob
import numpy as np

def load_all_data(strategies_dict, subsample_size=None, random_seed=42):
    """
    Load and combine data from all strategy directories.
    
    Args:
        strategies_dict (dict): Dictionary mapping strategy names to directory paths.
        subsample_size (int, optional): Number of samples per model/strategy to take. 
                                      If None, loads full dataset.
        random_seed (int): Seed for reproducibility in subsampling.
        
    Returns:
        pd.DataFrame: Combined dataframe containing all results.
    """
    all_dfs = []
    
    print("Loading data from strategies...")
    for strategy_name, directory in strategies_dict.items():
        dir_path = os.path.join('.', directory)
        if not os.path.exists(dir_path):
            # Try absolute path or relative to current script if '.' didn't work as expected
            # But usually '.' refers to CWD. If the script is run from final_analysis, '.' is fine if directories are there.
            # If directories are siblings-ish, we might need adjustment. 
            # Assuming user runs notebooks from `final_analysis/` and strategy dirs are inside `final_analysis/` ?
            # Based on previous file listing, strategy directories seem to be in `final_analysis/` implicitly or `llm-dce/` root?
            # The config says 'strategy_one_results', etc. 
            # Let's assume the CWD is correct for now, or just warn.
            print(f"Warning: Directory '{dir_path}' not found. Skipping...")
            continue
            
        csv_files = glob.glob(os.path.join(dir_path, 'results_*.csv'))
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            try:
                df = pd.read_csv(csv_file)
                df['strategy'] = strategy_name
                all_dfs.append(df)
                print(f"  Loaded: {filename} ({len(df)} rows)")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")

    if all_dfs:
        results_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✓ Successfully combined {len(all_dfs)} files. Total rows: {len(results_df)}")
        
        # Apply subsampling if configured
        if subsample_size is not None and subsample_size > 0:
            print(f"\nApplying subsampling: {subsample_size} samples per model/strategy...")
            try:
                # Group by model and strategy, then sample
                results_df = results_df.groupby(['model', 'strategy'], group_keys=False).apply(
                    lambda x: x.sample(min(len(x), subsample_size), random_state=random_seed)
                )
                print(f"✓ Subsampling complete. New total rows: {len(results_df)}")
            except Exception as e:
                print(f"Warning: Subsampling failed: {e}. Using full dataset.")
        
        print(f"✓ Models found: {results_df['model'].unique().tolist()}")
        print(f"✓ Strategies found: {results_df['strategy'].unique().tolist()}")
        return results_df
    else:
        print("❌ No data files were found or loaded.")
        return pd.DataFrame()

def save_analysis_results(df, output_dir, analysis_type):
    """
    Save analysis results to CSV files, split by model and strategy.
    
    Args:
        df (pd.DataFrame): The dataframe with analysis results.
        output_dir (str): The base directory to save into.
        analysis_type (str): 'llm_judge' or 'traditional' suffix for filename.
    """
    if df.empty:
        print("No results to save.")
        return

    print(f"\nSaving {analysis_type} results to {output_dir}...")
    
    # Iterate through unique model and strategy combinations
    groups = df.groupby(['model', 'strategy'])
    
    for (model, strategy), group_df in groups:
        # Sanitize model name to be filename safe (replace / with _)
        safe_model = model.replace('/', '_').replace(':', '_')
        filename = f"{safe_model}_{strategy}_{analysis_type}.csv"
        filepath = os.path.join(output_dir, filename)
        
        try:
            group_df.to_csv(filepath, index=False)
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Error saving {filename}: {e}")
            
    print("✓ Saving complete.")
