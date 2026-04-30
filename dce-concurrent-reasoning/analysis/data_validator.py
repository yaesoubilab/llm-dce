import os
import glob
import pandas as pd
import numpy as np

def validate_results(base_path, fix=False):
    """
    Scans for CSV results and identifies problematic rows.
    If fix=True, it backups the original and removes invalid rows 
    so querying scripts can rerun them.
    """
    # Look for results in both querying and analysis folders if needed
    search_patterns = [
        os.path.join(base_path, "querying/strategy_*/strategy_*_results/results_*.csv"),
        os.path.join(base_path, "analysis/raw_results/strategy_*_results/results_*.csv")
    ]
    
    files = []
    for pattern in search_patterns:
        files.extend(glob.glob(pattern))
    
    if not files:
        print(f"No result files found in {base_path}")
        return

    print(f"Found {len(files)} files. Validating...")
    print("-" * 50)
    
    total_invalid = 0
    files_with_issues = 0

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            
            # Criteria for invalidity
            # 1. Reasoning is missing, null, or just whitespace
            # 2. selected_choice is missing (LLM failed to answer)
            # 3. response_text is missing
            
            mask_reasoning_empty = df['reasoning'].isna() | (df['reasoning'].astype(str).str.strip() == "")
            mask_choice_empty = df['selected_choice'].isna() | (df['selected_choice'].astype(str).str.strip() == "")
            mask_response_empty = df['response_text'].isna() | (df['response_text'].astype(str).str.strip() == "")
            
            invalid_mask = mask_reasoning_empty | mask_choice_empty | mask_response_empty
            num_invalid = invalid_mask.sum()
            
            if num_invalid > 0:
                print(f"[ISSUE] {file_path}")
                print(f"  - Total Rows: {len(df)}")
                print(f"  - Empty Reasoning: {mask_reasoning_empty.sum()}")
                print(f"  - Empty Choice: {mask_choice_empty.sum()}")
                print(f"  - Empty Response: {mask_response_empty.sum()}")
                print(f"  - TOTAL INVALID: {num_invalid}")
                
                total_invalid += num_invalid
                files_with_issues += 1
                
                if fix:
                    # Backup
                    backup_path = file_path + ".bak"
                    if not os.path.exists(backup_path):
                        os.rename(file_path, backup_path)
                        print(f"  - Created backup: {os.path.basename(backup_path)}")
                    
                    # Write only valid rows
                    valid_df = df[~invalid_mask]
                    valid_df.to_csv(file_path, index=False)
                    print(f"  - Cleaned file saved. Rerun querying script to fill {num_invalid} gaps.")
                print("-" * 30)
                
        except Exception as e:
            print(f"[ERROR] Could not process {file_path}: {e}")

    print("-" * 50)
    print(f"SUMMARY:")
    print(f"Files scanned: {len(files)}")
    print(f"Files with issues: {files_with_issues}")
    print(f"Total invalid rows found: {total_invalid}")
    if fix:
        print("ACTION: Files have been cleaned. Rerunning your 'query_strategy_*.py' scripts will now attempt to fill these missing rows.")
    else:
        print("TIP: Run with '--fix' to automatically remove these rows so they can be re-queried.")

if __name__ == "__main__":
    import sys
    base = "/Users/rpenchala/Documents/GitHub/llm-dce/new/"
    should_fix = "--fix" in sys.argv
    validate_results(base, fix=should_fix)
