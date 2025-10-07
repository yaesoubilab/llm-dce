import json
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# File paths (adjust as needed)
INPUT_PATH = "deepseek_results_new/llm_responses.json"
OUTPUT_JSON_PATH = "deepseek_results_new/bad_choices.json"
OUTPUT_PAIR_COUNT_PNG = "deepseek_results_new/bad_choices_pair_counts.png"
OUTPUT_SELECTION_PNG = "deepseek_results_new/bad_choices_selection_counts.png"

# Options list, must exactly match main simulation script's options
options = [
    "Option A. No businesses are closed; 1 in 100 people will be infected during the next month.",
    "Option B. No businesses are closed; 10 in 100 people will be infected during the next month.",
    "Option C. No businesses are closed; 20 in 100 people will be infected during the next month.",
    "Option D. Crowded, high-risk businesses (such as clubs, theaters, and large sports venues) are closed; 1 in 100 people will be infected during the next month.",
    "Option E. Crowded, high-risk businesses (such as clubs, theaters, and large sports venues) are closed; 10 in 100 people will be infected during the next month.",
    "Option F. Crowded, high-risk businesses (such as clubs, theaters, and large sports venues) are closed; 20 in 100 people will be infected during the next month.",
    "Option G. All non-essential businesses (such as clubs, theaters, restaurants, bars, and gyms) are closed; 1 in 100 people will be infected during the next month.",
    "Option H. All non-essential businesses (such as clubs, theaters, restaurants, bars, and gyms) are closed; 10 in 100 people will be infected during the next month.",
    "Option I. All non-essential businesses (such as clubs, theaters, restaurants, bars, and gyms) are closed; 20 in 100 people will be infected during the next month.",
]

# Helper functions from main simulation
def extract_infection_rate(option):
    match = re.search(r"(\d+)", option)
    return int(match.group(1)) if match else None

def categorize_closure(option):
    if "No businesses are closed" in option:
        return "No closure"
    elif "Crowded, high-risk businesses" in option:
        return "Crowded closures"
    elif "All non-essential businesses" in option:
        return "Non-essential closures"
    return "Other"

def npi_level(option):
    category = categorize_closure(option)
    if category == "No closure":
        return 0
    elif category == "Crowded closures":
        return 1
    elif category == "Non-essential closures":
        return 2
    return -1

def is_bad_choice(selected_opt, other_opt):
    rate_sel = extract_infection_rate(selected_opt)
    rate_oth = extract_infection_rate(other_opt)
    npi_sel = npi_level(selected_opt)
    npi_oth = npi_level(other_opt)
    return rate_sel > rate_oth and npi_sel >= npi_oth

# Load conversation log
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

bad_choices = []
pair_counts = defaultdict(int)
scenario_selection_counts = defaultdict(lambda: {'selected':0, 'not_selected':0})

option_map = {opt.split()[1].replace(".", ""): opt for opt in options}

for entry in data:
    pair_letters = entry["pair"]  # e.g. ["A", "C"]
    selected_letter = entry.get("selected_choice") or entry.get("choice")
    if selected_letter is None:
        continue
    selected_opt = option_map[selected_letter]
    other_letter = [l for l in pair_letters if l != selected_letter][0]
    other_opt = option_map[other_letter]

    if is_bad_choice(selected_opt, other_opt):
        bad_choices.append(entry)
        # Track which pairs the bad choices appear in (sorted for consistency)
        sorted_pair = tuple(sorted(pair_letters))
        pair_counts[sorted_pair] += 1
        # Track scenario selection counts within bad choices
        scenario_selection_counts[selected_letter]['selected'] += 1
        scenario_selection_counts[other_letter]['not_selected'] += 1

# Save bad choices JSON
with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(bad_choices, f, indent=2)

print(f"Saved {len(bad_choices)} bad choices to {OUTPUT_JSON_PATH}")

# Prepare data for plotting pair counts
pair_labels = [" & ".join(pair) for pair in pair_counts.keys()]
pair_values = [pair_counts[pair] for pair in pair_counts.keys()]

plt.figure(figsize=(14,6))
plt.bar(pair_labels, pair_values, color="#CD5C5C")
plt.xticks(rotation=90, fontsize=8)
plt.xlabel("Question Pairs (Scenario Letters)")
plt.ylabel("Bad Choice Count")
plt.title("Bad Choice Frequency by Question Pair")
plt.tight_layout()
plt.savefig(OUTPUT_PAIR_COUNT_PNG)
plt.close()

print(f"Saved bad choice pair counts plot to {OUTPUT_PAIR_COUNT_PNG}")

# Prepare data for scenario selection paired bar graph within bad choices
df_scenarios = pd.DataFrame.from_dict(scenario_selection_counts, orient='index').fillna(0)
df_scenarios.index.name = "Scenario"
df_scenarios = df_scenarios.sort_index()

plt.figure(figsize=(10,6))
bar_width = 0.4
indices = range(len(df_scenarios))

plt.bar(indices, df_scenarios["selected"], bar_width, label='Selected (Bad Choices)', color="#2E8B57")
plt.bar([i + bar_width for i in indices], df_scenarios["not_selected"], bar_width, label='Not Selected (Good Choices)', color="#CD5C5C")

plt.xlabel("Scenario Letter")
plt.ylabel("Count")
plt.title("Scenario Selection Counts Within Bad Choices")
plt.xticks([i + bar_width / 2 for i in indices], df_scenarios.index, rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_SELECTION_PNG)
plt.close()

print(f"Saved scenario selection counts plot within bad choices to {OUTPUT_SELECTION_PNG}")
