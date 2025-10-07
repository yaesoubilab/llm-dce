"""
LLM Decision Simulation with Ollama (Llama3)
---------------------------------------------
Simulates LLM preferences between pairs of pandemic policy options,
tracks selection frequencies, and visualizes trends by infection rate
and business closure level.
"""

import os
import re
import json
import random
import requests
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from tqdm.auto import tqdm

# === Setup ===
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Options ===
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

option_pairs = list(combinations(options, 2))

# === Simulation parameters ===
num_iterations = 500 # you can increase later
scenario_counts = {opt: {'selected': 0, 'not_selected': 0} for opt in options}

# === Ollama query helper ===
def query_ollama(model, prompt, temperature):
    """Send prompt to local Ollama instance and return the model’s text output."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    try:
        response = requests.post(OLLAMA_URL, json=data, timeout=10)
        response.raise_for_status()
        for line in response.text.splitlines():
            try:
                parsed = json.loads(line)
                return parsed["message"]["content"].strip()
            except json.JSONDecodeError:
                continue
    except requests.RequestException as e:
        print(f"[Warning] Request error: {e}")
    return ""

# === Parsing helpers ===
def extract_valid_choice(text):
    """Extract a valid option letter (A–I) and return its full option string."""
    match = re.search(r"\b([A-I])\b", text.strip(), re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        for o in options:
            if o.startswith(f"Option {letter}"):
                return o
    return None

def get_option_letter(option_str):
    """Convert full option text → single letter (e.g. 'Option A.' → 'A')."""
    return option_str.split()[1].replace(".", "")

def extract_infection_rate(option):
    """Return infection rate as integer (e.g. 10 from '10 in 100')."""
    return int(re.search(r"(\d+)", option).group(1))

def categorize_closure(option):
    """Categorize business closure level based on keywords."""
    if "No businesses are closed" in option:
        return "No closure"
    elif "Crowded, high-risk businesses" in option:
        return "Crowded closures"
    elif "All non-essential businesses" in option:
        return "Non-essential closures"
    return "Other"

# === Main simulation ===
conversation_log = []

for run in tqdm(range(num_iterations), desc="LLM Runs"):
    temperature = random.uniform(0.3, 1.0)

    for pair in tqdm(option_pairs, desc=f"Iteration {run + 1}", leave=False):
        prompt = (
            "I am an official public health decisionmaker in charge of controlling the spread of SARS-CoV-2 in my county. I need to select between the following two scenarios. If it were up to you, which one would you have chosen?\n\n"
            "Select ONE option strictly by stating only its letter (A–I). "
            "No explanations, just the letter. Which do you choose?\n\n"
            + "\n".join(pair)
        )

        # Retry up to 3 times to get a valid A–I response
        selected_choice = None
        for _ in range(3):
            response = query_ollama(MODEL, prompt, temperature)
            choice = extract_valid_choice(response)
            if choice:
                selected_choice = choice
                break
        if not selected_choice:
            print(f"[Warning] Invalid response skipped: {response}")
            continue

        # Update counts
        not_choice = [opt for opt in pair if opt != selected_choice][0]
        scenario_counts[selected_choice]['selected'] += 1
        scenario_counts[not_choice]['not_selected'] += 1

        # Log conversation
        conversation_log.append({
            "iteration": run + 1,
            "temperature": temperature,
            "pair": pair,
            "choice": selected_choice,
            "response": response.strip(),
        })

# === Save results ===
with open(os.path.join(OUTPUT_DIR, "llm_responses.json"), "w") as f:
    json.dump(conversation_log, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "scenario_counts.json"), "w") as f:
    json.dump(scenario_counts, f, indent=2)

# === Convert results to DataFrame ===
df = pd.DataFrame(scenario_counts).T
df.index = df.index.map(get_option_letter)
df["infection_rate"] = [extract_infection_rate(o) for o in options]
df["closure_category"] = [categorize_closure(o) for o in options]

# === Figure 1: Infection rate distribution ===
infection_df = (
    df.groupby("infection_rate")[["selected", "not_selected"]].sum().reset_index()
)
plt.figure(figsize=(8, 5))
bar_width = 0.35
x = range(len(infection_df))
plt.bar(x, infection_df["selected"], width=bar_width, label="Selected", color="#4caf50")
plt.bar([i + bar_width for i in x], infection_df["not_selected"], width=bar_width,
        label="Not Selected", color="#e57373")
plt.xticks([i + bar_width / 2 for i in x], infection_df["infection_rate"])
plt.xlabel("Infection Rate (per 100 people)")
plt.ylabel("Count")
plt.title("Distribution of Infection Rates (Selected vs. Not Selected)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "infection_rate_distribution_bar.png"))
plt.close()

# === Figure 2: Business closure frequencies ===
closure_df = (
    df.groupby("closure_category")[["selected", "not_selected"]].sum().reset_index()
)
plt.figure(figsize=(8, 5))
bar_width = 0.35
x = range(len(closure_df))
plt.bar(x, closure_df["selected"], width=bar_width, label="Selected", color="#4caf50")
plt.bar([i + bar_width for i in x], closure_df["not_selected"], width=bar_width,
        label="Not Selected", color="#e57373")
plt.xticks([i + bar_width / 2 for i in x], closure_df["closure_category"], rotation=20)
plt.xlabel("Business Closure Category")
plt.ylabel("Count")
plt.title("Business Closure Category Counts (Selected vs. Not Selected)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "business_closure_bar.png"))
plt.close()

# === Figure 3: Scenario-level counts ===
plt.figure(figsize=(12, 6))
bar_width = 0.4
index = range(len(df))
plt.bar(index, df["selected"], bar_width, label="Selected", color="#2E8B57")
plt.bar([i + bar_width for i in index], df["not_selected"], bar_width,
        label="Not Selected", color="#CD5C5C")
plt.xlabel("Scenario (A–I)")
plt.ylabel("Count")
plt.title("Scenario Selection Counts")
plt.xticks([i + bar_width / 2 for i in index], df.index, rotation=35, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scenario_selection_counts.png"))
plt.close()

print(f"\n✅ All results saved in: {OUTPUT_DIR}/")
