"""
LLM Decision Simulation with Ollama (Phi3)
------------------------------------------
Simulates LLM preferences between pairs of pandemic policy options,
asks for reasoning, stores answers and reasoning, and visualizes trends.
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
MODEL = "phi3:mini-4k"  # Updated model name
OUTPUT_DIR = "phi3_results_new"
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
num_iterations = 50
scenario_counts = {opt: {'selected': 0, 'not_selected': 0} for opt in options}

# === Query helper ===
def query_ollama(model, prompt, temperature):
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
    match = re.search(r"\b([A-I])\b", text.strip(), re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        for o in options:
            if o.startswith(f"Option {letter}"):
                return o
    return None

def get_option_letter(option_str):
    return option_str.split()[1].replace(".", "")

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
    return {"No closure": 0, "Crowded closures": 1, "Non-essential closures": 2}.get(category, -1)

def is_bad_choice(selected_opt, other_opt):
    rate_sel = extract_infection_rate(selected_opt)
    rate_oth = extract_infection_rate(other_opt)
    npi_sel = npi_level(selected_opt)
    npi_oth = npi_level(other_opt)
    return rate_sel > rate_oth and npi_sel >= npi_oth

# === Main simulation loop ===
conversation_log = []
pair_choice_counts = defaultdict(lambda: defaultdict(int))
bad_choice_counts = 0

print(f"Starting simulation: {num_iterations} iterations")

for run in tqdm(range(num_iterations), desc="LLM Runs"):
    temperature = random.uniform(0.3, 1.0)
    for pair in tqdm(option_pairs, desc=f"Iteration {run+1}", leave=False):
        prompt = (
            "Put yourself back to a time when COVID-19 was spreading in your county and no vaccine was available.\n"
            "Your county’s public health officials are considering different options for reducing the spread of COVID-19 over the next month.\n"
            "Of the options shown, which of the following would you choose?\n\n"
            "Select ONE option by stating its letter (A–I), and then explain briefly why you chose it.\n\n"
            + "\n".join(pair)
        )

        selected_choice = None
        reasoning = ""
        for _ in range(3):
            response = query_ollama(MODEL, prompt, temperature)
            choice = extract_valid_choice(response)
            if choice:
                selected_choice = choice
                # Remove choice letter from response to get reasoning
                #reasoning = response.replace(get_option_letter(choice), "").strip()
                reasoning = response  # Keep full response as reasoning
                print(f"[Info] Selected: {get_option_letter(choice)} | Reasoning: {reasoning}")
                break
        if not selected_choice:
            print(f"[Warning] Invalid response skipped: {response}")
            continue

        # Track answers per question (pair)
        pair_letters = tuple(sorted(get_option_letter(opt) for opt in pair))
        choice_letter = get_option_letter(selected_choice)
        pair_choice_counts[pair_letters][choice_letter] += 1

        # Determine not chosen option
        not_choice = [opt for opt in pair if opt != selected_choice][0]

        # Check for bad choice
        if is_bad_choice(selected_choice, not_choice):
            bad_choice_counts += 1

        # Update counts
        scenario_counts[selected_choice]['selected'] += 1
        scenario_counts[not_choice]['not_selected'] += 1

        # Log the decision
        conversation_log.append({
            "iteration": run + 1,
            "pair": [get_option_letter(o) for o in pair],
            "selected_choice": choice_letter,
            "reasoning": reasoning,
            "temperature": temperature,
            "response": response.strip(),
        })

# === Save logs ===
df_log = pd.DataFrame(conversation_log)
df_log.to_csv(os.path.join(OUTPUT_DIR, "llm_responses_with_reasoning.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "scenario_counts.json"), "w") as f:
    json.dump(scenario_counts, f, indent=2)

print(f"\nSimulation complete. Bad choices count: {bad_choice_counts}")
print(f"✅ Responses with reasoning saved in: {OUTPUT_DIR}/llm_responses_with_reasoning.csv")
