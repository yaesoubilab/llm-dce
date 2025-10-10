"""
LLM Decision Simulation with Ollama (Llama3)
---------------------------------------------
Simulates LLM preferences between pairs of pandemic policy options,
records both the chosen option and reasoning,
tracks selection frequencies and visualizes trends.
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
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"
OUTPUT_DIR = "llama_results_reasoning_combined"
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
num_iterations = 5
scenario_counts = {opt: {'selected': 0, 'not_selected': 0} for opt in options}

# === Ollama query ===
def query_ollama(model, prompt, temperature=0.7):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature},
        "stream": True
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[Error] Request failed: {e}")
        return ""

    full_response = ""
    for line in response.iter_lines():
        if not line:
            continue
        try:
            parsed = json.loads(line)
            # /api/generate uses "response" field, not "message"
            if "response" in parsed:
                full_response += parsed["response"]
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON decode error: {e}")
            continue

    return full_response.strip()

# === Parsing ===
def parse_choice_and_reasoning(response_text):
    """
    Extracts the chosen option (A–I) and reasoning text from model output.
    Returns (choice_letter, reasoning_text).
    """
    if not response_text:
        return None, ""
    
    # Look for the letter more flexibly - at start of line or after common phrases
    # IMPORTANT: Order matters! More specific patterns first to avoid false matches
    patterns = [
        r"Option\s+([A-I])\b",  # "Option X" - most specific, check first
        r"(?:choose|select|prefer)\s+Option\s+([A-I])\b",  # "choose/select Option X"
        r"(?:choose|select|prefer)\s+([A-I])\b(?!\s+would)",  # After action verb, but not "I would"
        r"^([A-I])\b",  # Letter at start of response
        r"^(?:My choice is|I choose|I select|I prefer)\s+([A-I])\b",  # Explicit choice statements at start
        r"\b([A-I])\s*[.:](?=\s)",  # Letter followed by punctuation and space
    ]
    
    choice_letter = None
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            potential_letter = match.group(1).upper()
            # Double-check it's not the pronoun "I" by looking at context
            # If the letter is I, make sure it appears as "Option I" or similar
            if potential_letter == 'I':
                # Only accept if it's clearly "Option I" not pronoun "I"
                if re.search(r"\bOption\s+I\b", response_text, re.IGNORECASE):
                    choice_letter = 'I'
                    break
                else:
                    continue  # Skip this match, try next pattern
            else:
                choice_letter = potential_letter
                break
    
    # Extract reasoning - everything is reasoning
    reasoning = response_text.strip()
    
    # Optionally clean up the reasoning to remove the choice statement
    if choice_letter:
        # Remove common choice declaration patterns
        reasoning = re.sub(
            r"^.*?(?:choose|select|prefer|option)\s+" + choice_letter + r"\b[^\n]*\n?",
            "",
            reasoning,
            flags=re.IGNORECASE
        ).strip()
        
        # Also try removing if it's just the letter at the start
        reasoning = re.sub(r"^" + choice_letter + r"\b[.:\s]*", "", reasoning).strip()

    return choice_letter, reasoning


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

# === Simulation ===
conversation_log = []
pair_choice_counts = defaultdict(lambda: defaultdict(int))
bad_choice_counts = 0

print(f"Starting simulation: {num_iterations} iterations")

for run in tqdm(range(num_iterations), desc="LLM Runs"):
    temperature = random.uniform(0.3, 1.0)
    for pair in tqdm(option_pairs, desc=f"Iteration {run+1}", leave=False):

        prompt_general = (
            "Put yourself back to a time when COVID-19 was spreading in your county and no vaccine was available. "
            "Your county's public health officials are considering different options for reducing the spread of COVID-19 over the next month. "
            "Of the options shown, which would you choose?\n\n"
            "Please reply with:\n"
            "1. First, state your chosen option letter (A-I)\n"
            "2. Then explain your reasoning in 2-3 sentences, focusing on the tradeoffs between infection rate and business closures.\n\n"
            "Options:\n"
            + "\n".join(pair)
        )

        # Query and parse
        response = query_ollama(MODEL, prompt_general, temperature)
        
        if not response:
            print(f"[Warning] Empty response received, skipping pair")
            continue
            
        choice_letter, reasoning = parse_choice_and_reasoning(response)
        #print(f"\n[Debug] Full Response:\n{response}\n")
        #print(f"[Debug] Parsed - Choice: {choice_letter}, Reasoning: {reasoning[:100]}...")

        if not choice_letter:
            print(f"[Warning] Invalid response - no choice letter found. Response: {response[:200]}")
            continue

        selected_choice = next((o for o in options if o.startswith(f"Option {choice_letter}")), None)
        if not selected_choice:
            print(f"[Warning] Could not match choice letter '{choice_letter}' to options in pair")
            continue
        
        # Verify the choice is actually one of the two options in the pair
        if selected_choice not in pair:
            print(f"[Warning] Selected choice '{choice_letter}' not in current pair {[get_option_letter(o) for o in pair]}")
            continue

        # Track counts
        pair_letters = tuple(sorted(get_option_letter(opt) for opt in pair))
        pair_choice_counts[pair_letters][choice_letter] += 1

        not_choice = [opt for opt in pair if opt != selected_choice][0]
        if is_bad_choice(selected_choice, not_choice):
            bad_choice_counts += 1

        scenario_counts[selected_choice]['selected'] += 1
        scenario_counts[not_choice]['not_selected'] += 1

        # Log entry
        conversation_log.append({
            "iteration": run + 1,
            "pair": [get_option_letter(o) for o in pair],
            "selected_choice": choice_letter,
            "reasoning": reasoning,
            "temperature": temperature,
            "full_response": response
        })

# === Save results ===
with open(os.path.join(OUTPUT_DIR, "llm_responses_with_reasoning.json"), "w") as f:
    json.dump(conversation_log, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "scenario_counts.json"), "w") as f:
    json.dump(scenario_counts, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "pair_choice_counts.json"), "w") as f:
    json.dump({str(k): dict(v) for k, v in pair_choice_counts.items()}, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "reasoning_log.txt"), "w") as f:
    for entry in conversation_log:
        f.write(f"Run {entry['iteration']} | Pair {entry['pair']} | Choice {entry['selected_choice']}\n")
        f.write(f"Reasoning: {entry['reasoning']}\n")
        f.write(f"Full Response: {entry['full_response']}\n")
        f.write("-" * 80 + "\n\n")

print(f"\nSimulation complete. Bad choices count: {bad_choice_counts}")
print(f"Valid responses: {len(conversation_log)}")
print(f"✅ Results saved in: {OUTPUT_DIR}/")