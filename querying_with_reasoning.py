"""
LLM Decision Simulation with Ollama (Llama3)
---------------------------------------------
Simulates LLM preferences between pairs of pandemic policy options,
records the chosen option, reasoning, infection rate, and NPI level
for both selected and non-selected options,
and saves results in per-model CSVs for later cross-model comparison.
"""

import os
import re
import json
import random
import uuid
import requests
import pandas as pd
from itertools import combinations
from datetime import datetime
from tqdm.auto import tqdm

# === Setup ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini-4k"
OUTPUT_DIR = "combined_results"
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
num_iterations = 2

# === Ollama Query ===
def query_ollama(model, prompt, temperature=0.7):
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature},
        "stream": True,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60)
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
            if "response" in parsed:
                full_response += parsed["response"]
        except json.JSONDecodeError:
            continue

    return full_response.strip()

# === Parsing ===
def parse_choice_and_reasoning(response_text):
    """Extract chosen option (A–I) and reasoning."""
    if not response_text:
        return None, ""

    patterns = [
        r"Option\s+([A-I])\b",
        r"(?:choose|select|prefer)\s+Option\s+([A-I])\b",
        r"(?:choose|select|prefer)\s+([A-I])\b",
        r"^([A-I])\b",
        r"^(?:My choice is|I choose|I select|I prefer)\s+([A-I])\b",
        r"\b([A-I])\s*[.:](?=\s)",
    ]

    choice_letter = None
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            potential_letter = match.group(1).upper()
            if potential_letter == "I":
                if re.search(r"\bOption\s+I\b", response_text, re.IGNORECASE):
                    choice_letter = "I"
                    break
                else:
                    continue
            else:
                choice_letter = potential_letter
                break

    reasoning = response_text.strip()
    if choice_letter:
        reasoning = re.sub(
            r"^.*?(?:choose|select|prefer|option)\s+" + choice_letter + r"\b[^\n]*\n?",
            "",
            reasoning,
            flags=re.IGNORECASE,
        ).strip()
        reasoning = re.sub(r"^" + choice_letter + r"\b[.:\s]*", "", reasoning).strip()

    return choice_letter, reasoning

# === Infection/NPI helpers ===
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
    cat = categorize_closure(option)
    return {"No closure": 0, "Crowded closures": 1, "Non-essential closures": 2}.get(cat, -1)

# === CSV Logger ===
def log_response_to_csv(
    model: str,
    iteration: int,
    pair: list,
    selected_choice: str,
    reasoning: str,
    temperature: float,
    response_text: str,
    output_dir: str = "results",
):
    """Logs a single LLM decision with selected + non-selected metadata."""
    os.makedirs(output_dir, exist_ok=True)

    option_letters = [opt.split()[1].replace(".", "") for opt in pair]
    option_texts = [opt for opt in pair]

    selected_option = next((o for o in option_texts if o.startswith(f"Option {selected_choice}")), None)
    if not selected_option:
        selected_option = "N/A"

    # Identify the non-selected option
    non_selected_option = next((o for o in option_texts if not o.startswith(f"Option {selected_choice}")), None)

    # Extract infection & NPI data for both
    infection_rate_sel = extract_infection_rate(selected_option)
    npi_sel = npi_level(selected_option)
    infection_rate_non = extract_infection_rate(non_selected_option)
    npi_non = npi_level(non_selected_option)

    entry_id = (
        f"{model.replace(':', '_')}_"
        f"{iteration}_"
        f"{option_letters[0]}_{option_letters[1]}_"
        f"{temperature:.2f}_"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    record = {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "iteration": iteration,
        "pair": f"{option_letters[0]}-{option_letters[1]}",
        "option_1": option_texts[0],
        "option_2": option_texts[1],
        "selected_choice": selected_choice,
        "reasoning": reasoning,
        "response_text": response_text,
        "infection_rate_selected": infection_rate_sel,
        "npi_level_selected": npi_sel,
        "infection_rate_non_selected": infection_rate_non,
        "npi_level_non_selected": npi_non,
        "temperature": round(temperature, 3),
    }

    csv_path = os.path.join(output_dir, f"results_{model.replace(':', '_')}.csv")
    df = pd.DataFrame([record])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", index=False, header=False)

    return record

# === Simulation ===
print(f"Starting simulation with model: {MODEL}")

for run in tqdm(range(num_iterations), desc="LLM Runs"):
    temperature = random.uniform(0.3, 1.0)

    for pair in tqdm(option_pairs, desc=f"Iteration {run + 1}", leave=False):
        prompt = (
            "Put yourself back to a time when COVID-19 was spreading in your county and no vaccine was available. "
            "Your county's public health officials are considering different options for reducing the spread of COVID-19 over the next month. "
            "Of the options shown, which would you choose?\n\n"
            "Please reply with:\n"
            "1. First, state your chosen option letter (A–I)\n"
            "2. Then explain your reasoning in 2–3 sentences, focusing on the tradeoffs between infection rate and business closures.\n\n"
            "Options:\n" + "\n".join(pair)
        )

        response = query_ollama(MODEL, prompt, temperature)
        if not response:
            continue

        choice_letter, reasoning = parse_choice_and_reasoning(response)
        if not choice_letter:
            print(f"[Warning] Invalid response: {response[:120]}")
            continue

        log_response_to_csv(
            model=MODEL,
            iteration=run + 1,
            pair=pair,
            selected_choice=choice_letter,
            reasoning=reasoning,
            temperature=temperature,
            response_text=response,
            output_dir=OUTPUT_DIR,
        )

print(f"\n✅ Simulation complete. Results saved in {OUTPUT_DIR}/results_{MODEL}.csv")