import os
import re
import json
import uuid
import requests
import pandas as pd
from itertools import combinations
from datetime import datetime
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# === Setup ===
OLLAMA_URL = "http://localhost:11434/api/generate"
ALL_MODELS = ["mistral:latest", "llama3:8b", "phi3:mini-4k", "phi4-mini:latest",
          "deepseek-llm:7b", "qwen3:4b", "llama3.2:3b", 
          "qwen3:8b", "gemma3:1b", "gpt-oss:20b", "gemma3:4b"]
MODELS = ["phi4-mini:latest"]
OUTPUT_DIR = "strategy_two_results"
NUM_WORKERS = 2
TEMPERATURE = 0.1


os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_lock = threading.Lock()


# === Context Variations ===
COUNTY_TYPES = ["rural", "urban", "suburban"]
STATES = ["California", "Texas", "Florida", "New York"]

# Generate all combinations of county type and state
PROMPT_CONTEXTS = []
for county_type in COUNTY_TYPES:
    for state in STATES:
        PROMPT_CONTEXTS.append({
            "county_type": county_type,
            "state": state,
        })


# === Canonical Options ===
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
canonical_option_map = {re.search(r"Option\s+([A-I])", o).group(1): o for o in options}


# === Prompt Builder ===
def build_prompt(context, pair):
    """Build a prompt with county type and state variations."""
    
    prompt = (
        f"You are a public health official for a {context['county_type']} county in {context['state']}.\n"
        "Put yourself back to a time when COVID-19 was spreading and no vaccine was available.\n"
        "You are considering different options for reducing the spread of COVID-19 over the next month. "
        "You must balance public health protection with economic impacts on local businesses and workers.\n"
        "Of the options shown, which would you choose?\n\n"
        "Please reply with only the option letter (A–I) that you would choose.\n\n"
        "Options:\n" + "\n".join(pair)
    )
    
    return prompt


# === Ollama Query ===
def query_ollama(model, prompt, temperature=0.1):
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
def parse_choice(response_text):
    if not response_text:
        return None
    clean_response = response_text.strip()
    if len(clean_response) == 1 and clean_response.upper() in 'ABCDEFGHI':
        return clean_response.upper()
    single_letter_match = re.match(r'^\s*([A-I])\s*[.,;:!]?\s*$', clean_response, re.IGNORECASE)
    if single_letter_match:
        return single_letter_match.group(1).upper()
    patterns = [
        r"^\s*[12]([A-I])\b",
        r"Option\s+([A-I])\b",
        r"(?:choose|select|prefer|pick)\s+Option\s+([A-I])\b",
        r"^Option\s+([A-I])\b",
        r"^\s*([A-I])\s*[.:\-]",
        r"(?:My choice is|choice is|choice:)\s+(?:Option\s+)?([A-I])\b",
        r"\b([A-I])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


# === Infection/NPI helpers ===
def extract_infection_rate(option_text):
    match = re.search(r"(\d+)", option_text)
    return int(match.group(1)) if match else None


def categorize_closure(option_text):
    if "No businesses are closed" in option_text:
        return "No closure"
    elif "Crowded, high-risk businesses" in option_text:
        return "Crowded closures"
    elif "All non-essential businesses" in option_text:
        return "Non-essential closures"
    return "Other"


def npi_level(option_text):
    cat = categorize_closure(option_text)
    return {"No closure": 0, "Crowded closures": 1, "Non-essential closures": 2}.get(cat, -1)


# === CSV Logger ===
def log_response_to_csv(model, context, pair, selected_choice, response_text, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    selected_option = canonical_option_map.get(selected_choice)
    if not selected_option:
        print(f"[Warning] Could not find canonical option for {selected_choice}")
        return

    non_selected_option = next((o for o in pair if o != selected_option), None)
    infection_rate_sel = extract_infection_rate(selected_option)
    npi_sel = npi_level(selected_option)
    infection_rate_non = extract_infection_rate(non_selected_option) if non_selected_option else None
    npi_non = npi_level(non_selected_option) if non_selected_option else None
    option_letters = [re.search(r"Option\s+([A-I])", o).group(1) for o in pair]

    entry_id = (
        f"{model.replace(':', '_')}_{context['county_type']}_{context['state'].replace(' ', '')}_{option_letters[0]}_{option_letters[1]}_"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )

    record = {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "county_type": context["county_type"],
        "state": context["state"],
        "pair": f"{option_letters[0]}-{option_letters[1]}",
        "option_1": pair[0],
        "option_2": pair[1],
        "selected_choice": selected_choice,
        "response_text": response_text,
        "infection_rate_selected": infection_rate_sel,
        "npi_level_selected": npi_sel,
        "infection_rate_non_selected": infection_rate_non,
        "npi_level_non_selected": npi_non,
        "temperature": TEMPERATURE,
    }

    csv_path = os.path.join(output_dir, f"results_{model.replace(':', '_')}.csv")
    with csv_lock:
        df = pd.DataFrame([record])
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", index=False, header=False)


# === Process Single Pair ===
def process_pair(args):
    model, context, pair = args
    
    prompt = build_prompt(context, pair)
    response = query_ollama(model, prompt, TEMPERATURE)
    
    if not response:
        return False
    
    choice_letter = parse_choice(response)
    if not choice_letter:
        print(f"[Warning] Invalid response for {model}: {response[:120]}")
        return False

    log_response_to_csv(model, context, pair, choice_letter, response, OUTPUT_DIR)
    return True


# === Parallel Simulation ===
def run_simulation():
    print(f"Starting simulation with models: {', '.join(MODELS)}")
    print(f"Using {NUM_WORKERS} workers")
    print(f"Prompt contexts: {len(PROMPT_CONTEXTS)}")
    print(f"Option pairs: {len(option_pairs)}")

    tasks = [(model, context, pair) for model in MODELS for context in PROMPT_CONTEXTS for pair in option_pairs]
    print(f"Total tasks: {len(tasks)}")

    successful, failed = 0, 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_pair, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Processing") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"[Error] Task failed: {e}")
                    failed += 1
                finally:
                    pbar.update(1)

    print(f"\nSimulation complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Results saved in {OUTPUT_DIR}/")
    for model in MODELS:
        print(f"  - results_{model.replace(':', '_')}.csv")


if __name__ == "__main__":
    run_simulation()
