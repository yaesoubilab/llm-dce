"""
LLM Decision Simulation with Ollama - State-Based Parallel Processing
---------------------------------------------------------------------
Simulates LLM preferences between pairs of pandemic policy options
from the perspective of public health officials in all 50 US states.
Uses temperature = 0.1 and requests only the chosen option letter.
"""

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
MODEL = "gpt-oss:20b"
OUTPUT_DIR = "state_results"
NUM_WORKERS = 2  # Number of parallel workers
TEMPERATURE = 0.1  # Fixed temperature

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thread-safe file writing
csv_lock = threading.Lock()

# === US States ===
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

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

# === Canonical lookup map ===
canonical_option_map = {re.search(r"Option\s+([A-I])", o).group(1): o for o in options}

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
    """Extract chosen option (A–I)."""
    if not response_text:
        return None

    # Strip whitespace and get clean response
    clean_response = response_text.strip()
    
    # First check: Is it just a single letter A-I?
    if len(clean_response) == 1 and clean_response.upper() in 'ABCDEFGHI':
        return clean_response.upper()
    
    # Second check: Single letter with minimal punctuation
    single_letter_match = re.match(r'^\s*([A-I])\s*[.,;:!]?\s*$', clean_response, re.IGNORECASE)
    if single_letter_match:
        return single_letter_match.group(1).upper()
    
    # Patterns ordered from most specific to least specific
    patterns = [
        r"^\s*[12]([A-I])\b",  # Numeric prefix format: "1A", "2E"
        r"Option\s+([A-I])\b",  # "Option X"
        r"(?:choose|select|prefer|pick)\s+Option\s+([A-I])\b",
        r"^Option\s+([A-I])\b",
        r"^\s*([A-I])\s*[.:\-]",  # Letter at start followed by punctuation
        r"(?:My choice is|choice is|choice:)\s+(?:Option\s+)?([A-I])\b",
        r"\b([A-I])\b",  # Any single letter A-I as a word
    ]

    choice_letter = None
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            potential_letter = match.group(1).upper()
            choice_letter = potential_letter
            break

    return choice_letter


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


# === CSV Logger (Thread-Safe) ===
def log_response_to_csv(
    model: str,
    state: str,
    pair: list,
    selected_choice: str,
    response_text: str,
    output_dir: str = "state_results",
):
    """Logs a single LLM decision with selected + non-selected metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Canonical lookup for selected option
    selected_option = canonical_option_map.get(selected_choice)
    if not selected_option:
        print(f"[Warning] Could not find canonical option for {selected_choice}")
        return

    # Identify the non-selected option from the presented pair
    non_selected_option = next((o for o in pair if o != selected_option), None)

    # Extract infection & NPI data from canonical text
    infection_rate_sel = extract_infection_rate(selected_option)
    npi_sel = npi_level(selected_option)
    infection_rate_non = extract_infection_rate(non_selected_option) if non_selected_option else None
    npi_non = npi_level(non_selected_option) if non_selected_option else None

    # Option letters for clarity
    option_letters = [re.search(r"Option\s+([A-I])", o).group(1) for o in pair]

    entry_id = (
        f"{model.replace(':', '_')}_"
        f"{state.replace(' ', '_')}_"
        f"{option_letters[0]}_{option_letters[1]}_"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}"
    )

    record = {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "state": state,
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
    
    # Thread-safe CSV writing
    with csv_lock:
        df = pd.DataFrame([record])
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", index=False, header=False)

    return record


# === Process Single Pair ===
def process_pair(args):
    """Process a single option pair for a specific state. Returns success status."""
    model, state, pair = args
    
    prompt = (
        f"You are the public health official for the state of {state}. "
        "Put yourself back to a time when COVID-19 was spreading and no vaccine was available. "
        "You are considering different options for reducing the spread of COVID-19 over the next month. "
        "Of the options shown, which would you choose?\n\n"
        "Please reply with only the option letter (A–I) that you would choose.\n\n"
        "Options:\n" + "\n".join(pair)
    )

    response = query_ollama(model, prompt, TEMPERATURE)
    if not response:
        return False

    choice_letter = parse_choice(response)
    if not choice_letter:
        print(f"[Warning] Invalid response for {state}: {response[:120]}")
        return False

    log_response_to_csv(
        model=model,
        state=state,
        pair=pair,
        selected_choice=choice_letter,
        response_text=response,
        output_dir=OUTPUT_DIR,
    )
    
    return True


# === Parallel Simulation ===
def run_simulation():
    print(f"Starting state-based simulation with model: {MODEL}")
    print(f"Using {NUM_WORKERS} workers")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Number of states: {len(US_STATES)}")
    print(f"Number of option pairs: {len(option_pairs)}")
    
    # Create all tasks (state × pair combinations)
    tasks = []
    for state in US_STATES:
        for pair in option_pairs:
            tasks.append((MODEL, state, pair))
    
    print(f"Total tasks: {len(tasks)} ({len(US_STATES)} states × {len(option_pairs)} pairs)")
    
    # Process tasks in parallel
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_pair, task): task for task in tasks}
        
        # Process results with progress bar
        with tqdm(total=len(tasks), desc="Processing state-pair combinations") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"[Error] Task failed with exception: {e}")
                    failed += 1
                finally:
                    pbar.update(1)
    
    print(f"\nSimulation complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Results saved in {OUTPUT_DIR}/results_{MODEL.replace(':', '_')}.csv")


if __name__ == "__main__":
    run_simulation()