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
MODELS = ["mistral:latest", "llama3:8b", "phi3:mini-4k", 
          "deepseek-llm:7b", "llama3.2:3b", "phi4-mini:latest",
          "gemma3:1b", "gpt-oss:20b", "gemma3:4b"]
MODELS = ['gpt-oss:20b', 'deepseek-llm:7b', 'phi3:mini-4k', 'llama3:8b']  # For testing
OUTPUT_DIR = "strategy_one_numerical_results_repeats"
NUM_WORKERS = 1
TEMPERATURE = 0.1
NUM_REPEATS = 3  # NEW: Number of times to repeat the entire question set




os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_lock = threading.Lock()





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




# Generate pairs in both orders
base_option_pairs = list(combinations(options, 2))
option_pairs = []
for pair in base_option_pairs:
    option_pairs.append(pair)  # Original order (A, B)
    option_pairs.append((pair[1], pair[0]))  # Reversed order (B, A)




canonical_option_map = {re.search(r"Option\s+([A-I])", o).group(1): o for o in options}





# === Prompt Builder ===
def build_prompt(pair):
    """Build a prompt with numerical labels (Option 1 and Option 2)."""

    # Remove the original letter labels from the option text
    option_1_text = re.sub(r"Option\s+[A-I]\.\s*", "", pair[0])
    option_2_text = re.sub(r"Option\s+[A-I]\.\s*", "", pair[1])

    prompt = (
        "I am a public health official for a county in the United States. "
        "A novel viral pathogen (similar to H1N1 and SARS-CoV-2) is spreading in my jurisdiction. "
        "Among those who get infected, 5% would become so sick that they require hospitalization. "
        "We are considering closing non-essential businesses to reduce the spread of the pathogen over the next month. "
        "Our goal is to balance the public health benefits and the economic costs on local businesses and workers. "
        "Given this objective, which of the following two options would you choose?\n\n"
        f"Option 1: {option_1_text}\n"
        f"Option 2: {option_2_text}\n\n"
        "Please reply with only the number (1 or 2) of the option you would choose. Do not include any additional text."
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
def parse_choice(response_text, pair):
    """Parse the numerical choice (1 or 2) and map back to original letter."""
    if not response_text:
        return None

    clean_response = response_text.strip()

    # Direct single digit
    if clean_response in ['1', '2']:
        choice_num = int(clean_response)
        selected_option = pair[choice_num - 1]
        return re.search(r"Option\s+([A-I])", selected_option).group(1)

    # Various patterns to extract 1 or 2
    patterns = [
        r'^\s*([12])\s*[.,;:!]?\s*$',  # Just "1" or "2" with optional punctuation
        r'Option\s+([12])\b',  # "Option 1" or "Option 2"
        r'(?:choose|select|prefer|pick)\s+(?:Option\s+)?([12])\b',
        r'(?:My choice is|choice is|choice:)\s+(?:Option\s+)?([12])\b',
        r'\b([12])\b',  # Any "1" or "2" as word boundary
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            choice_num = int(match.group(1))
            if choice_num in [1, 2]:
                selected_option = pair[choice_num - 1]
                return re.search(r"Option\s+([A-I])", selected_option).group(1)

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
def log_response_to_csv(model, pair, selected_choice, response_text, repeat_num, output_dir):
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
        f"{model.replace(':', '_')}_{option_letters[0]}_{option_letters[1]}_"
        f"rep{repeat_num}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )




    record = {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "repeat_number": repeat_num,  # NEW: Track which repeat this is
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
    model, pair, repeat_num = args  # NEW: Added repeat_num

    prompt = build_prompt(pair)
    response = query_ollama(model, prompt, TEMPERATURE)

    if not response:
        return False

    choice_letter = parse_choice(response, pair)
    if not choice_letter:
        print(f"[Warning] Invalid response for {model} (repeat {repeat_num}): {response[:120]}")
        return False




    log_response_to_csv(model, pair, choice_letter, response, repeat_num, OUTPUT_DIR)
    return True





# === Parallel Simulation ===
def run_simulation():
    print(f"Starting simulation with models: {', '.join(MODELS)}")
    print(f"Using {NUM_WORKERS} workers")
    print(f"Number of repeats: {NUM_REPEATS}")
    print(f"Base option pairs: {len(base_option_pairs)}")
    print(f"Total option pairs (with reversals): {len(option_pairs)}")




    # NEW: Generate tasks with repeat numbers
    tasks = [
        (model, pair, repeat_num) 
        for repeat_num in range(1, NUM_REPEATS + 1)
        for model in MODELS 
        for pair in option_pairs
    ]
    print(f"Total tasks (including repeats): {len(tasks)}")
    print(f"Tasks per model: {len(option_pairs) * NUM_REPEATS}")




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
    print(f"Success rate: {successful / len(tasks) * 100:.2f}%")
    print(f"Results saved in {OUTPUT_DIR}/")
    for model in MODELS:
        print(f"  - results_{model.replace(':', '_')}.csv")





if __name__ == "__main__":
    run_simulation()