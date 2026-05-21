import os
import re
import json
import uuid
import asyncio
import aiohttp
import ssl
import certifi
import pandas as pd
from itertools import combinations
from datetime import datetime
from tqdm.asyncio import tqdm
import threading
from dotenv import load_dotenv

load_dotenv()

# === Setup ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# Validate API key is present
if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable is not set. "
        "Please set it using: os.environ['OPENROUTER_API_KEY'] = 'your-key-here'"
    )


ALL_MODELS = ["x-ai/grok-4-fast", "google/gemini-2.5-flash", "deepseek/deepseek-chat-v3.1",
              "openai/gpt-5-nano", "anthropic/claude-sonnet-4.5", "openai/gpt-5"]
MODELS = ["x-ai/grok-4-fast", "google/gemini-2.5-flash", "deepseek/deepseek-chat-v3.1", "openai/gpt-5-nano"]
#MODELS = ["google/gemini-2.5-flash"]
OUTPUT_DIR = "strategy_three_results"
MAX_CONCURRENT = 30
TEMPERATURE = [0.7]
NUM_REPEATS = 1  # Number of times to repeat each query config
SEED = 42  # Added seed for reproducibility


# Strategy 3: Hospital capacity context (income loss is now a per-option attribute)
CAPACITY_CONTEXTS = [20, 50, 80]  # Percentage of hospital beds currently occupied


# Optional: Add your site URL and name for rankings on openrouter.ai
SITE_URL = ""
SITE_NAME = ""


os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_lock = threading.Lock()


# === Scenarios (21 total: 3 no-closure + 9 crowded-closure + 9 non-essential) ===
INFECTION_RATES = [1, 10, 20]
INCOME_LOSS_LEVELS = [10, 20, 30]


def _build_scenarios():
    scenarios = []
    for x in INFECTION_RATES:
        scenarios.append({
            "label": f"NC_0_{x}",
            "closure": 0,
            "income_loss": 0,
            "infection": x,
            "text": (
                f"No businesses are closed, resulting in 0% of the population "
                f"experiencing income loss; {x} out of 100 people will be infected "
                f"in the next month."
            ),
        })
    for y in INCOME_LOSS_LEVELS:
        for x in INFECTION_RATES:
            scenarios.append({
                "label": f"CC_{y}_{x}",
                "closure": 1,
                "income_loss": y,
                "infection": x,
                "text": (
                    f"Crowded, high-risk businesses (such as clubs, theaters, and "
                    f"large sports venues) are closed, resulting in {y}% of the "
                    f"population experiencing income loss; {x} out of 100 people "
                    f"will be infected in the next month."
                ),
            })
    for y in INCOME_LOSS_LEVELS:
        for x in INFECTION_RATES:
            scenarios.append({
                "label": f"NE_{y}_{x}",
                "closure": 2,
                "income_loss": y,
                "infection": x,
                "text": (
                    f"All non-essential businesses (such as clubs, theaters, "
                    f"restaurants, bars, and gyms) are closed, resulting in {y}% "
                    f"of the population experiencing income loss; {x} out of 100 "
                    f"people will be infected in the next month."
                ),
            })
    return scenarios


scenarios = _build_scenarios()
scenarios_by_label = {s["label"]: s for s in scenarios}


# Generate pairs in both orders — C(21,2) = 210 base pairs, 420 total
base_pairs = list(combinations(scenarios, 2))
option_pairs = []
for pair in base_pairs:
    option_pairs.append(pair)                  # (s1, s2)
    option_pairs.append((pair[1], pair[0]))    # (s2, s1)
# Total: 420 pairs


# === Prompt Builder for Strategy 3 ===
def build_prompt(pair, capacity_context):
    """Build a prompt with numerical labels and hospital capacity context.
    Income loss is embedded in each scenario's text field.
    """
    option_1_text = pair[0]["text"]
    option_2_text = pair[1]["text"]

    prompt = (
        "I am a public health official for a county in the United States. "
        "A novel viral pathogen (similar to H1N1 and SARS-CoV-2) is spreading in my jurisdiction. "
        "Among those who get infected, 5% would become so sick that they require hospitalization. "
        f"Our hospital capacity is 15 beds per 100,000 population, and {capacity_context}% of the hospital beds are currently occupied. "
        "We are considering closing non-essential businesses to reduce the spread of the pathogen over the next month. "
        "Our goal is to balance the public health benefits and the economic costs on local businesses and workers. "
        "Given this objective, which of the following two options would you choose?\n\n"
        f"Option 1: {option_1_text}\n"
        f"Option 2: {option_2_text}\n\n"
        "Please reply with only the number (1 or 2) of the option you would choose. Do not include any additional text."
    )

    return prompt


# === Async OpenRouter Query ===
async def query_openrouter_async(session, model, messages, temperature=0.1, seed=None):
    """Async query to OpenRouter API with seed support and internal retries."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    if SITE_URL:
        headers["HTTP-Referer"] = SITE_URL
    if SITE_NAME:
        headers["X-Title"] = SITE_NAME

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    if seed is not None:
        payload["seed"] = seed

    # Internal retry configuration
    internal_retries = 10 # RETRY NUM
    base_delay = 2

    for attempt in range(internal_retries):
        try:
            async with session.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response_text = await response.text()

                if response.status != 200:
                    print(f"[Error] HTTP {response.status}: {response_text[:200]}")
                    # If it's a rate limit (429) or server error (5xx), we might want to retry
                    if response.status == 429 or response.status >= 500:
                        await asyncio.sleep(base_delay * (2 ** attempt))
                        continue
                    return ""

                try:
                    result = json.loads(response_text)
                    return result["choices"][0]["message"]["content"].strip()
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[Error] Failed to parse response: {e}")
                    return ""

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"[Warning] Request failed (attempt {attempt+1}/{internal_retries}): {e}")
            if attempt < internal_retries - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))
            else:
                print(f"[Error] All {internal_retries} attempts failed for model {model}")
                return ""
    return ""


# === Parsing ===
def parse_choice(response_text, pair):
    """Parse the numerical choice (1 or 2) and map back to scenario label."""
    if not response_text:
        return None

    clean_response = response_text.strip()

    if clean_response in ['1', '2']:
        choice_num = int(clean_response)
        return pair[choice_num - 1]["label"]

    patterns = [
        r'^\s*([12])\s*[.,;:!]?\s*$',
        r'Option\s+([12])\b',
        r'(?:choose|select|prefer|pick)\s+(?:Option\s+)?([12])\b',
        r'(?:My choice is|choice is|choice:)\s+(?:Option\s+)?([12])\b',
        r'\b([12])\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            choice_num = int(match.group(1))
            if choice_num in [1, 2]:
                return pair[choice_num - 1]["label"]

    return None


# === Infection/NPI/income helpers ===
def extract_infection_rate(option_text):
    m = re.search(r"(\d+)\s+out of 100 people will be infected", option_text)
    return int(m.group(1)) if m else None


def extract_income_loss(option_text):
    m = re.search(r"(\d+)%\s+of the population experiencing income loss", option_text)
    return int(m.group(1)) if m else None


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


def get_existing_entries(model, output_dir):
    """Scan the result CSV for existing entries and return a set of fingerprints."""
    csv_path = os.path.join(output_dir, f"results_{model.replace('/', '_').replace(':', '_')}.csv")
    existing_fingerprints = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Use pair string, temperature, capacity_context, and repeat_index to uniquely identify a query
            required_cols = ['pair', 'temperature', 'capacity_context', 'repeat_index']
            if not df.empty and all(col in df.columns for col in required_cols):
                for _, row in df.iterrows():
                    fingerprint = (
                        str(row['pair']),
                        float(row['temperature']),
                        int(row['capacity_context']),
                        int(row['repeat_index'])
                    )
                    existing_fingerprints.add(fingerprint)
        except Exception as e:
            print(f"[Warning] Failed to read {csv_path}: {e}")
    return existing_fingerprints


# === CSV Logger for Strategy 3 ===
def log_response_to_csv(model, pair, selected_label, response_text, reasoning,
                        temperature, capacity_context, repeat_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    selected_scenario = scenarios_by_label.get(selected_label)
    if not selected_scenario:
        print(f"[Warning] Could not find scenario for label {selected_label}")
        return
    non_selected_scenario = pair[1] if pair[0]["label"] == selected_label else pair[0]

    # Full option text for the row
    option_1 = pair[0]["text"]
    option_2 = pair[1]["text"]
    pair_str = f"{pair[0]['label']}-{pair[1]['label']}"

    entry_id = (
        f"{model.replace('/', '_').replace(':', '_')}_"
        f"{pair[0]['label']}_{pair[1]['label']}_"
        f"cap{capacity_context}_temp{temperature}_rep{repeat_index}_"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )

    record = {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "temperature": temperature,
        "capacity_context": capacity_context,
        "pair": pair_str,
        "option_1": option_1,
        "option_2": option_2,
        "selected_choice": selected_label,
        "response_text": response_text,
        "reasoning": reasoning,
        "infection_rate_selected": selected_scenario["infection"],
        "npi_level_selected": selected_scenario["closure"],
        "infection_rate_non_selected": non_selected_scenario["infection"],
        "npi_level_non_selected": non_selected_scenario["closure"],
        "income_loss_selected": selected_scenario["income_loss"],
        "income_loss_non_selected": non_selected_scenario["income_loss"],
        "repeat_index": repeat_index,
    }

    csv_path = os.path.join(output_dir, f"results_{model.replace('/', '_').replace(':', '_')}.csv")
    with csv_lock:
        df = pd.DataFrame([record])
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", index=False, header=False)


# === Async Process Single Pair with Retry Logic ===
async def process_pair_async(session, model, pair, temperature, capacity_context, repeat_index, semaphore, max_retries=3):
    """Process with semaphore to control concurrency and retry on failure."""
    async with semaphore:
        prompt = build_prompt(pair, capacity_context)
        messages = [{"role": "user", "content": prompt}]

        # Use a unique seed for each repeat if SEED is provided
        current_seed = SEED + repeat_index if SEED is not None else None

        # Retry loop for answer
        for attempt in range(1, max_retries + 1):
            response = await query_openrouter_async(session, model, messages, temperature, seed=current_seed)

            if not response:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return False, (model, pair, temperature, capacity_context, repeat_index)

            selected_label = parse_choice(response, pair)
            if not selected_label:
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    print(f"[Warning] Invalid response after {max_retries} attempts for {model} (temp={temperature}, cap={capacity_context}%, repeat={repeat_index}): {response[:120]}")
                    return False, (model, pair, temperature, capacity_context, repeat_index)

            # Success with answer - now ask for reasoning
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Please explain your reasoning for this choice."})

            reasoning = ""
            # Simple retry for reasoning
            for r_attempt in range(1, max_retries + 1):
                reasoning = await query_openrouter_async(session, model, messages, temperature, seed=current_seed)
                if reasoning:
                    break
                await asyncio.sleep(2 ** r_attempt)

            if not reasoning:
                print(f"[Warning] Empty reasoning after {max_retries} attempts for {model} "
                      f"(temp={temperature}, cap={capacity_context}%, repeat={repeat_index}) — skipping.")
                return False, (model, pair, temperature, capacity_context, repeat_index)

            # Success - log and return
            log_response_to_csv(model, pair, selected_label, response, reasoning,
                                temperature, capacity_context, repeat_index, OUTPUT_DIR)
            return True, None

        return False, (model, pair, temperature, capacity_context, repeat_index)


# === Async Simulation ===
async def run_simulation_async():
    print(f"Starting simulation with models: {', '.join(MODELS)}")
    print(f"Temperature values: {TEMPERATURE}")
    print(f"Capacity contexts: {CAPACITY_CONTEXTS}%")
    print(f"Number of repeats: {NUM_REPEATS}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Base option pairs: {len(base_pairs)}")
    print(f"Total option pairs (with reversals): {len(option_pairs)}")

    # Load existing entries to avoid duplicates
    tasks_data = []
    skipped = 0
    for model in MODELS:
        existing = get_existing_entries(model, OUTPUT_DIR)
        for pair in option_pairs:
            pair_str = f"{pair[0]['label']}-{pair[1]['label']}"
            for temp in TEMPERATURE:
                for cap in CAPACITY_CONTEXTS:
                    for repeat in range(NUM_REPEATS):
                        fingerprint = (pair_str, float(temp), int(cap), int(repeat))
                        if fingerprint in existing:
                            skipped += 1
                        else:
                            tasks_data.append((model, pair, temp, cap, repeat))

    print(f"Total tasks planned: {len(tasks_data) + skipped}")
    print(f"Already completed: {skipped}")
    print(f"Tasks to run: {len(tasks_data)}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    successful = 0
    failed = 0
    failed_tasks = []

    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_pair_async(session, model, pair, temp, cap, repeat, semaphore)
            for model, pair, temp, cap, repeat in tasks_data
        ]

        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
            try:
                success, failed_info = await coro
                if success:
                    successful += 1
                else:
                    failed += 1
                    if failed_info:
                        failed_tasks.append(failed_info)

            except Exception as e:
                print(f"[Error] Task failed with exception: {e}")
                failed += 1

    print(f"\nSimulation complete!")
    print(f"Successful: {successful}, Failed: {failed}")

    if failed_tasks:
        print(f"\n{'='*60}")
        print(f"Failed Tasks After 3 Retries ({len(failed_tasks)} total):")
        print(f"{'='*60}")
        for model, pair, temp, cap, repeat in failed_tasks:
            print(f"  Model: {model}")
            print(f"  Pair: {pair[0]['label']}-{pair[1]['label']}")
            print(f"  Temperature: {temp}")
            print(f"  Capacity: {cap}%")
            print(f"  Repeat: {repeat}")
            print(f"  {'-'*58}")

    print(f"\nResults saved in {OUTPUT_DIR}/")
    for model in MODELS:
        print(f"  - results_{model.replace('/', '_').replace(':', '_')}.csv")


# === Main Entry Point ===
if __name__ == "__main__":
    try:
        get_ipython()
        import nest_asyncio
        nest_asyncio.apply()
    except (NameError, ImportError):
        pass

    asyncio.run(run_simulation_async())
