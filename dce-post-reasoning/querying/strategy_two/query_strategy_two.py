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


# === Setup ===
OPENROUTER_API_KEY = "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"
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
OUTPUT_DIR = "strategy_two_results"
MAX_CONCURRENT = 10
TEMPERATURE = [0.7]
NUM_REPEATS = 1  # Number of times to repeat each query config
SEED = 42  # Added seed for reproducibility

# Strategy 2: Per-option income loss embedded in scenario text; no blanket context variable.
# (Hospital capacity context is added in Strategy 3.)

SITE_URL = ""
SITE_NAME = ""

os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_lock = threading.Lock()


# === Scenarios (12 total: 3 no-closure + 9 non-essential; CC removed) ===
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

# Generate pairs in both orders — C(12,2) = 66 base pairs, 132 total
base_pairs = list(combinations(scenarios, 2))
option_pairs = []
for pair in base_pairs:
    option_pairs.append(pair)
    option_pairs.append((pair[1], pair[0]))


# === Prompt Builder for Strategy 2 ===
def build_prompt(pair):
    """Build a prompt with per-option income loss. No hospital capacity context."""
    option_1_text = pair[0]["text"]
    option_2_text = pair[1]["text"]

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

    internal_retries = 10
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
def parse_choice(response_text):
    """Parse the numerical choice (1 or 2). Returns int or None."""
    if not response_text:
        return None

    clean_response = response_text.strip()

    if clean_response in ['1', '2']:
        return int(clean_response)

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
                return choice_num

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
    elif "All non-essential businesses" in option_text:
        return "Non-essential closures"
    return "Other"


def npi_level(option_text):
    cat = categorize_closure(option_text)
    return {"No closure": 0, "Non-essential closures": 2}.get(cat, -1)


def get_existing_entries(model, output_dir):
    """Scan the result CSV for existing entries and return a set of fingerprints."""
    csv_path = os.path.join(output_dir, f"results_{model.replace('/', '_').replace(':', '_')}.csv")
    existing_fingerprints = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            required_cols = ['pair', 'temperature', 'repeat_index']
            if not df.empty and all(col in df.columns for col in required_cols):
                for _, row in df.iterrows():
                    fingerprint = (
                        str(row['pair']),
                        float(row['temperature']),
                        int(row['repeat_index'])
                    )
                    existing_fingerprints.add(fingerprint)
        except Exception as e:
            print(f"[Warning] Failed to read {csv_path}: {e}")
    return existing_fingerprints


# === CSV Logger for Strategy 2 ===
def log_response_to_csv(model, pair, selected_label, response_text, reasoning,
                        temperature, repeat_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    selected_scenario = scenarios_by_label.get(selected_label)
    if not selected_scenario:
        print(f"[Warning] Could not find scenario for label {selected_label}")
        return
    non_selected_scenario = pair[1] if pair[0]["label"] == selected_label else pair[0]

    pair_str = f"{pair[0]['label']}-{pair[1]['label']}"

    entry_id = (
        f"{model.replace('/', '_').replace(':', '_')}_"
        f"{pair[0]['label']}_{pair[1]['label']}_"
        f"temp{temperature}_rep{repeat_index}_"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )

    record = {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "temperature": temperature,
        "pair": pair_str,
        "option_1": pair[0]["text"],
        "option_2": pair[1]["text"],
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
async def process_pair_async(session, model, pair, temperature, repeat_index, semaphore, max_retries=3):
    """Two-turn post-reasoning: first get choice, then ask for reasoning."""
    async with semaphore:
        prompt = build_prompt(pair)
        messages = [{"role": "user", "content": prompt}]

        current_seed = SEED + repeat_index if SEED is not None else None

        # --- Turn 1: get choice ---
        for attempt in range(1, max_retries + 1):
            response = await query_openrouter_async(session, model, messages, temperature, seed=current_seed)

            if not response:
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return False, (model, pair, temperature, repeat_index)

            choice_num = parse_choice(response)
            if choice_num is None:
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    print(f"[Warning] Invalid response after {max_retries} attempts for {model} "
                          f"(temp={temperature}, repeat={repeat_index}): {response[:120]}")
                    return False, (model, pair, temperature, repeat_index)

            # Valid choice obtained — proceed to reasoning turn
            break
        else:
            return False, (model, pair, temperature, repeat_index)

        # --- Turn 2: get reasoning ---
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "Please explain your reasoning for this choice."})

        reasoning = ""
        for r_attempt in range(1, max_retries + 1):
            reasoning = await query_openrouter_async(session, model, messages, temperature, seed=current_seed)
            if reasoning:
                break
            await asyncio.sleep(2 ** r_attempt)

        if not reasoning:
            print(f"[Warning] Empty reasoning after {max_retries} attempts for {model} "
                  f"(temp={temperature}, repeat={repeat_index}) — skipping.")
            return False, (model, pair, temperature, repeat_index)

        selected_label = pair[choice_num - 1]["label"]
        log_response_to_csv(model, pair, selected_label, response, reasoning,
                            temperature, repeat_index, OUTPUT_DIR)
        return True, None


# === Async Simulation ===
async def run_simulation_async():
    print(f"Starting simulation with models: {', '.join(MODELS)}")
    print(f"Temperature values: {TEMPERATURE}")
    print(f"Number of repeats: {NUM_REPEATS}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Base option pairs: {len(base_pairs)}")
    print(f"Total option pairs (with reversals): {len(option_pairs)}")

    tasks_data = []
    skipped = 0
    for model in MODELS:
        existing = get_existing_entries(model, OUTPUT_DIR)
        for pair in option_pairs:
            pair_str = f"{pair[0]['label']}-{pair[1]['label']}"
            for temp in TEMPERATURE:
                for repeat in range(NUM_REPEATS):
                    fingerprint = (pair_str, float(temp), int(repeat))
                    if fingerprint in existing:
                        skipped += 1
                    else:
                        tasks_data.append((model, pair, temp, repeat))

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
            process_pair_async(session, model, pair, temp, repeat, semaphore)
            for model, pair, temp, repeat in tasks_data
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
        print(f"Failed Tasks After {max_retries} Retries ({len(failed_tasks)} total):")
        print(f"{'='*60}")
        for model, pair, temp, repeat in failed_tasks:
            print(f"  Model: {model}")
            print(f"  Pair: {pair[0]['label']}-{pair[1]['label']}")
            print(f"  Temperature: {temp}")
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
