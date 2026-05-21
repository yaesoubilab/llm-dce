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
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"  # Full endpoint URL

ALL_MODELS = ["google/gemini-2.5-flash", "google/gemini-2.5-pro", "google/gemini-2.5-flash-lite",
              "x-ai/grok-4-fast", "x-ai/grok-4",
              "deepseek/deepseek-chat-v3.1", 
              "openai/gpt-5-nano", "openai/gpt-5", "openai/gpt-oss-120b",
              "anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5",
              "qwen/qwen3-max", 
              "mistralai/mistral-large-2411", 
              "meta-llama/llama-3.1-405b", 
              "z-ai/glm-4.6", 
              "moonshotai/kimi-k2-thinking"]
MODELS = ["deepseek/deepseek-chat-v3.1", "openai/gpt-5-nano", "google/gemini-2.5-flash", "x-ai/grok-4-fast"]
OUTPUT_DIR = "strategy_one_results"
MAX_CONCURRENT = 10  # Control concurrent requests to respect rate limits
TEMPERATURE = [0.7]  # List of temperatures to test
NUM_REPEATS = 1  # Number of times to repeat each query
SEED = 42  # Seed for reproducibility

# Optional: Add your site URL and name for rankings on openrouter.ai
SITE_URL = ""  # Optional: your site URL
SITE_NAME = ""  # Optional: your site name

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
        "Please structure your response using the following format:\n\n"
        "REASONING: [explain your reasoning for this choice]\n"
        "RESPONSE: [the number (1 or 2) of the option you would choose]"
    )

    return prompt

# === Async OpenRouter Query ===
async def query_openrouter_async(session, model, messages, temperature=0.1, seed=None):
    """Async query to OpenRouter API with seed support and internal retries."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Add optional headers for rankings on openrouter.ai
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
    internal_retries = 3
    base_delay = 2
    
    for attempt in range(internal_retries):
        try:
            async with session.post(
                OPENROUTER_URL,  # Use the full URL with /chat/completions
                headers=headers, 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                # Get response text first for debugging
                response_text = await response.text()
                
                # Check if response is successful
                if response.status != 200:
                    print(f"[Error] HTTP {response.status}: {response_text[:200]}")
                    # If it's a rate limit (429) or server error (5xx), we might want to retry
                    if response.status == 429 or response.status >= 500:
                         await asyncio.sleep(base_delay * (2 ** attempt))
                         continue
                    return ""
                
                # Try to parse as JSON
                try:
                    result = json.loads(response_text)
                    return result["choices"][0]["message"]["content"].strip()
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"[Error] Failed to parse response: {e}")
                    print(f"[Debug] Response text: {response_text[:200]}")
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
def parse_concurrent_response(response_text):
    """Parse REASONING/RESPONSE structured output. Returns (reasoning, choice_num or None).

    Extraction order:
      1. Primary:  REASONING: ... RESPONSE: <1|2>  (structured headers present)
      2. Fallback: keyword patterns searched over the last 3 lines only
                   (avoids false-positive matches on numbers inside the reasoning body)
      3. Last resort: bare digit search over full text — logs a warning when used
    """
    if not response_text:
        return "", None

    # Primary: extract structured REASONING/RESPONSE sections
    # Patterns tolerate optional markdown bold markers (**LABEL:** format used by some models)
    reasoning_match = re.search(
        r'\*{0,2}\s*REASONING\s*:\s*\*{0,2}\s*(.*?)(?=\n*\s*\*{0,2}\s*RESPONSE\s*:\s*\*{0,2})',
        response_text, re.DOTALL | re.IGNORECASE
    )
    response_match = re.search(
        r'\*{0,2}\s*RESPONSE\s*:\s*\*{0,2}\s*([12])',
        response_text, re.IGNORECASE
    )

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    if not reasoning:
        print(f"[Warning] REASONING: section missing or empty — model may not have followed the format.")

    if response_match:
        return reasoning, int(response_match.group(1))

    # Fallback: RESPONSE: label absent — will retry via outer loop
    print(f"[Warning] RESPONSE: label missing — will retry.")
    return reasoning, None

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

def get_existing_entries(model, output_dir):
    """Scan the result CSV for existing entries and return a set of fingerprints."""
    csv_path = os.path.join(output_dir, f"results_{model.replace('/', '_').replace(':', '_')}.csv")
    existing_fingerprints = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Use pair string, temperature, and repeat_index to uniquely identify a query
            if not df.empty and all(col in df.columns for col in ['pair', 'temperature', 'repeat_index']):
                for _, row in df.iterrows():
                    fingerprint = (str(row['pair']), float(row['temperature']), int(row['repeat_index']))
                    existing_fingerprints.add(fingerprint)
        except Exception as e:
            print(f"[Warning] Failed to read {csv_path}: {e}")
    return existing_fingerprints

# === CSV Logger ===
def log_response_to_csv(model, pair, selected_choice, response_text, reasoning, temperature, repeat_index, output_dir):
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
        f"{model.replace('/', '_').replace(':', '_')}_{option_letters[0]}_{option_letters[1]}_"
        f"temp{temperature}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )

    record = {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "temperature": temperature,
        "pair": f"{option_letters[0]}-{option_letters[1]}",
        "option_1": pair[0],
        "option_2": pair[1],
        "selected_choice": selected_choice,
        "response_text": response_text,
        "reasoning": reasoning,
        "infection_rate_selected": infection_rate_sel,
        "npi_level_selected": npi_sel,
        "infection_rate_non_selected": infection_rate_non,
        "npi_level_non_selected": npi_non,
        "repeat_index": repeat_index
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
    """Single-turn concurrent reasoning: one call returns REASONING + RESPONSE together."""
    async with semaphore:
        prompt = build_prompt(pair)
        messages = [{"role": "user", "content": prompt}]

        current_seed = SEED + repeat_index if SEED is not None else None

        for attempt in range(1, max_retries + 1):
            response = await query_openrouter_async(session, model, messages, temperature, seed=current_seed)

            if not response:
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    return False, (model, pair, temperature, repeat_index)

            reasoning, choice_num = parse_concurrent_response(response)
            if choice_num is None:
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    print(f"[Warning] Invalid response after {max_retries} attempts for {model} (temp={temperature}, repeat={repeat_index}): {response[:120]}")
                    return False, (model, pair, temperature, repeat_index)

            selected_option = pair[choice_num - 1]
            choice_letter = re.search(r"Option\s+([A-I])", selected_option).group(1)
            log_response_to_csv(model, pair, choice_letter, response, reasoning, temperature, repeat_index, OUTPUT_DIR)
            return True, None

        return False, (model, pair, temperature, repeat_index)

# === Async Simulation ===
async def run_simulation_async():
    print(f"Starting simulation with models: {', '.join(MODELS)}")
    print(f"Temperature values: {TEMPERATURE}")
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    print(f"Number of repeats: {NUM_REPEATS}")
    print(f"Base option pairs: {len(base_option_pairs)}")
    print(f"Total option pairs (with reversals): {len(option_pairs)}")

    # Load existing entries to avoid duplicates
    tasks_data = []
    skipped = 0
    for model in MODELS:
        existing = get_existing_entries(model, OUTPUT_DIR)
        for pair in option_pairs:
            option_letters = [re.search(r"Option\s+([A-I])", o).group(1) for o in pair]
            pair_str = f"{option_letters[0]}-{option_letters[1]}"
            for temp in TEMPERATURE:
                for r in range(NUM_REPEATS):
                    fingerprint = (pair_str, float(temp), int(r))
                    if fingerprint in existing:
                        skipped += 1
                    else:
                        tasks_data.append((model, pair, temp, r))
    
    print(f"Total tasks planned: {len(tasks_data) + skipped}")
    print(f"Already completed: {skipped}")
    print(f"Tasks to run: {len(tasks_data)}")

    # Control concurrency with semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Create SSL context using certifi certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    successful = 0
    failed = 0
    failed_tasks = []  # Track failed tasks
    
    # Create connector with SSL context
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_pair_async(session, model, pair, temp, r, semaphore)
            for model, pair, temp, r in tasks_data
        ]
        
        # Process with progress bar using tqdm.asyncio.as_completed
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
    
    # Print failed tasks
    if failed_tasks:
        print(f"\n{'='*60}")
        print(f"Failed Tasks After 3 Retries ({len(failed_tasks)} total):")
        print(f"{'='*60}")
        for model, pair, temp, r in failed_tasks:
            option_letters = [re.search(r"Option\s+([A-I])", o).group(1) for o in pair]
            print(f"  Model: {model}")
            print(f"  Pair: {option_letters[0]}-{option_letters[1]}")
            print(f"  Temperature: {temp}")
            print(f"  Repeat: {r}")
            print(f"  {'-'*58}")
    
    print(f"\nResults saved in {OUTPUT_DIR}/")
    for model in MODELS:
        print(f"  - results_{model.replace('/', '_').replace(':', '_')}.csv")

if __name__ == "__main__":
    asyncio.run(run_simulation_async())
