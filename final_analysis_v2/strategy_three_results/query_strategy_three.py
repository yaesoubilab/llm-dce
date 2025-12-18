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
MODELS = ["deepseek/deepseek-chat-v3.1", "x-ai/grok-4-fast", "google/gemini-2.5-flash", "openai/gpt-5-nano"] 
MODELS = ["google/gemini-2.5-flash"]
OUTPUT_DIR = "strategy_three_results"
MAX_CONCURRENT = 2
TEMPERATURE = [0.1, 0.7]
SEED = 42  # Added seed for reproducibility

# Strategy 3: Economic context (percentage of population losing income)
ECONOMIC_CONTEXTS = [10, 20, 30]  # Percentage of population experiencing loss of income

# Optional: Add your site URL and name for rankings on openrouter.ai
SITE_URL = ""
SITE_NAME = ""

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

# === Prompt Builder for Strategy 3 ===
def build_prompt(pair, economic_context):
    """Build a prompt with numerical labels and economic context."""
    # Remove the original letter labels from the option text
    option_1_text = re.sub(r"Option\s+[A-I]\.\s*", "", pair[0])
    option_2_text = re.sub(r"Option\s+[A-I]\.\s*", "", pair[1])
    
    prompt = (
        "I am a public health official for a county in the United States. "
        "A novel viral pathogen (similar to H1N1 and SARS-CoV-2) is spreading in my jurisdiction. "
        "Among those who get infected, 5% would become so sick that they require hospitalization. "
        "Our hospital capacity is 15 beds per 100,000 population, and 50% of the hospital beds are currently occupied. "
        "We are considering closing non-essential businesses to reduce the spread of the pathogen over the next month. "
        f"This results in {economic_context}% of the population experiencing a loss of income during this period. "
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
    internal_retries = 3
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
    """Parse the numerical choice (1 or 2) and map back to original letter."""
    if not response_text:
        return None
    
    clean_response = response_text.strip()
    
    if clean_response in ['1', '2']:
        choice_num = int(clean_response)
        selected_option = pair[choice_num - 1]
        return re.search(r"Option\s+([A-I])", selected_option).group(1)
    
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

# === CSV Logger for Strategy 3 ===
def log_response_to_csv(model, pair, selected_choice, response_text, reasoning, 
                       temperature, economic_context, output_dir):
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
        f"econ{economic_context}_temp{temperature}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )

    record = {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "temperature": temperature,
        "economic_context": economic_context,
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
    }

    csv_path = os.path.join(output_dir, f"results_{model.replace('/', '_').replace(':', '_')}.csv")
    with csv_lock:
        df = pd.DataFrame([record])
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", index=False, header=False)

# === Async Process Single Pair with Retry Logic ===
async def process_pair_async(session, model, pair, temperature, economic_context, semaphore, max_retries=3):
    """Process with semaphore to control concurrency and retry on failure."""
    async with semaphore:
        prompt = build_prompt(pair, economic_context)
        messages = [{"role": "user", "content": prompt}]
        
        # Retry loop for answer
        for attempt in range(1, max_retries + 1):
            response = await query_openrouter_async(session, model, messages, temperature, seed=SEED)
            
            if not response:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return False, (model, pair, temperature, economic_context)
            
            choice_letter = parse_choice(response, pair)
            if not choice_letter:
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    print(f"[Warning] Invalid response after {max_retries} attempts for {model} (temp={temperature}, econ={economic_context}%): {response[:120]}")
                    return False, (model, pair, temperature, economic_context)

            # Success with answer - now ask for reasoning
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Please explain your reasoning for this choice."})
            
            reasoning = ""
            # Simple retry for reasoning
            for r_attempt in range(1, max_retries + 1):
                reasoning = await query_openrouter_async(session, model, messages, temperature, seed=SEED)
                if reasoning:
                    break
                await asyncio.sleep(2 ** r_attempt)
            
            # Success - log and return
            log_response_to_csv(model, pair, choice_letter, response, reasoning, 
                              temperature, economic_context, OUTPUT_DIR)
            return True, None
        
        return False, (model, pair, temperature, economic_context)

# === Async Simulation ===
async def run_simulation_async():
    print(f"Starting simulation with models: {', '.join(MODELS)}")
    print(f"Temperature values: {TEMPERATURE}")
    print(f"Economic contexts: {ECONOMIC_CONTEXTS}%")
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    print(f"Seed: {SEED}")
    print(f"Base option pairs: {len(base_option_pairs)}")
    print(f"Total option pairs (with reversals): {len(option_pairs)}")

    # Generate all combinations: model × pair × temperature × economic_context
    tasks_data = [
        (model, pair, temp, econ) 
        for model in MODELS 
        for pair in option_pairs 
        for temp in TEMPERATURE 
        for econ in ECONOMIC_CONTEXTS
    ]
    print(f"Total tasks: {len(tasks_data)}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    successful = 0
    failed = 0
    failed_tasks = []
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_pair_async(session, model, pair, temp, econ, semaphore)
            for model, pair, temp, econ in tasks_data
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
        for model, pair, temp, econ in failed_tasks:
            option_letters = [re.search(r"Option\s+([A-I])", o).group(1) for o in pair]
            print(f"  Model: {model}")
            print(f"  Pair: {option_letters[0]}-{option_letters[1]}")
            print(f"  Temperature: {temp}")
            print(f"  Economic Context: {econ}%")
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
