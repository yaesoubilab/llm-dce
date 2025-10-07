# %%
from openai import OpenAI
from itertools import combinations
import json
from collections import defaultdict
from tqdm.auto import tqdm
import random
import requests
import re

import pandas as pd
import matplotlib.pyplot as plt

# %%
# setup local ollama
OLLAMA_URL = "http://localhost:11434/api/chat"

def query_ollama(model, prompt, temperature):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    response = requests.post(OLLAMA_URL, json=data)
    raw_text = response.text.strip()
    first_line = raw_text.splitlines()[0]
    result = json.loads(first_line)
    return result["message"]["content"].strip()

# %%
# all 9 options
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

# get all pairwise combinations (72 total)
option_pairs = list(combinations(options, 2))

# roles to test
roles = [
    "public health official",
    "mother",
    "father",
    "student",
    "male",
    "female",
    "senior citizen (>65 years old)",
    "young adult (18-25 years old)",
    "essential worker (e.g., grocery store worker, nurse, etc.)",
    "non-essential worker (e.g., office worker, teacher, etc.)",
    "small business owner",
    "black american",
    "white american",
    "democrat voter",
    "republican voter",
    "independent voter",
]

# %%
# num of full reruns
num_iterations = 10

# structured counter: role -> (pair -> {option: count})
selection_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# store full convo logs
conversation_log = []

def extract_valid_choice(text):
    # Regex pattern to match option starting with a letter A-I and optionally the word "Option"
    pattern = r"^(Option\s)?([A-I])\b.*"
    match = re.match(pattern, text.strip(), re.IGNORECASE)
    if match:
        # Return full matched line starting with valid letter option (like "Option A. ...", or "A. ...")
        return text.strip()
    else:
        # If doesn't match, return None or empty to ignore
        return None

# %%
# iterate over reruns
for run in tqdm(range(num_iterations), desc="Reruns", position=0):
    # vary temperature each run
    temperature = random.uniform(0.3, 1.0)

    # iterate over roles
    for role in tqdm(roles, desc=f"Iteration {run+1}", leave=True, position=1):
        # iterate over all pairs
        for pair in tqdm(option_pairs, desc=f"Iteration {run+1} ({role})", leave=False, position=2):
            prompt_text = (
                f"As a {role}, put yourself back to a time when COVID-19 was spreading in your county and no vaccine was available. "
                "Your county’s public health officials are considering different options for reducing the spread of COVID-19 over the next month. "
                "Of the options shown, which of the following would you choose? Only state the chosen option letter and no reasoning.\n\n"
                + "\n".join(pair)
            )

            model_choice = query_ollama("llama3", prompt_text, temperature)
            model_choice = model_choice.split("\n")[0].strip()  # take only first line

            valid_choice = extract_valid_choice(model_choice)

            if valid_choice is None:
                # Handle invalid response: maybe skip or log warning
                print(f"Invalid model choice ignored: {model_choice}")
            else:
                # Use valid_choice for counting
                selection_counter[role][pair][valid_choice] += 1

            # append convo
            conversation_log.append({
                "iteration": run + 1,
                "role": role,
                "temperature": temperature,
                "options_presented": pair,
                "model_choice": valid_choice if valid_choice else model_choice,
            })

# %%
# write full log to JSON
with open("llm_covid_responses_roles.json", "w") as f:
    json.dump(conversation_log, f, indent=2)

# save structured summary
summary = {role: {str(pair): counts for pair, counts in pairs.items()}
           for role, pairs in selection_counter.items()}

with open("llm_covid_selection_summary_roles.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Completed all reruns. Summary written to JSON files.")

# %%



