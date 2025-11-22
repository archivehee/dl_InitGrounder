# ==================================
# 0. Import & Configuration
# ==================================
import os
import json
import random
from collections import defaultdict
from tqdm import tqdm
import requests
import re

# Paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
AMAZON_ROOT = os.path.join(PROJECT_ROOT, "data", "amazon")
INTER_ROOT = os.path.join(AMAZON_ROOT, "inter_cdr")

DOMAINS = ["Electronics", "Home_and_Kitchen"]

META_JSON = {
    "Electronics": os.path.join(AMAZON_ROOT, "f_meta_Electronics.json"),
    "Home_and_Kitchen": os.path.join(AMAZON_ROOT, "f_meta_Home_and_Kitchen.json"),
}

# Ollama server settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:8b"

# Output file paths for each domain
OUTPUT_JSON = {
    domain: os.path.join(AMAZON_ROOT, f"usr_prof_{domain}.json")
    for domain in DOMAINS
}

# Generation settings
NUM_USERS = 3000
MAX_LEN_PER_DOMAIN = 5
MAX_META_CHARS = 300


# ==================================
# 1. Prompt Template
# ==================================
BASE_PROMPT = """<user profile sentence generation>

You are an assistant that generates a natural language profile for a user based on their interaction history with items. Your goal is to infer what kinds of items this user typically prefers and describe that in 2 coherent and expressive sentences.

1. Input format (JSON, one line):
{
  "current_user": <User ID>,
  "history": {
    "<item_id1>": "<Item_title || Item_description>",
    "<item_id2>": "...",
  }
}

2. Output format (single-line JSON):
{
  "User_id": "<User ID>",
  "profile": "<descriptive sentence>"
}

Important Notes:
- Only return the JSON object.
- The profile must be natural-language sentences.
"""


# ==================================
# JSON Extraction Utility for DeepSeek-style output
# ==================================
def extract_json_block(text: str):
    """
    Extracts the JSON block from DeepSeek-R1 style output
    of the form <think>...</think>{...}.
    """
    # Prefer content after </think>
    if "</think>" in text:
        suffix = text.split("</think>", 1)[1]
        m = re.search(r"\{.*\}", suffix, re.DOTALL)
        if m:
            candidate = m.group()
            try:
                return json.loads(candidate)
            except Exception:
                pass

    # Fallback: search the entire text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidate = m.group()
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


# ==================================
# 2. Utility Functions
# ==================================
def load_id_map(path):
    idx2raw = {}
    with open(path, "r") as f:
        for line in f:
            raw, idx = line.strip().split()
            idx2raw[int(idx)] = raw
    return idx2raw


def load_train(train_path):
    user_items = {}
    with open(train_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u = int(parts[0])
                items = list(map(int, parts[1:]))
                user_items[u] = items
    return user_items


def load_item_text(meta_path):
    """
    Converts meta.json list entries into a mapping:
        asin → "title || description"
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if not isinstance(meta, list):
        raise ValueError(f"Expected list meta format, got {type(meta)} in {meta_path}")

    asin2text = {}

    for entry in meta:
        if not isinstance(entry, dict):
            continue

        asin = entry.get("parent_asin") or entry.get("asin")
        if asin is None:
            continue

        title = entry.get("title", "")
        desc_field = entry.get("description", "")

        if isinstance(desc_field, list):
            desc = " ".join(str(x) for x in desc_field if x)
        else:
            desc = str(desc_field) if desc_field else ""

        pieces = [p for p in [title, desc] if p]
        if not pieces:
            continue

        text = " || ".join(pieces)
        asin2text[asin] = text[:MAX_META_CHARS]

    return asin2text


# ==================================
# 3. Ollama /api/generate Call
# ==================================
def call_ollama(model, host, base_prompt, user_input):
    """
    Sends a single prompt to Ollama and extracts only the JSON output.
    """
    full_prompt = base_prompt + "\n\nInput JSON:\n" + json.dumps(
        user_input, ensure_ascii=False
    )

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
    }

    resp = requests.post(f"{host}/api/generate", json=payload)
    resp.raise_for_status()

    out = resp.json()["response"].strip()

    parsed = extract_json_block(out)
    if parsed is not None:
        return {
            "User_id": parsed.get("User_id", user_input["current_user"]),
            "profile": parsed.get("profile", None),
            "raw": out,
        }

    return {
        "User_id": user_input["current_user"],
        "profile": None,
        "raw": out,
    }


# ==================================
# 4. Main Pipeline
# ==================================
def main():
    # Load user interaction sequences (raw IDs) for each domain
    domain_user_seq_raw = {}

    for domain in DOMAINS:
        domain_dir = os.path.join(INTER_ROOT, domain)
        maps_dir = os.path.join(domain_dir, "maps")

        user2id = load_id_map(os.path.join(maps_dir, "user2id.txt"))
        item2id = load_id_map(os.path.join(maps_dir, "item2id.txt"))
        train_seq = load_train(os.path.join(domain_dir, "train.txt"))

        user_seq_raw = defaultdict(list)
        for u_idx, items in train_seq.items():
            u_raw = user2id[u_idx]
            for it_idx in items:
                asin = item2id[it_idx]
                user_seq_raw[u_raw].append(asin)

        domain_user_seq_raw[domain] = user_seq_raw

    # Identify users common to both domains
    overlap = sorted(set.intersection(*(set(domain_user_seq_raw[d].keys()) for d in DOMAINS)))

    print(f"Common users: {len(overlap)}")

    if len(overlap) < NUM_USERS:
        target_users = overlap
    else:
        random.seed(2025)
        target_users = random.sample(overlap, NUM_USERS)

    print(f"Target: {len(target_users)} users")

    # Load item title/description text
    domain_item_text = {
        domain: load_item_text(META_JSON[domain]) for domain in DOMAINS
    }

    # Profile generation per domain
    for domain in DOMAINS:
        print(f"\n[Generating profiles for domain: {domain}]")

        result_dict = {}

        for idx, user_id in enumerate(tqdm(target_users, desc=f"{domain}")):
            history = {}
            seq = domain_user_seq_raw[domain].get(user_id, [])
            if not seq:
                continue

            recent_items = seq[-MAX_LEN_PER_DOMAIN:]
            item_text_map = domain_item_text[domain]

            for asin in recent_items:
                txt = item_text_map.get(asin, "").strip()
                if txt:
                    history[asin] = txt

            if not history:
                continue

            user_input = {
                "current_user": user_id,
                "history": history,
            }

            result = call_ollama(
                model=OLLAMA_MODEL,
                host=OLLAMA_HOST,
                base_prompt=BASE_PROMPT,
                user_input=user_input,
            )

            user_key = f"User_{idx}"
            cleaned = {
                "User_id": result["User_id"],
                "profile": result["profile"],
                "raw": result["raw"],
            }

            result_dict[user_key] = cleaned

            print(f"\n[{user_key} | raw_id={user_id}]")
            print("Profile:")
            print(cleaned["profile"])
            print("-" * 50)

        with open(OUTPUT_JSON[domain], "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        print(f"Saved domain JSON → {OUTPUT_JSON[domain]}")


if __name__ == "__main__":
    main()
