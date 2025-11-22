import os
import json
from collections import defaultdict

from tqdm import tqdm

BASE_DIR = "/home/heek/edda_backbone/preprocess_raw/amazon/23"
USER_ID_PATH = "/home/heek/aigs/InitGrounder/data/amazon/f_usr_id.json"
REVIEW_DIR = os.path.join(BASE_DIR, "user_reviews/5_core")
META_DIR = os.path.join(BASE_DIR, "item_meta")
OUTPUT_DIR = "/home/heek/aigs/InitGrounder/data/amazon"
OUTPUT_META_DIR = "/home/heek/aigs/InitGrounder/data/amazon"

REQUIRED_DOMAINS = [
    "Home_and_Kitchen",
    "Electronics",
]


def load_selected_users(path: str):
    selected_users = set()
    with open(path, "r") as f:
        try:
            users = json.load(f)
            for obj in users:
                user_id = obj.get("user_id")
                if user_id:
                    selected_users.add(user_id)
        except Exception as e:
            print(f"Failed to load user file: {e}")
    print(f"#User: {len(selected_users):,}")
    return selected_users


def filter_reviews_and_collect_items(selected_users):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    user_item_ids_per_domain = defaultdict(set)

    for domain in REQUIRED_DOMAINS:
        input_path = os.path.join(REVIEW_DIR, f"{domain}.json")
        if not os.path.exists(input_path):
            print(f"Review file not found for domain '{domain}': {input_path}")
            continue

        with open(input_path, "r") as f:
            total_lines = sum(1 for _ in f)

        filtered_reviews = []

        with open(input_path, "r") as f_in:
            for line in tqdm(f_in, desc=f"Processing {domain}", total=total_lines):
                try:
                    entry = json.loads(line)
                    user_id = entry.get("user_id")
                    if user_id not in selected_users:
                        continue
                    filtered_reviews.append(entry)
                    item_id = entry.get("parent_asin") or entry.get("asin")
                    if item_id:
                        user_item_ids_per_domain[domain].add(item_id)
                except Exception:
                    continue

        output_path = os.path.join(OUTPUT_DIR, f"f_{domain}_rev.json")
        with open(output_path, "w") as f_out:
            json.dump(filtered_reviews, f_out, indent=2)

        print(
            f"Saved {len(filtered_reviews)} entries for domain '{domain}' "
            f"to {output_path}"
        )

    return user_item_ids_per_domain


def filter_meta(user_item_ids_per_domain):
    os.makedirs(OUTPUT_META_DIR, exist_ok=True)

    for domain, item_ids in user_item_ids_per_domain.items():
        if not item_ids:
            print(f"No items collected for domain '{domain}', skipping meta filter")
            continue

        meta_path = os.path.join(META_DIR, f"meta_{domain}.jsonl")
        if not os.path.exists(meta_path):
            print(f"no item meta data discovered: {meta_path}")
            continue

        with open(meta_path, "r") as f_in:
            lines = list(f_in)

        filtered_meta = []
        for line in tqdm(lines, desc=f"Filtering {domain} meta", total=len(lines)):
            try:
                record = json.loads(line)
                item_id = record.get("parent_asin") or record.get("asin")
                if item_id in item_ids:
                    filtered_meta.append(record)
            except Exception:
                continue

        output_path = os.path.join(OUTPUT_META_DIR, f"f_meta_{domain}.json")
        with open(output_path, "w") as f_out:
            json.dump(filtered_meta, f_out, indent=2)

        print(f"Saved {len(filtered_meta):,} items to {output_path}")


def main():
    selected_users = load_selected_users(USER_ID_PATH)
    user_item_ids_per_domain = filter_reviews_and_collect_items(selected_users)
    filter_meta(user_item_ids_per_domain)
    print("Process completed")


if __name__ == "__main__":
    main()

