import os
import json
from tqdm import tqdm

# path setting
BASE_DIR = "/home/heek/edda_backbone/preprocess_raw/amazon/23"
USER_ID_PATH = "/home/heek/aigs/InitGrounder/data/amazon/f_usr_id.json"
REVIEW_DIR = os.path.join(BASE_DIR, "user_reviews/5_core")
OUTPUT_DIR = "/home/heek/aigs/InitGrounder/data/amazon"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. User ID set up
target_users = set()
with open(USER_ID_PATH, "r") as f:
    try:
        user_list = json.load(f)
        for obj in user_list:
            uid = obj.get("user_id")
            if uid:
                target_users.add(uid)
    except:
        print("유저 ID 파일 로딩 실패")

print(f"Loaded {len(target_users)} user IDs")

# 2. 도메인별 리뷰 필터링
for fname in os.listdir(REVIEW_DIR):
    if not fname.endswith(".json"):
        continue

    domain = fname.replace(".json", "")
    input_path = os.path.join(REVIEW_DIR, fname)
    output_path = os.path.join(OUTPUT_DIR, f"{domain}.json")  

    with open(input_path, "r") as f:
        total_lines = sum(1 for _ in f)

    filtered_reviews = []

    with open(input_path, "r") as f_in:
        for line in tqdm(f_in, desc=f"Processing {domain}", total=total_lines):
            try:
                entry = json.loads(line)
                user_id = entry.get("user_id")
                if user_id in target_users:
                    filtered_reviews.append(entry)
            except:
                continue

    with open(output_path, "w") as f_out:
        json.dump(filtered_reviews, f_out, indent=2)

    print(f"Saved {len(filtered_reviews)} entries for domain '{domain}' to {output_path}")