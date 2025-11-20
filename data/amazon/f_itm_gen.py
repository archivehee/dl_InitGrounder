import os
import json
from tqdm import tqdm
from collections import defaultdict

# path setting
BASE_DIR = "/home/heek/edda_backbone/preprocess_raw/amazon/23"
REVIEW_DIR = os.path.join(BASE_DIR, "user_reviews/5_core")
META_DIR = os.path.join(BASE_DIR, "item_meta")
OUTPUT_META_DIR = "/home/heek/aigs/InitGrounder/data/amazon"   
USER_ID_PATH = "/home/heek/aigs/InitGrounder/data/amazon/f_usr_id.json"  

# 1. User ID set up
selected_users = set()
with open(USER_ID_PATH, "r") as f:
    try:
        users = json.load(f)
        for obj in users:
            user_id = obj.get("user_id")
            if user_id:
                selected_users.add(user_id)
    except Exception as e:
        print(f"Failed to load user file: {e}")


print(f"#User: {len(selected_users):,}")

# 2. 도메인별 user-review 파일에서 해당 유저의 item_id 수집
user_item_ids_per_domain = defaultdict(set)

for file_name in os.listdir(REVIEW_DIR):
    if not file_name.endswith(".json"):  # ← 확장자 체크 수정
        continue
    domain = file_name.replace(".json", "")
    file_path = os.path.join(REVIEW_DIR, file_name)

    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)

    with open(file_path, "r") as f:
        for line in tqdm(f, desc=f"Scanning {domain}", total=total_lines):
            try:
                record = json.loads(line)  # ← 한 줄씩 로드
                if record.get("user_id") in selected_users:
                    item_id = record.get("parent_asin") or record.get("asin")
                    if item_id:
                        user_item_ids_per_domain[domain].add(item_id)
            except:
                continue


# 3. 각 도메인의 메타데이터에서 해당 item_id만 필터링하여 저장
os.makedirs(OUTPUT_META_DIR, exist_ok=True)

for domain, item_ids in user_item_ids_per_domain.items():
    meta_path = os.path.join(META_DIR, f"meta_{domain}.jsonl")
    output_path = os.path.join(OUTPUT_META_DIR, f"f_meta_{domain}.json")  # ← 확장자 수정
    
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
        except:
            continue

    with open(output_path, "w") as f_out:
        json.dump(filtered_meta, f_out, indent=2)

    print(f"Saved {len(filtered_meta):,} items to {output_path}")

print("Process completed")