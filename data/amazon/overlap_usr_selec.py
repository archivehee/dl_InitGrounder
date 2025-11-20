import os
import json
import random
from tqdm import tqdm

# ===== 설정 =====
BASE_DIR = "/home/heek/edda_backbone/preprocess_raw/amazon/23"
REVIEW_DIR = os.path.join(BASE_DIR, "user_reviews/5_core")
META_DIR = os.path.join(BASE_DIR, "item_meta")
OUTPUT_PATH = "/home/heek/aigs/InitGrounder/data/amazon/f_usr_id.json"

REQUIRED_DOMAINS = [
    "Home_and_Kitchen",
    "Electronics",
]

REQUIRED_FIELDS = ["main_category", "categories", "title", "description"]
MAX_USERS = 3000
MIN_INTERACTIONS_PER_DOMAIN = 5

# ===== 유효성 검사 함수 =====
def is_valid_field(value):
    if value in [None, "", [], {}, "null"]:
        return False

    # 문자열 처리
    str_val = str(value).strip().lower()
    if str_val in ["", "[]", "{}", "none", "null", "[ ]"]:
        return False

    # 문자열로 된 리스트인 척 하는 경우도 체크
    if str_val.startswith("[") and str_val.endswith("]"):
        try:
            parsed = json.loads(str_val.replace("'", '"'))
            if isinstance(parsed, list) and all(str(v).strip().lower() in ["", "none", "null"] for v in parsed):
                return False
        except:
            pass

    # 실제 list 객체
    if isinstance(value, list) and all(str(v).strip().lower() in ["", "none", "null"] for v in value):
        return False

    return True

# ===== 1. 메타데이터 기준 유효 asin 추출 =====
domain_valid_asins = {}

for domain in REQUIRED_DOMAINS:
    meta_path = os.path.join(META_DIR, f"meta_{domain}.jsonl")
    valid_asins = set()
    with open(meta_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                asin = entry.get("parent_asin") or entry.get("asin")
                if asin and all(is_valid_field(entry.get(field)) for field in REQUIRED_FIELDS):
                    valid_asins.add(asin)
            except:
                continue
    domain_valid_asins[domain] = valid_asins

# ===== 2. 도메인별 유효 asin 기반 리뷰 유저 수집 =====
domain_user_valid_reviews = {domain: {} for domain in REQUIRED_DOMAINS}

for domain in REQUIRED_DOMAINS:
    review_path = os.path.join(REVIEW_DIR, f"{domain}.json")
    valid_asins = domain_valid_asins[domain]

    with open(review_path, "r") as f:
        for line in tqdm(f, desc=f"Loading {domain}", dynamic_ncols=True):
            try:
                entry = json.loads(line)
                user_id = entry.get("user_id")
                asin = entry.get("parent_asin") or entry.get("asin")
                if user_id and asin in valid_asins:
                    domain_user_valid_reviews[domain].setdefault(user_id, 0)
                    domain_user_valid_reviews[domain][user_id] += 1
            except:
                continue


# ===== 3. 모든 도메인에 등장하는 유저만 필터링 =====
candidate_users = set.intersection(*[set(users.keys()) for users in domain_user_valid_reviews.values()])
print(f"Valid overlapping users with meta-based 2 non-empty fields in all 2 domains: {len(candidate_users)}")

# ===== 4. 최소 인터랙션 조건 만족 유저만 필터링 =====
qualified_users = [
    user_id
    for user_id in candidate_users
    if all(domain_user_valid_reviews[domain].get(user_id, 0) >= MIN_INTERACTIONS_PER_DOMAIN for domain in REQUIRED_DOMAINS)
]
print(
    f"Users with >= {MIN_INTERACTIONS_PER_DOMAIN} interactions in each domain: "
    f"{len(qualified_users)}"
)

if len(qualified_users) < MAX_USERS:
    raise RuntimeError(
        f"Not enough users meeting the interaction threshold: "
        f"required {MAX_USERS}, available {len(qualified_users)}"
    )

# ===== 4. 랜덤 샘플링 =====
random.seed(2025)
sampled_users = random.sample(qualified_users, MAX_USERS)

# ===== 5. 저장 =====
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump([{"user_id": uid} for uid in sampled_users], f, indent=2)

print(f"Saved {len(sampled_users)} users to {OUTPUT_PATH}")
