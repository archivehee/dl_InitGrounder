import os
import json
import pickle
from collections import OrderedDict

import numpy as np
import requests
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
AMAZON_ROOT = os.path.join(PROJECT_ROOT, "data", "amazon")
INTER_ROOT = os.path.join(AMAZON_ROOT, "inter_cdr")

DOMAINS = ["Electronics", "Home_and_Kitchen"]

USER_PROF_JSON = {
    "Electronics": os.path.join(AMAZON_ROOT, "usr_prof_Electronics.json"),
    "Home_and_Kitchen": os.path.join(AMAZON_ROOT, "usr_prof_Home_and_Kitchen.json"),
}

OUTPUT_ROOT = os.path.join(AMAZON_ROOT, "user_prof_emb")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

EMB_MODEL = "nomic-embed-text-v1.5"
NOMIC_API_KEY = os.environ.get("NOMIC_API_KEY")
NOMIC_ENDPOINT = "https://api-atlas.nomic.ai/v1/embedding/text"


def get_embedding(text):
    if NOMIC_API_KEY is None:
        raise RuntimeError("NOMIC_API_KEY 환경변수가 설정되어 있지 않습니다.")

    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMB_MODEL,
        "texts": [text],             
        "dimensionality": 64,        
        # "task_type": "search_document",  
    }

    resp = requests.post(NOMIC_ENDPOINT, headers=headers, json=payload)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Nomic embedding API 호출 실패: status={resp.status_code}, body={resp.text}"
        )

    data = resp.json()
    emb = np.array(data["embeddings"][0], dtype=np.float32)
    return emb

def load_user2id(domain: str):
    path = os.path.join(INTER_ROOT, domain, "maps", "user2id.txt")
    idx2user = []
    user2idx = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Unexpected format in {path}: {line}")
            raw_uid, idx_str = parts
            idx = int(idx_str)

            if idx >= len(idx2user):
                idx2user.extend([None] * (idx - len(idx2user) + 1))
            idx2user[idx] = raw_uid
            user2idx[raw_uid] = idx

    if any(u is None for u in idx2user):
        raise ValueError(f"Some user indices are missing in {path}")

    return idx2user, user2idx


def load_profiles(domain: str):
    path = USER_PROF_JSON[domain]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = {}
    for _, v in data.items():
        if not isinstance(v, dict):
            continue

        raw_uid = v.get("User_id")
        if raw_uid is None:
            continue

        profile_text = v.get("profile")

        if profile_text is None:
            profiles[raw_uid] = None
        else:
            s = str(profile_text).strip()
            profiles[raw_uid] = s if s != "" else None

    return profiles


def build_user_text_list(idx2user, profiles_by_uid):
    texts = []
    for uid in idx2user:
        texts.append(profiles_by_uid.get(uid, None))
    return texts


def process_domain(domain: str):
    print(f"\n=== Domain: {domain} ===")

    idx2user, user2idx = load_user2id(domain)
    n_users = len(idx2user)
    print(f"#users: {n_users}")

    profiles_by_uid = load_profiles(domain)
    print(f"#profiles in json: {len(profiles_by_uid)}")

    user_texts = build_user_text_list(idx2user, profiles_by_uid)

    all_embs = []
    emb_dim = None

    for text in tqdm(user_texts, desc=f"Embedding {domain} users"):
        if text is None:
            if emb_dim is None:
                dummy = get_embedding("[DUMMY PROFILE]")
                emb_dim = dummy.shape[0]
            emb = np.zeros(emb_dim, dtype=np.float32)
        else:
            emb = get_embedding(text)
            if emb_dim is None:
                emb_dim = emb.shape[0]

        all_embs.append(emb)

    user_emb = np.stack(all_embs, axis=0)
    _, dim = user_emb.shape
    print(f"user_emb shape: {user_emb.shape}")

    out_path = os.path.join(OUTPUT_ROOT, f"usr_t_emb_{domain}.pkl")

    save_dict = OrderedDict()
    save_dict["user_emb"] = user_emb
    save_dict["idx2user"] = idx2user
    save_dict["user2idx"] = user2idx
    save_dict["meta"] = {
        "domain": domain,
        "model": EMB_MODEL,
        "dim": dim,
        "zero_profile_users": int(sum(t is None for t in user_texts)),
    }

    with open(out_path, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {out_path}")


def main():
    for d in DOMAINS:
        process_domain(d)


if __name__ == "__main__":
    main()
