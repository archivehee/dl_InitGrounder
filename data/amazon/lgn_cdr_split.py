from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

USER_ID_PATH = Path("/home/heek/aigs/InitGrounder/data/amazon/f_usr_id.json")
REVIEW_DIR = Path("/home/heek/aigs/InitGrounder/data/amazon")
OUT_ROOT = Path("./data/amazon/lgn_cdr/")
DOMAINS: Sequence[str] = ("Electronics", "Home_and_Kitchen")
SPLITS: Sequence[str] = ("train", "valid", "test")


def load_user_pool(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        raw_users = json.load(f)
    ordered_users: List[str] = []
    seen = set()
    for entry in raw_users:
        uid = entry.get("user_id")
        if uid and uid not in seen:
            ordered_users.append(uid)
            seen.add(uid)
    if not ordered_users:
        raise RuntimeError(f"No user ids discovered in {path}")
    return ordered_users


def parse_domain_reviews(domain: str, allowed_users: Iterable[str]) -> Tuple[Dict[str, List[Tuple[int, int, str]]], List[str]]:
    allowed = set(allowed_users)
    domain_path = REVIEW_DIR / f"f_{domain}_rev.json"
    if not domain_path.exists():
        raise FileNotFoundError(f"Missing review file for domain '{domain}': {domain_path}")

    with domain_path.open("r", encoding="utf-8") as f:
        reviews = json.load(f)

    histories: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    item_ids: List[str] = []

    for seq, entry in enumerate(reviews):
        user_id = entry.get("user_id")
        if user_id not in allowed:
            continue
        item_id = entry.get("parent_asin") or entry.get("asin")
        if not item_id:
            continue

        ts = entry.get("timestamp")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            ts_val = int(ts)
        else:
            ts_val = seq

        histories[user_id].append((ts_val, seq, item_id))
        item_ids.append(item_id)

    if not histories:
        raise RuntimeError(f"No overlapping users found in domain '{domain}'")
    return histories, item_ids


def build_user_map(user_pool: Sequence[str], seen_users: Iterable[str]) -> Dict[str, int]:
    seen = set(seen_users)
    ordered: List[str] = []
    for uid in user_pool:
        if uid in seen:
            ordered.append(uid)
            seen.remove(uid)
    if seen:
        ordered.extend(sorted(seen))
    return {uid: idx for idx, uid in enumerate(ordered)}


def split_history(history: List[Tuple[int, int, str]]) -> Dict[str, List[str]]:
    sorted_hist = sorted(history, key=lambda x: (x[0], x[1]))
    items = [item for _, _, item in sorted_hist]
    splits = {"train": [], "valid": [], "test": []}
    if not items:
        return splits
    if len(items) == 1:
        splits["train"] = items
        return splits
    if len(items) == 2:
        splits["train"] = items[:-1]
        splits["test"] = items[-1:]
        return splits
    splits["train"] = items[:-2]
    splits["valid"] = items[-2:-1]
    splits["test"] = items[-1:]
    return splits


def write_pairs(path: Path, pairs: List[Tuple[int, int]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for u, i in pairs:
            f.write(f"{u}\t{i}\n")


def write_map(path: Path, mapping: Dict[str, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for raw, idx in mapping.items():
            f.write(f"{raw}\t{idx}\n")


def main():
    user_pool = load_user_pool(USER_ID_PATH)

    domain_histories: Dict[str, Dict[str, List[Tuple[int, int, str]]]] = {}
    domain_items: Dict[str, List[str]] = {}
    seen_users = set()

    for domain in DOMAINS:
        histories, item_ids = parse_domain_reviews(domain, user_pool)
        domain_histories[domain] = histories
        domain_items[domain] = sorted(set(item_ids))
        seen_users.update(histories.keys())

    global_user_map = build_user_map(user_pool, seen_users)
    if not global_user_map:
        raise RuntimeError("No overlapping users detected across domains")

    # Build contiguous global item ids (domain-aware)
    global_item_map: Dict[str, int] = {}
    domain_item_ranges: Dict[str, Dict[str, int]] = {}
    current_idx = 0
    for domain in DOMAINS:
        start = current_idx
        for item in domain_items[domain]:
            raw_key = f"{domain}::{item}"
            global_item_map[raw_key] = current_idx
            current_idx += 1
        domain_item_ranges[domain] = {"start": start, "end": current_idx}

    split_pairs: Dict[str, List[Tuple[int, int]]] = {split: [] for split in SPLITS}
    domain_split_counts: Dict[str, Dict[str, int]] = {
        domain: {split: 0 for split in SPLITS} for domain in DOMAINS
    }

    for domain in DOMAINS:
        histories = domain_histories[domain]
        for user_id, history in histories.items():
            user_idx = global_user_map.get(user_id)
            if user_idx is None:
                continue
            user_splits = split_history(history)
            for split_name, items in user_splits.items():
                for item in items:
                    raw_key = f"{domain}::{item}"
                    item_idx = global_item_map.get(raw_key)
                    if item_idx is None:
                        continue
                    split_pairs[split_name].append((user_idx, item_idx))
                    domain_split_counts[domain][split_name] += 1

    # sort for determinism
    for split_name in SPLITS:
        split_pairs[split_name] = sorted(split_pairs[split_name])

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    map_dir = OUT_ROOT / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    write_map(map_dir / "user2id.txt", global_user_map)
    write_map(map_dir / "item2id.txt", global_item_map)

    split_stats = {}
    for split_name, pairs in split_pairs.items():
        write_pairs(OUT_ROOT / f"{split_name}.txt", pairs)
        split_stats[split_name] = len(pairs)

    manifest = {
        "user_id_path": str(USER_ID_PATH),
        "review_dir": str(REVIEW_DIR),
        "out_root": str(OUT_ROOT),
        "domains": list(DOMAINS),
        "users": len(global_user_map),
        "items": current_idx,
        "domain_item_ranges": domain_item_ranges,
        "split_counts": split_stats,
        "domain_split_counts": domain_split_counts,
    }

    with (OUT_ROOT / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote merged CDR splits to {OUT_ROOT}")


if __name__ == "__main__":
    main()
