from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Path configuration for CDR preprocessing
USER_ID_PATH = Path("/home/heek/aigs/InitGrounder/data/amazon/f_usr_id.json")
REVIEW_DIR = Path("/home/heek/aigs/InitGrounder/data/amazon")
OUT_ROOT = Path("./data/amazon/inter_cdr/")
DOMAINS: Sequence[str] = ("Home_and_Kitchen", "Electronics")
SPLITS: Sequence[str] = ("train", "valid", "test")


def load_user_pool(path: Path) -> List[str]:
    """Load candidate user ids while preserving file order."""
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
    """
    Parse domain review file and collect per-user histories.

    Returns (histories, items) where histories maps user_id->[(ts, seq, item_id)].
    """
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
    """Build deterministic user map prioritising the id order from user_pool."""
    seen = set(seen_users)
    ordered: List[str] = []
    for uid in user_pool:
        if uid in seen:
            ordered.append(uid)
            seen.remove(uid)
    if seen:
        ordered.extend(sorted(seen))
    return {uid: idx for idx, uid in enumerate(ordered)}


def build_item_map(item_ids: Iterable[str]) -> Dict[str, int]:
    unique = sorted(set(item_ids))
    return {iid: idx for idx, iid in enumerate(unique)}


def split_history(history: List[Tuple[int, int, str]]) -> Dict[str, List[str]]:
    """Leave-two-out split (last -> test, second last -> valid)."""
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
    domain_items: Dict[str, Dict[str, int]] = {}
    seen_users = set()

    for domain in DOMAINS:
        histories, item_ids = parse_domain_reviews(domain, user_pool)
        domain_histories[domain] = histories
        domain_items[domain] = build_item_map(item_ids)
        seen_users.update(histories.keys())

    global_user_map = build_user_map(user_pool, seen_users)
    if not global_user_map:
        raise RuntimeError("No overlapping users detected across domains")

    manifest = {
        "user_id_path": str(USER_ID_PATH),
        "review_dir": str(REVIEW_DIR),
        "out_root": str(OUT_ROOT),
        "domains": list(DOMAINS),
        "users": len(global_user_map),
        "items_per_domain": {domain: len(item_map) for domain, item_map in domain_items.items()},
        "splits": {},
    }

    for domain in DOMAINS:
        histories = domain_histories[domain]
        item_map = domain_items[domain]
        out_dir = OUT_ROOT / domain
        map_dir = out_dir / "maps"
        map_dir.mkdir(parents=True, exist_ok=True)
        write_map(map_dir / "user2id.txt", global_user_map)
        write_map(map_dir / "item2id.txt", item_map)

        split_pairs: Dict[str, List[Tuple[int, int]]] = {split: [] for split in SPLITS}
        for user_id, history in histories.items():
            user_idx = global_user_map.get(user_id)
            if user_idx is None:
                continue
            user_splits = split_history(history)
            for split_name, items in user_splits.items():
                item_indices = [(user_idx, item_map[item]) for item in items if item in item_map]
                split_pairs[split_name].extend(item_indices)

        stats = {}
        for split_name, pairs in split_pairs.items():
            pairs_sorted = sorted(pairs)
            write_pairs(out_dir / f"{split_name}.txt", pairs_sorted)
            stats[split_name] = len(pairs_sorted)
        manifest["splits"][domain] = stats

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote CDR splits to {OUT_ROOT}")


if __name__ == "__main__":
    main()
