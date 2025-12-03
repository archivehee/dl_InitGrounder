import os
import pickle
import time
from os.path import join

import numpy as np
import torch
from scipy.sparse import csr_matrix
from tensorboardX import SummaryWriter

import Procedure
import register
import utils
import world
from register import dataset
from world import cprint

MASK_USER_COUNT = 600
MASK_SEED = 2025
MASK_ENABLED = bool(getattr(world, "MASK_USERS", 1))

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================


def _filter_pairs(users_arr, items_arr, mask_values):
    """
    Remove interactions involving masked users.
    Returns filtered arrays and the number of removed pairs.
    """
    if users_arr is None or len(users_arr) == 0:
        return users_arr, items_arr, 0
    keep_mask = ~np.isin(users_arr, mask_values)
    removed = int(len(users_arr) - int(np.sum(keep_mask)))
    return users_arr[keep_mask], items_arr[keep_mask], removed


def _save_masked_users(user_ids, dataset_name, seed_value):
    if user_ids.size == 0:
        return None
    mask_dir = os.path.join(world.FILE_PATH, "masked_users")
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(
        mask_dir,
        f"{dataset_name}_seed{seed_value}_n{user_ids.size}.txt"
    )
    np.savetxt(mask_path, user_ids, fmt="%d")
    return mask_path


def mask_dataset_interactions(ds, target_count=MASK_USER_COUNT, seed=MASK_SEED):
    """
    Select `target_count` users from the test split (seeded) and
    drop all of their interactions from train/valid so that only
    the test pair is left for evaluation.
    """
    if not hasattr(ds, "testUniqueUsers"):
        raise ValueError("Masking is only supported for InterCDRDataset.")

    candidates = getattr(ds, "testUniqueUsers", None)
    if candidates is None or len(candidates) == 0:
        raise ValueError("No test users available for masking.")

    mask_size = min(target_count, len(candidates))
    rng = np.random.default_rng(seed)
    masked_users = np.sort(rng.choice(candidates, size=mask_size, replace=False).astype(np.int64))
    ds.masked_users = masked_users

    # Remove train/valid interactions for the held-out users.
    original_train = len(ds.trainUser)
    ds.trainUser, ds.trainItem, removed_train = _filter_pairs(ds.trainUser, ds.trainItem, masked_users)
    ds.traindataSize = len(ds.trainUser)
    ds.trainUniqueUsers = np.unique(ds.trainUser)

    removed_valid = 0
    if hasattr(ds, "validUser"):
        ds.validUser, ds.validItem, removed_valid = _filter_pairs(ds.validUser, ds.validItem, masked_users)
        if hasattr(ds, "rebuild_domain_eval_dicts"):
            ds.rebuild_domain_eval_dicts()
        elif hasattr(ds, "refresh_eval_dictionaries"):
            ds.refresh_eval_dictionaries()
        elif hasattr(ds, "_InterCDRDataset__build_dict"):
            build_dict = getattr(ds, "_InterCDRDataset__build_dict")
            ds._InterCDRDataset__validDict = build_dict(ds.validUser, ds.validItem)

    # Rebuild graph-related tensors so downstream sampling sees the masked data.
    ds.UserItemNet = csr_matrix(
        (np.ones(len(ds.trainUser)), (ds.trainUser, ds.trainItem)),
        shape=(ds.n_user, ds.m_item)
    )
    ds.users_D = np.array(ds.UserItemNet.sum(axis=1)).squeeze()
    ds.users_D[ds.users_D == 0.] = 1
    ds.items_D = np.array(ds.UserItemNet.sum(axis=0)).squeeze()
    ds.items_D[ds.items_D == 0.] = 1.
    ds._allPos = ds.getUserPosItems(list(range(ds.n_user)))
    ds.Graph = None  # force LightGCN to rebuild adjacency with masked interactions
    setattr(ds, "mask_active", True)
    if hasattr(ds, "rebuild_domain_eval_dicts"):
        ds.rebuild_domain_eval_dicts()

    cprint(f"[MASK] held out {mask_size} users (seed={seed}). "
           f"Removed {removed_train} train and {removed_valid} valid interactions "
           f"(train size: {original_train} -> {ds.traindataSize}).")
    return masked_users


if MASK_ENABLED:
    masked_users = mask_dataset_interactions(dataset)
    mask_path = _save_masked_users(masked_users, world.dataset, MASK_SEED)
    if mask_path:
        cprint(f"[MASK] saved held-out user ids to {mask_path}")
else:
    masked_users = np.array([], dtype=np.int64)
    cprint("[MASK] disabled (use --mask_users 1 to enable)")

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    best_ndcg20 = -1.0
    best_epoch = -1
    no_improve_cnt = 0
    patience = 10
    try:
        ndcg20_idx = list(world.topks).index(20)
    except ValueError:
        ndcg20_idx = len(world.topks) - 1

    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
        if (epoch + 1) % 5 == 0:
            cprint("[VALIDATION]")
            val_dict = getattr(dataset, "validDict", None)
            val_results = Procedure.Test(dataset,
                                         Recmodel,
                                         epoch,
                                         w,
                                         world.config['multicore'],
                                         eval_dict=val_dict,
                                         split_name="valid")
            current_ndcg20 = float(val_results['ndcg'][ndcg20_idx])
            if current_ndcg20 > best_ndcg20:
                best_ndcg20 = current_ndcg20
                best_epoch = epoch
                no_improve_cnt = 0
                best_state_dict = Recmodel.state_dict()
                cprint(f"[NEW BEST] epoch {epoch + 1}, NDCG@20={best_ndcg20:.6f}")
            else:
                no_improve_cnt += 1
                cprint(f"[NO IMPROVE] {no_improve_cnt}/{patience} validations")
                if no_improve_cnt >= patience:
                    cprint(f"[EARLY STOP] no NDCG@20 improvement for {patience} validations")
                    break
finally:
    if world.tensorboard:
        w.close()
    try:
        if 'best_state_dict' in locals():
            Recmodel.load_state_dict(best_state_dict)
            cprint(f"[LOAD BEST] epoch {best_epoch + 1} with NDCG@20={best_ndcg20:.6f}")
    except Exception as e:
        print(f"failed to load best state dict: {e}")
    cprint("[FINAL TEST]")
    Procedure.Test(dataset,
                   Recmodel,
                   world.TRAIN_epochs,
                   None,
                   world.config['multicore'],
                   eval_dict=getattr(dataset, "testDict", None),
                   split_name="test")
    domain_tests = getattr(dataset, "domain_test_dicts", None)
    if domain_tests:
        for domain_name, eval_dict in domain_tests.items():
            if not eval_dict:
                continue
            cprint(f"[FINAL TEST][{domain_name}]")
            Procedure.Test(dataset,
                           Recmodel,
                           world.TRAIN_epochs,
                           None,
                           world.config['multicore'],
                           eval_dict=eval_dict,
                           split_name=f"test-{domain_name}")
    if world.SAVE_PRETRAIN:
        save_dir = world.PRETRAIN_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"user_{world.dataset}.pkl")
        user_emb = Recmodel.embedding_user.weight.detach().cpu().numpy()
        with open(save_path, "wb") as fout:
            pickle.dump(user_emb, fout)
        cprint(f"[SAVE USER EMB] {save_path}")
