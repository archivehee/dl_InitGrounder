import os
import sys
import time
import argparse
import torch
import numpy as np
import csv
from os.path import join
from scipy.sparse import csr_matrix
from tensorboardX import SummaryWriter

# [Project Imports]
ROOT_PATH = "/home/parkdw00/Codes/DL-HW/dl_InitGrounder-main"
CODE_PATH = join(ROOT_PATH, "model/LightGCN-PyTorch/code")
sys.path.append(CODE_PATH)

import utils
import Procedure
from dataloader import InterCDRDataset 
from ours import Ours 
import world
from world import cprint

# ============================================================================
# 1. Parse Arguments
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Ours Model Parameter Tuning")
    
    # [Tuning Parameters] - 실험 대상
    parser.add_argument('--layer', type=int, default=3, help="number of LightGCN layers")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument('--dropout', type=int, default=0, help="enable dropout (0: off, 1: on)")
    parser.add_argument('--keepprob', type=float, default=0.6, help="dropout keep probability (only if dropout=1)")
    parser.add_argument('--top_k_neighbors', type=int, default=10, help='K neighbors for injection')
    parser.add_argument('--num_neg', type=int, default=999, help='candidate negative items for evaluation')
    
    # [Training Settings]
    parser.add_argument('--epochs', type=int, default=1000, help="training epochs")
    parser.add_argument('--bpr_batch', type=int, default=2048, help="batch size")
    parser.add_argument('--recdim', type=int, default=64, help="embedding size")
    parser.add_argument('--a_fold', type=int, default=100, help="fold num for large adj matrix")
    parser.add_argument('--testbatch', type=int, default=100, help="test batch size")
    
    # [Others]
    parser.add_argument('--topks', nargs='?', default="[10, 20]", help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1, help="enable tensorboard")
    parser.add_argument('--load', type=int, default=0, help="load saved model")
    parser.add_argument('--multicore', type=int, default=0, help='multiprocessing')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model type')
    
    # [Data & Paths]
    #parser.add_argument('--dataset', type=str, default='Home_and_Kitchen', help="Target Domain Dataset")
    parser.add_argument('--dataset', type=str, default='Electronics', help="Target Domain Dataset")
    parser.add_argument('--data_path', type=str, default=join(ROOT_PATH, 'data/amazon/inter_cdr/'), help="Data root path")
    parser.add_argument('--path', type=str, default="./checkpoints", help="path to save weights")
    parser.add_argument('--board_path', type=str, default="./runs", help="tensorboard path")
    
    # parser.add_argument('--src_profile', type=str, 
    #                     default=join(ROOT_PATH, 'data/amazon/user_prof_emb/usr_t_emb_Electronics.pkl'))
    # parser.add_argument('--tgt_profile', type=str, 
    #                     default=join(ROOT_PATH, 'data/amazon/user_prof_emb/usr_t_emb_Home_and_Kitchen.pkl'))
    # parser.add_argument('--src_lgn', type=str, 
    #                     default=join(ROOT_PATH, 'model/LightGCN-PyTorch/lgn_pre_emb/user_Electronics.pkl'))
    parser.add_argument('--src_profile', type=str, 
                        default=join(ROOT_PATH, 'data/amazon/user_prof_emb/usr_t_emb_Home_and_Kitchen.pkl'))
    parser.add_argument('--tgt_profile', type=str, 
                        default=join(ROOT_PATH, 'data/amazon/user_prof_emb/usr_t_emb_Electronics.pkl'))
    parser.add_argument('--src_lgn', type=str, 
                        default=join(ROOT_PATH, 'model/LightGCN-PyTorch/lgn_pre_emb/user_Home_and_Kitchen.pkl'))


    return parser.parse_args()

# ============================================================================
# 2. Helper Functions
# ============================================================================
def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
    
def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def get_config(args):
    config = {}
    config['bpr_batch_size'] = args.bpr_batch
    config['latent_dim_rec'] = args.recdim
    config['lightGCN_n_layers'] = args.layer
    config['dropout'] = args.dropout
    config['keep_prob'] = args.keepprob
    config['A_n_fold'] = args.a_fold
    config['test_u_batch_size'] = args.testbatch
    config['multicore'] = args.multicore
    config['lr'] = args.lr
    config['decay'] = args.decay
    config['pretrain'] = 0
    config['A_split'] = False
    
    # Ours Specific
    config['src_profile'] = args.src_profile
    config['tgt_profile'] = args.tgt_profile
    config['src_lgn'] = args.src_lgn
    config['top_k'] = args.top_k_neighbors
    
    config['data_path'] = args.data_path
    config['dataset'] = args.dataset
    
    GPU = torch.cuda.is_available()
    device = torch.device('cuda' if GPU else "cpu")
    config['device'] = device
    
    return config, device

def log_results_to_csv(args, cold_before, cold_after, warm_before, warm_after, all_before,all_after, filename="tuning_results.csv"):
    """실험 결과를 CSV 파일에 누적 저장"""
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 헤더 작성 (파일이 처음 생성될 때만)
        if not file_exists:
            headers = ['Dataset', 'Layer', 'LR', 'Decay', 'Dropout', 'KeepProb', 'TopK_Neighbor', 
                       'Cold_NDCG_Before', 'Cold_NDCG_After', 'Improvement(%)', 'Warm_NDCG_Before','Warm_NDCG_After', 'All_NDCG_Before', 'All_NDCG_After']
            writer.writerow(headers)
        
        improvement = (cold_after - cold_before) * 100
        row = [args.dataset, args.layer, args.lr, args.decay, args.dropout, args.keepprob, args.top_k_neighbors,
               f"{cold_before:.5f}", f"{cold_after:.5f}", f"{improvement:.2f}", f"{warm_before:.5f}", f"{warm_after:.5f}", f"{all_before:.5f}",f"{all_after:.5f}"]
        writer.writerow(row)
    
    print(f"\n[Logger] Result saved to {filename}")

def simulate_cold_start_scenario(dataset, seed, cold_ratio=0.2):
    print(f"[{time.strftime('%H:%M:%S')}] Applying Cold Start Masking (Ratio: {cold_ratio})...")
    num_users = dataset.n_users
    num_cold = int(num_users * cold_ratio)
    
    np.random.seed(seed)
    perm = np.random.permutation(num_users)
    cold_users = perm[:num_cold]
    warm_users = perm[num_cold:]
    
    mask = np.isin(dataset.trainUser, cold_users, invert=True)
    dataset.trainUser = dataset.trainUser[mask]
    dataset.trainItem = dataset.trainItem[mask]
    dataset.m_train = len(dataset.trainUser)
    
    for uid in cold_users:
        # 빈 정수형 배열로 교체하여 샘플링 시 참조할 아이템이 없도록 만듦
        dataset.allPos[uid] = np.array([], dtype=int)
    
    dataset.UserItemNet = csr_matrix(
        (np.ones(len(dataset.trainUser)), (dataset.trainUser, dataset.trainItem)),
        shape=(dataset.n_user, dataset.m_item)
    )
    dataset.Graph = None 
    
    return cold_users, warm_users

# ============================================================================
# 3. Main Execution
# ============================================================================
if __name__ == '__main__':
    args = parse_args()
    config, device = get_config(args)
    set_seed(args.seed)
    
    # World 설정 동기화
    world.config = config
    world.dataset = args.dataset
    world.device = device
    world.TRAIN_epochs = args.epochs
    world.topks = eval(args.topks)
    world.tensorboard = args.tensorboard
    
    # 실험 고유 ID 생성 (파라미터 조합)
    run_id = f"L{args.layer}_lr{args.lr}_dec{args.decay}_drop{args.dropout}_kp{args.keepprob}_K{args.top_k_neighbors}"
    print(f">> RUN ID: {run_id}")
    print(f">> DEVICE: {device}")
    
    # 1. Dataset Load
    dataset_path = join(args.data_path, args.dataset)
    dataset = InterCDRDataset(config, path=dataset_path)
    
    # 2. Cold Start Masking
    cold_users, warm_users = simulate_cold_start_scenario(dataset, args.seed, cold_ratio=0.2)
    
    # 3. Model Init
    Recmodel = Ours(config, dataset, cold_users)
    Recmodel = Recmodel.to(device)
    bpr = utils.BPRLoss(Recmodel, config)
    
    # 4. Checkpoint & Tensorboard
    if not os.path.exists(args.path): os.makedirs(args.path, exist_ok=True)
    weight_file = join(args.path, f"{args.dataset}_{run_id}.pth.tar")
    
    w = None
    if args.tensorboard:
        if not os.path.exists(args.board_path): os.makedirs(args.board_path, exist_ok=True)
        w = SummaryWriter(join(args.board_path, f"{args.dataset}_{run_id}"))

    # 5. Training Loop
    best_ndcg20 = -1.0
    best_epoch = -1
    no_improve_cnt = 0
    patience = 5
    best_state_dict = None
    
    try:
        ndcg20_idx = world.topks.index(20)
    except:
        ndcg20_idx = -1

    print(f"[{time.strftime('%H:%M:%S')}] Start Training...")
    try:
        for epoch in range(args.epochs):
            output_info = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=1, w=w)
            print(f'EPOCH[{epoch+1}/{args.epochs}] {output_info}')
            
            if (epoch + 1) % 5 == 0:
                cprint("[VALIDATION]")
                val_results = Procedure.Test(dataset, Recmodel, epoch, w, config['multicore'], 
                                             eval_dict=dataset.validDict, split_name="valid", num_neg = args.num_neg)
                
                current_ndcg20 = float(val_results['ndcg'][ndcg20_idx])
                
                if current_ndcg20 > best_ndcg20:
                    best_ndcg20 = current_ndcg20
                    best_epoch = epoch
                    no_improve_cnt = 0
                    best_state_dict = Recmodel.state_dict()
                    torch.save(Recmodel.state_dict(), weight_file)
                    cprint(f"[NEW BEST] epoch {epoch+1}, NDCG@20={best_ndcg20:.6f}")
                else:
                    no_improve_cnt += 1
                    cprint(f"[NO IMPROVE] {no_improve_cnt}/{patience}")
                    if no_improve_cnt >= patience:
                        cprint(f"[EARLY STOP] No improvement for {patience} validations")
                        break
    finally:
        if w: w.close()

    # 6. Evaluation Setup
    if best_state_dict is not None:
        Recmodel.load_state_dict(best_state_dict)
        cprint(f"[LOAD BEST] epoch {best_epoch+1}")

    cold_test_dict = {u: dataset.testDict[u] for u in cold_users if u in dataset.testDict}
    warm_test_dict = {u: dataset.testDict[u] for u in warm_users if u in dataset.testDict}
    
    def run_eval(eval_dict):
        if len(eval_dict) == 0: return 0.0
        res = Procedure.Test(dataset, Recmodel, args.epochs, None, config['multicore'], eval_dict=eval_dict, split_name="test", num_neg = args.num_neg)
        try: return float(res['ndcg'][ndcg20_idx])
        except: return 0.0

    # 7. Before Injection
    cprint("\n[EVAL - BEFORE]")
    ndcg_all_before = run_eval(dataset.testDict)
    ndcg_cold_before = run_eval(cold_test_dict)
    ndcg_warm_before = run_eval(warm_test_dict)
    print(f"   > All : {ndcg_all_before:.5f}")
    print(f"   > Cold: {ndcg_cold_before:.5f}")
    print(f"   > Warm: {ndcg_warm_before:.5f}")

    # 8. Injection
    cprint("\n[INJECTION]")
    Recmodel.inject_cold_embeddings(cold_users, warm_users)

    # 9. After Injection
    cprint("\n[EVAL - AFTER]")
    ndcg_all_after = run_eval(dataset.testDict)
    ndcg_cold_after = run_eval(cold_test_dict)
    ndcg_warm_after = run_eval(warm_test_dict)

    print(f"   > All : {ndcg_all_after:.5f}")
    print(f"   > Cold: {ndcg_cold_after:.5f}")
    print(f"   > Warm: {ndcg_warm_after:.5f}")
    
    print("\n" + "="*30)
    print(f"   Improvement: {(ndcg_cold_after - ndcg_cold_before)*100:.2f}%")
    print("="*30)
    
    # 10. Save Result to CSV
    log_results_to_csv(args, ndcg_all_before, ndcg_cold_before, ndcg_warm_before, ndcg_cold_after, ndcg_warm_after, ndcg_all_after)