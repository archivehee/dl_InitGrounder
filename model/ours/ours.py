"""
Ours Model Implementation based on LightGCN
Goal: Online User Cold-start Recommendation with user embedding injection
"""
import torch
from torch import nn
import numpy as np
import pickle
import os
import torch.nn.functional as F
from model import LightGCN 

class Ours(LightGCN):
    def __init__(self, config, dataset, cold_users):
        super(Ours, self).__init__(config, dataset)
        
        # [Config]
        self.device = config['device']
        self.top_k = config.get('top_k', 10)
        self.cold_users = cold_users
        
        # [Paths]
        self.path_config = {
            'src_profile': config['src_profile'],
            'tgt_profile': config['tgt_profile'],
            'src_lgn': config['src_lgn']
        }
        
        # [Load External Embeddings]
        self.src_prof_emb, self.tgt_prof_emb, self.src_lgn_emb = self._load_external_embeddings()
        
        # [Dimension Projection Layer]
        # Profile Embedding -> LightGCN Latent dim (e.g., 768 or 64 -> 64)
        #input_dim = self.tgt_prof_emb.shape[1]
        #self.emb_projector = nn.Linear(input_dim * 2, self.latent_dim).to(self.device)
        
        # [Initial Setup]
        self._init_warm_weights()

    def _load_pickle(self, path):
        """
        [수정됨] Robust Pickle Loader for OrderedDict
        데이터 구조가 {'user_emb': array, ...} 형태인 경우를 처리합니다.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                data = pickle.load(f, encoding='latin1')
        
        # 1. Dictionary (OrderedDict 포함)인 경우 처리
        if isinstance(data, dict):
            # Case A: User Profile Embedding (제공해주신 데이터 구조)
            if 'user_emb' in data:
                print(f"[Ours] Found 'user_emb' key in {os.path.basename(path)}")
                data = data['user_emb']
                
            # Case B: PyTorch State Dict 형태
            elif 'weight' in data:
                data = data['weight']
            elif 'embedding_user.weight' in data:
                data = data['embedding_user.weight']
                
            # Case C: 기타 딕셔너리 구조일 경우 (Key 값으로 정렬하여 리스트화 시도)
            elif not isinstance(data, (torch.Tensor, np.ndarray)):
                print(f"[Ours] Detected generic dictionary in {os.path.basename(path)}. Sorting keys...")
                try:
                    sorted_keys = sorted(data.keys())
                    data = [data[k] for k in sorted_keys]
                    data = np.array(data)
                except Exception as e:
                    # 데이터가 텐서 변환이 불가능한 메타데이터일 수 있으므로 Warning 출력
                    print(f"[Warning] Could not convert dictionary values to array directly: {e}")

        # 2. Numpy Array -> Tensor 변환
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            
        # 3. List -> Tensor 변환
        elif isinstance(data, list):
            data = torch.tensor(data)

        # 4. 최종 Tensor 확인 및 Device 이동
        if isinstance(data, torch.Tensor):
            return data.float().to(self.device)
        else:
            raise RuntimeError(f"Failed to extract tensor from {path}. Data type: {type(data)}")

    def _load_external_embeddings(self):
        """Load Profile and Pre-trained LightGCN embeddings"""
        print(f"[Ours] Loading external embeddings...")
        src_prof = self._load_pickle(self.path_config['src_profile'])
        tgt_prof = self._load_pickle(self.path_config['tgt_profile'])
        src_lgn = self._load_pickle(self.path_config['src_lgn'])
        
        # User 수 검증 및 조정
        if src_prof.shape[0] != self.num_users:
            print(f"[Warning] src_prof shape {src_prof.shape} != num_users {self.num_users}.")
            if src_prof.shape[0] > self.num_users:
                print(" -> Slicing embeddings to match num_users.")
                src_prof = src_prof[:self.num_users]
                tgt_prof = tgt_prof[:self.num_users]
                src_lgn = src_lgn[:self.num_users]
            else:
                 # 임베딩이 유저 수보다 적은 경우 치명적 에러
                 raise ValueError(f"Embedding size ({src_prof.shape[0]}) is smaller than num_users ({self.num_users})")

        return src_prof, tgt_prof, src_lgn

    def _init_warm_weights(self):
        """
        Target Domain Warm User Embedding Initialization
        Fix: CPU/GPU Device mismatch during initialization
        """
        print("[Ours] Initializing Warm User Embeddings (Partial Copy)...")
        
        with torch.no_grad():
            # 1. 계산은 GPU에서 수행 (빠름)
            # (N, 64) - 현재 GPU에 있음
            smart_init_emb = (self.tgt_prof_emb + self.src_lgn_emb) / 2
            
            # 2. 대입할 타겟(모델 파라미터)의 장치 확인 (보통 __init__ 시점엔 CPU)
            target_device = self.embedding_user.weight.device
            
            # 3. Warm User 식별 및 대입
            if self.cold_users is not None:
                # 인덱스 계산은 GPU에서 수행
                all_indices = torch.arange(self.num_users).to(self.device)
                cold_indices_tensor = torch.LongTensor(self.cold_users).to(self.device)
                
                is_cold = torch.isin(all_indices, cold_indices_tensor)
                warm_indices = all_indices[~is_cold]
                
                print(f"   > Applying Init to {len(warm_indices)} Warm Users only. (Cold users remain Random)")
                
                # [핵심 수정] 대입할 때 인덱스와 데이터를 target_device(CPU)로 이동
                self.embedding_user.weight.data[warm_indices.to(target_device)] = \
                    smart_init_emb[warm_indices].to(target_device)
                
            else:
                print("   > No cold_users provided. Initializing ALL users.")
                # 전체 복사 시에도 장치 이동 필요
                self.embedding_user.weight.data.copy_(smart_init_emb.to(target_device))
        
        print("[Ours] User embeddings initialized.")

    def get_similar_users(self, target_emb, source_embs, k=10, candidate_indices=None):
        """Calculate Cosine Similarity and return Top-K indices"""
        target_emb_norm = F.normalize(target_emb, p=2, dim=1)
        source_embs_norm = F.normalize(source_embs, p=2, dim=1)
        
        # Similarity Matrix (All x All)
        sim_matrix = torch.mm(target_emb_norm, source_embs_norm.t())
        
        # [Masking Logic]
        if candidate_indices is not None:
            # 1. 전체를 -inf로 채운 마스크 생성
            mask = torch.full_like(sim_matrix, float('-inf'))
            
            # 2. Candidate(Warm User) 열만 0으로 설정 (유사도 값 유지)
            # sim_matrix의 shape은 (N, N)이므로, 열(Column) 기준으로 마스킹해야 함
            mask[:, candidate_indices] = 0
            
            # 3. 마스크 적용 (후보가 아닌 유저는 -inf가 되어 topk에서 탈락)
            sim_matrix = sim_matrix + mask

        # Top K
        # 자기 자신(Cold)은 Warm User 후보에 없으므로 k+1을 할 필요 없음
        _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        
        return topk_indices
    def inject_cold_embeddings(self, cold_user_indices, warm_user_indices):
        """Cold User Embedding Injection"""
        print(f"[Ours] Injecting embeddings for {len(cold_user_indices)} Cold Users...")
        
        cold_user_indices = torch.LongTensor(cold_user_indices).to(self.device)
        current_user_emb_weight = self.embedding_user.weight.data
        
        # 1. Sim Users (Profile & LightGCN)
        sim_idx_prof = self.get_similar_users(self.src_prof_emb, self.src_prof_emb, k=self.top_k, candidate_indices=warm_user_indices)
        sim_idx_lgn = self.get_similar_users(self.src_lgn_emb, self.src_lgn_emb, k=self.top_k, candidate_indices=warm_user_indices)
        
        with torch.no_grad():
            for i, uid in enumerate(cold_user_indices):
                # Retrieve Neighbors (자기 자신 제외 로직을 위해 k+1개 뽑았다고 가정)
                # 단순화를 위해 상위 K개 사용 (자기 자신이 포함될 수도 있음 - Cross domain이라 괜찮음)
                neighbors_prof = sim_idx_prof[uid]
                neighbors_lgn = sim_idx_lgn[uid]
                
                # Neighbor Embeddings 가져오기 (학습된 Target Domain Weight 기준)
        
                # Neighbor's Profile 
                n_prof_tgt_prof = self.tgt_prof_emb[neighbors_prof] # (K, 64)
                n_lgn_tgt_prof = self.tgt_prof_emb[neighbors_lgn] # (K, 64)
                
                # Neighbor's Learned Embedding
                n_prof_tgt_lgn = current_user_emb_weight[neighbors_prof] # (K, 64)
                n_lgn_tgt_lgn = current_user_emb_weight[neighbors_lgn] # (K, 64)
                
                feat_prof_group = (n_prof_tgt_prof + n_prof_tgt_lgn) / 2 # (K, 64)
                feat_lgn_group = (n_lgn_tgt_prof + n_lgn_tgt_lgn) / 2 # (K, 64)
                # Combine Logic [Source 37]
                #feat_prof = torch.cat([n_prof_tgt_prof, n_prof_tgt_lgn])  # 128
                #feat_lgn = torch.cat([n_lgn_tgt_prof, n_lgn_tgt_lgn]) # 128
                mean_feat_prof = torch.mean(feat_prof_group, dim=0) # 64
                mean_feat_lgn = torch.mean(feat_lgn_group, dim=0) # 64
                
                # 전체 평균
                combined_emb = (mean_feat_prof + mean_feat_lgn) / 2 # 64
                
                # Update Cold User Embedding
                self.embedding_user.weight.data[uid] = combined_emb #64
                
        print("[Ours] Injection Completed.")