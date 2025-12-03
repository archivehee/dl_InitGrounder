"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
import json
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Base Loader kept for backward compatibility (not used in inter_cdr setting).
    """

    def __init__(self, config=world.config, path="../data/gowalla"):
        raise NotImplementedError("Use InterCDRDataset for this project.")

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems


def _build_norm_adj_matrix(n_users, m_items, user_item_net):
    start = time()
    adj_mat = sp.dok_matrix(
        (n_users + m_items, n_users + m_items),
        dtype=np.float32
    )
    adj_mat = adj_mat.tolil()
    R = user_item_net.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()

    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    end = time()
    print(f"costing {end - start}s, generated norm_mat...")
    return norm_adj


class InterCDRDataset(BasicDataset):
    """
    Dataset for InitGrounder amazon/inter_cdr domains.
    Expects train.txt and test.txt with lines: `user item` (tab or space separated).
    """

    def __init__(self, config=world.config, path=None):
        if path is None:
            raise ValueError("path must be provided for InterCDRDataset")
        cprint(f'loading inter_cdr [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        self.path = path
        train_file = join(path, 'train.txt')
        valid_file = join(path, 'valid.txt')
        test_file = join(path, 'test.txt')

        trainUser, trainItem = [], []
        with open(train_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace('\t', ' ').split()
                if len(parts) < 2:
                    continue
                uid = int(parts[0])
                iid = int(parts[1])
                trainUser.append(uid)
                trainItem.append(iid)
                if uid > self.n_user:
                    self.n_user = uid
                if iid > self.m_item:
                    self.m_item = iid
        self.trainUser = np.array(trainUser, dtype=np.int64)
        self.trainItem = np.array(trainItem, dtype=np.int64)
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.traindataSize = len(self.trainUser)

        # valid interactions
        validUser, validItem = [], []
        if os.path.exists(valid_file):
            with open(valid_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.replace('\t', ' ').split()
                    if len(parts) < 2:
                        continue
                    uid = int(parts[0])
                    iid = int(parts[1])
                    validUser.append(uid)
                    validItem.append(iid)
                    if uid > self.n_user:
                        self.n_user = uid
                    if iid > self.m_item:
                        self.m_item = iid
        self.validUser = np.array(validUser, dtype=np.int64) if len(validUser) > 0 else np.array([], dtype=np.int64)
        self.validItem = np.array(validItem, dtype=np.int64) if len(validItem) > 0 else np.array([], dtype=np.int64)

        # test interactions
        testUser, testItem = [], []
        with open(test_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace('\t', ' ').split()
                if len(parts) < 2:
                    continue
                uid = int(parts[0])
                iid = int(parts[1])
                testUser.append(uid)
                testItem.append(iid)
                if uid > self.n_user:
                    self.n_user = uid
                if iid > self.m_item:
                    self.m_item = iid
        self.testUser = np.array(testUser, dtype=np.int64)
        self.testItem = np.array(testItem, dtype=np.int64)
        self.testUniqueUsers = np.unique(self.testUser)
        self.testDataSize = len(self.testUser)

        self.m_item += 1
        self.n_user += 1

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__validDict = self.__build_dict(self.validUser, self.validItem)
        self.__testDict = self.__build_dict(self.testUser, self.testItem)
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            mask_active = getattr(self, "mask_active", False)
            pre_adj_path = self.path + '/s_pre_adj_mat.npz'
            norm_adj = None
            if not mask_active:
                try:
                    pre_adj_mat = sp.load_npz(pre_adj_path)
                    print("successfully loaded...")
                    norm_adj = pre_adj_mat
                except Exception as e:
                    print(f"failed to load pre_adj_mat ({e}), regenerating...")
            if norm_adj is None:
                print("generating adjacency matrix")
                norm_adj = _build_norm_adj_matrix(self.n_users, self.m_items, self.UserItemNet)
                if not mask_active:
                    sp.save_npz(pre_adj_path, norm_adj)
                else:
                    print("mask active - skip saving adjacency cache")

            if self.split is True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_dict(self, users_arr, items_arr):
        """
        return:
            dict: {user: [items]}
        """
        data = {}
        for i, item in enumerate(items_arr):
            user = users_arr[i]
            if data.get(user):
                data[user].append(item)
            else:
                data[user] = [item]
        return data

    def refresh_eval_dictionaries(self):
        self.__validDict = self.__build_dict(self.validUser, self.validItem)
        self.__testDict = self.__build_dict(self.testUser, self.testItem)

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


def _load_pairs_file(file_path):
    users, items = [], []
    if not os.path.exists(file_path):
        return np.array(users, dtype=np.int64), np.array(items, dtype=np.int64)
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace('\t', ' ').split()
            if len(parts) < 2:
                continue
            users.append(int(parts[0]))
            items.append(int(parts[1]))
    return np.array(users, dtype=np.int64), np.array(items, dtype=np.int64)


class CrossDomainCDRDataset(BasicDataset):
    """Merged-dataset loader for joint Electronics+Home_and_Kitchen training."""

    def __init__(self, config=world.config, path=None):
        if path is None:
            raise ValueError("path must be provided for CrossDomainCDRDataset")
        cprint(f'loading cross-domain inter_cdr [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.path = path

        manifest_path = join(path, 'manifest.json')
        self.manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                self.manifest = json.load(f)
        self.domain_item_ranges = {}
        for domain, rng in self.manifest.get('domain_item_ranges', {}).items():
            self.domain_item_ranges[domain] = (int(rng['start']), int(rng['end']))

        train_file = join(path, 'train.txt')
        valid_file = join(path, 'valid.txt')
        test_file = join(path, 'test.txt')

        self.trainUser, self.trainItem = _load_pairs_file(train_file)
        self.validUser, self.validItem = _load_pairs_file(valid_file)
        self.testUser, self.testItem = _load_pairs_file(test_file)

        self.trainUniqueUsers = np.unique(self.trainUser) if len(self.trainUser) else np.array([], dtype=np.int64)
        self.traindataSize = len(self.trainUser)
        self.testUniqueUsers = np.unique(self.testUser) if len(self.testUser) else np.array([], dtype=np.int64)
        self.testDataSize = len(self.testUser)

        self.n_user = int(self.manifest.get('users', self._infer_size([self.trainUser, self.validUser, self.testUser])) )
        self.m_item = int(self.manifest.get('items', self._infer_size([self.trainItem, self.validItem, self.testItem])) )
        self.Graph = None

        print(f"{self.traindataSize} interactions for training (cross-domain)")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__validDict = self.__build_dict(self.validUser, self.validItem)
        self.__testDict = self.__build_dict(self.testUser, self.testItem)
        self._domain_valid = self._build_domain_eval_dicts(self.validUser, self.validItem)
        self._domain_test = self._build_domain_eval_dicts(self.testUser, self.testItem)
        print(f"{world.dataset} is ready to go (domains={list(self.domain_item_ranges.keys())})")

    def _infer_size(self, arrays):
        max_idx = -1
        for arr in arrays:
            if arr is not None and len(arr) > 0:
                max_idx = max(max_idx, int(np.max(arr)))
        return max_idx + 1 if max_idx >= 0 else 0

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def domain_valid_dicts(self):
        return self._domain_valid

    @property
    def domain_test_dicts(self):
        return self._domain_test

    def rebuild_domain_eval_dicts(self):
        self._domain_valid = self._build_domain_eval_dicts(self.validUser, self.validItem)
        self._domain_test = self._build_domain_eval_dicts(self.testUser, self.testItem)
        self.__validDict = self.__build_dict(self.validUser, self.validItem)
        self.__testDict = self.__build_dict(self.testUser, self.testItem)

    def _split_A_hat(self, A):
        return InterCDRDataset._split_A_hat(self, A)

    def _convert_sp_mat_to_sp_tensor(self, X):
        return InterCDRDataset._convert_sp_mat_to_sp_tensor(self, X)

    def getSparseGraph(self):
        return InterCDRDataset.getSparseGraph(self)

    def __build_dict(self, users_arr, items_arr):
        data = {}
        for i, item in enumerate(items_arr):
            if i >= len(users_arr):
                break
            user = users_arr[i]
            if data.get(user):
                data[user].append(item)
            else:
                data[user] = [item]
        return data

    def _build_domain_eval_dicts(self, users_arr, items_arr):
        domain_data = {domain: {} for domain in self.domain_item_ranges}
        if len(users_arr) != len(items_arr):
            return domain_data
        for idx, item in enumerate(items_arr):
            domain = self._domain_for_item(item)
            if domain is None:
                continue
            user = users_arr[idx]
            user_dict = domain_data[domain]
            if user_dict.get(user):
                user_dict[user].append(item)
            else:
                user_dict[user] = [item]
        return domain_data

    def _domain_for_item(self, item_idx):
        for domain, (start, end) in self.domain_item_ranges.items():
            if start <= item_idx < end:
                return domain
        return None

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
