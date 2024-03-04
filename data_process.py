#!/usr/bin/env python
# coding: utf-8

# 1. prepare the dataset for model evaluation
# 2. ndcg@K，模型评估，可以参考diffnet的评估，最大的topK
# 3. 计算评分结果

import os
import pickle
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
from collections import defaultdict

def padding_users(vali_data, NUM_ITEMS):
    user_list = vali_data[:, 0]
    item_list = vali_data[:, 1]
    rating_list = vali_data[:, 2]
    
    tmp_u_item_dict = defaultdict(list)
    tmp_u_rating_dict = defaultdict(list)
    
    u_item_dict = defaultdict(list)
    u_item_length_dict = dict()
    
    for idx, u in enumerate(user_list):
        tmp_u_item_dict[u].append(item_list[idx])
        tmp_u_rating_dict[u].append(rating_list[idx])
    
    new_vali_data = []
    
    #import pdb; pdb.set_trace()
    max_length = 0
    for u in tmp_u_item_dict.keys():
        real_item_rating_list = tmp_u_rating_dict[u]
        sort_index = np.argsort(real_item_rating_list)
        sort_index = sort_index[::-1]
        
        for nxy_idx in sort_index:
            u_item_dict[u].append(tmp_u_item_dict[u][nxy_idx])
        
        u_item_length_dict[u] = len(u_item_dict[u])
        
        max_length = max(max_length, u_item_length_dict[u])
        
        final_idx = len(sort_index)
        positive_set = set(u_item_dict[u])
        while final_idx < 100:
            sample_item = np.random.randint(NUM_ITEMS)
            while sample_item in positive_set:
                sample_item = np.random.randint(NUM_ITEMS)
            
            u_item_dict[u].append(sample_item)
            
            positive_set.add(sample_item)
            final_idx = final_idx + 1
            
    return u_item_dict, u_item_length_dict, max_length

def remap(raw_data_path, processed_data_path):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    read = pd.read_csv(raw_data_path, sep='\t', names=names, header=1)

    users = list(set(read['user_id']))
    user_id_index = dict((user_id, index) for user_id, index in zip(users, range(len(users))))
    pickle.dump(user_id_index, open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'wb'))

    items = list(set(read['item_id']))
    item_id_index = dict((item_id, index) for item_id, index in zip(items, range(len(items))))
    pickle.dump(item_id_index, open(os.path.join(processed_data_path, 'item_id_index.pkl'), 'wb'))

    return user_id_index, item_id_index

def standardize_values(raw_data_path, processed_data_path, user_id_index, item_id_index):
    # iterator all users and find each users' max label and min label
    user_rating_dict = defaultdict(list)
    item_rating_dict = defaultdict(list)
    with open(raw_data_path, 'r') as f:
        lines = f.readlines()
        #import pdb; pdb.set_trace()
        for i in range(1, len(lines)):
            line = lines[i].strip().split('\t')
            user_id = int(line[0])
            item_id = int(line[1])
            rating = float(line[2])
            user_rating_dict[user_id].append(rating)
            item_rating_dict[item_id].append(rating)

    avg_user_rating_dict = dict()
    std_user_rating_dict = dict()
    for user in user_rating_dict:
        avg_user_rating_dict[user] = np.mean(user_rating_dict[user])
        std_user_rating_dict[user] = np.std(user_rating_dict[user])

    avg_item_rating_dict = dict()
    std_item_rating_dict = dict()
    for item in item_rating_dict:
        avg_item_rating_dict[item] = np.mean(item_rating_dict[item])
        std_item_rating_dict[item] = np.std(item_rating_dict[item])

    min_rating = 0
    max_rating = 0
    data = []
    with open(raw_data_path, 'r') as f:
        lines = f.readlines()
        count_user = 0
        count_item = 0
        for i in range(1, len(lines)):
            line = lines[i].strip().split('\t')
            user_id = int(line[0])
            item_id = int(line[1])
            rating = float(line[2]) 
            #if std_user_rating_dict[user_id]:
                #import pdb; pdb.set_trace()
            rating_user = (rating - avg_user_rating_dict[user_id]) / (std_user_rating_dict[user_id] + 1e-6)
            rating_item = (rating - avg_item_rating_dict[item_id]) / (std_item_rating_dict[item_id] + 1e-6)
            rating = (rating_user + rating_item) / 2.0 / 6.0
            
            min_rating = min(min_rating, rating)
            max_rating = max(max_rating, rating)

            data.append([user_id_index[user_id], item_id_index[item_id], rating])

    data = np.array(data)
    np.random.shuffle(data)
    np.savetxt(os.path.join(processed_data_path, 'data.txt'), data, fmt='%f')

    return data 

def main(raw_data_path, processed_data_path):
    # choose dataset to process

    user_id_index, item_id_index = remap(raw_data_path, processed_data_path)

    NUM_USERS = max(user_id_index.values()) + 1
    NUM_ITEMS = max(item_id_index.values()) + 1

    data = standardize_values(raw_data_path, processed_data_path, user_id_index, item_id_index)

    # set split ratio and split the data into train, valid, and test files
    ratio = 0.8

    train_data = data[:int(ratio*data.shape[0])]
    pickle.dump(train_data, open('%s/train_data.pkl' % processed_data_path, 'wb')) 

    vali_data = data[int(ratio*data.shape[0]):int((ratio+(1-ratio)/2)*data.shape[0])]
    pickle.dump(vali_data, open('%s/vali_data.pkl' % processed_data_path, 'wb')) 

    vali_u_item_dict, vali_u_item_length_dict, vali_max_length = padding_users(vali_data, NUM_ITEMS)
    pickle.dump(vali_u_item_dict, open('%s/valid_u_item_dict.pkl' % processed_data_path, 'wb'))
    pickle.dump(vali_u_item_length_dict, open('%s/valid_u_item_length_dict.pkl' % processed_data_path, 'wb'))

    test_data = data[int((ratio+(1-ratio)/2)*data.shape[0]):]
    pickle.dump(test_data, open('%s/test_data.pkl' % processed_data_path, 'wb')) 

    test_u_item_dict, test_u_item_length_dict, test_max_length = padding_users(test_data, NUM_ITEMS)
    pickle.dump(test_u_item_dict, open('%s/test_u_item_dict.pkl' % processed_data_path, 'wb'))
    pickle.dump(test_u_item_length_dict, open('%s/test_u_item_length_dict.pkl' % processed_data_path, 'wb'))

# prepare data for lightgcn
def process_for_lightgcn(root_dir, processed_data_path):

    def get_norm_adj_mat(processed_data_path, dataset, adj_type = 'pre'):
        r"""Get the normalized interaction matrix of users and items via scipy.sparse.csr_matrix

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        user_id_index = pickle.load(open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'rb'))
        NUM_USERS = max(user_id_index.values()) + 1
        n_users = NUM_USERS

        item_id_index = pickle.load(open(os.path.join(processed_data_path, 'item_id_index.pkl'), 'rb'))
        NUM_ITEMS = max(item_id_index.values()) + 1
        n_items = NUM_ITEMS

        user_np, item_np = dataset['user_id'], dataset['item_id']
        ratings = dataset['ratings']
        #ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = n_users + n_items
        #import pdb; pdb.set_trace()
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+n_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj
        
        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        adj_matrix = adj_matrix.tocoo()

        index = torch.LongTensor([adj_matrix.row, adj_matrix.col])
        data = torch.FloatTensor(adj_matrix.data)
        SparseL = torch.sparse.FloatTensor(index, data, torch.Size(adj_matrix.shape))
        
        return SparseL

    import numpy as np

    def process_file(file_path):
        user_id = []
        item_id = []
        ratings = []

        with open(file_path, 'r') as f:
            for line in f:
                u, i, r = line.split(' ')
                user_id.append(int(float(u)))
                item_id.append(int(float(i)))
                #ratings.append(r.strip())
                ratings.append(1)

        data_dict = {
            'user_id': np.array(user_id), 
            'item_id': np.array(item_id), 
            'ratings': np.array(ratings)
        }

        return data_dict

    file_path = '%s/data.txt' % root_dir
    data_dict = process_file(file_path)

    SparseGraph = get_norm_adj_mat(processed_data_path, data_dict)
    #denseGraph = SparseGraph.to_dense() 
    torch.save(SparseGraph, "%s/SparseGraph.pth" % root_dir)

if __name__ == "__main__":
    dataset_name = 'steam-v1'
    raw_data_path = os.path.join('/root/autodl-fs/pmf_torch/data/steam-v1/steam.inter')
    processed_data_path = os.path.join('/root/autodl-fs/pmf_torch/processed_data/', dataset_name)
    #main(raw_data_path, processed_data_path)

    root_dir = '/root/autodl-fs/pmf_torch/processed_data/steam-v1/'
    process_for_lightgcn(root_dir, processed_data_path)