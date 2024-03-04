import numpy as np
import math
import torch 

def RMSE(preds, truth):
    return np.sqrt(np.mean(np.square(preds-truth)))


def getIdcg(length):
    idcg = 0.0
    for i in range(length):
        idcg = idcg + math.log(2) / math.log(i + 2)
    return idcg

def getDcg(value):
    dcg = math.log(2) / math.log(value + 2)
    return dcg

def getHr(value):
    hit = 1.0
    return hit

def getHrNdcg(vali_preds, vali_user_list, vali_u_item_dict, vali_u_item_length_dict, ddp_flag=None, rank=None, debugger=None):
    
    topK = 10

    tmp_hr_list, tmp_ndcg_list = [], []

    #print(f'rank:{rank}')

    for idx, u in enumerate(vali_user_list):
        #import pdb; pdb.set_trace()
        prediction_list = vali_preds[100*idx:100*idx+100]
        # real item ranking for each user

        '''
        if rank == 1:
            debugger.set_trace()
        else:
            torch.distributed.barrier()
        #'''
        
        if rank != None:
            real_item_list = vali_u_item_dict[rank][u][:vali_u_item_length_dict[u]]
        else:
            real_item_list = vali_u_item_dict[u][:vali_u_item_length_dict[u]]
        #import pdb; pdb.set_trace()
        target_length = min(topK, len(real_item_list))
        
        item_predction_list = []
        for xx_idx, item in enumerate(real_item_list):
            item_predction_list.append(prediction_list[xx_idx])
        
        sort_index = np.argsort(prediction_list.tolist())
        sort_index = sort_index[::-1]
        
        ranking_order_list = []
        relative_order_list = []
        for rel_id, item in enumerate(real_item_list):
            relative_order_list.append(rel_id)
            ranking_order_list.append(sort_index[rel_id])
        
        positive_length = len(real_item_list)
        
        user_hr_list = []
        user_ndcg_list = []
        hits_num = 0
                    
        for current_index in range(topK):
            rel_index = sort_index[current_index]
            if rel_index < positive_length:
                hits_num += 1
                user_hr_list.append(getHr(idx))
                user_ndcg_list.append(getDcg(current_index))

        idcg = getIdcg(vali_u_item_length_dict[u])
        
        tmp_hr = np.sum(user_hr_list) / target_length
        tmp_hr_list.append(tmp_hr)
            
        tmp_ndcg = np.sum(user_ndcg_list) / idcg
        tmp_ndcg_list.append(tmp_ndcg)

    if ddp_flag != None:
        return tmp_hr_list, tmp_ndcg_list
    else:
        return np.mean(tmp_hr_list), np.mean(tmp_ndcg_list)