from __future__ import print_function

import torch.utils.data
import torch.nn.init as init
import matplotlib.pyplot as plt
from models.pmf import *
from evaluations import *
import pickle
import os
from collections import defaultdict
import time 

def create_instance(module_name, class_name, **kwargs):
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**kwargs)

def train_model(args, config):
    # choose dataset name and load dataset
    dataset = config['dataset']
    #processed_data_path = os.path.join(os.getcwd(), 'processed_data', dataset)
    processed_data_path = config['processed_data_path']
    # set split ratio
    ratio = config['ratio']
    epoches = config['epoches']
    batch_size = config['batch_size']
    weight_decay = config['weight_decay']

    user_id_index = pickle.load(open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'rb'))
    NUM_USERS = max(user_id_index.values()) + 1
    config['n_users'] = NUM_USERS

    item_id_index = pickle.load(open(os.path.join(processed_data_path, 'item_id_index.pkl'), 'rb'))
    NUM_ITEMS = max(item_id_index.values()) + 1
    config['n_items'] = NUM_ITEMS

    data = np.loadtxt(os.path.join(processed_data_path, 'data.txt'), dtype=float)
    print('dataset density:{:f}'.format(len(data)*1.0/(NUM_USERS*NUM_ITEMS)))

    no_cuda=config['no_cuda']
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed=config['seed'])
    if cuda:
        torch.cuda.manual_seed(seed=config['seed'])

    # construct data_loader
    train_data = pickle.load(open('%s/train_data.pkl' % processed_data_path, 'rb'))  # (user, item, rating) 
    train_data_loader = torch.utils.data.DataLoader(torch.from_numpy(train_data), batch_size=batch_size, shuffle=False, num_workers=8)

    #import pdb; pdb.set_trace()
    model = create_instance('models.%s' % config['model'].lower(), config['model'], config=config)

    if cuda:
        #model.cuda()
        model.to(gpu_id)

    # loss function and optimizer
    loss_function = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=config['weight_decay'])

    # train function
    def train(epoch, train_data_loader):
        model.train()
        epoch_loss = 0.0

        optimizer.zero_grad()

        for batch_idx, ele in enumerate(train_data_loader):
            optimizer.zero_grad()

            row = ele[:, 0]
            col = ele[:, 1]
            val = ele[:, 2]

            row = Variable(row.long())
            # TODO: turn this into a collate_fn like the data_loader
            if isinstance(col, list):
                col = tuple(Variable(c.long()) for c in col)
            else:
                col = Variable(col.long())
            val = Variable(val.float())

            if cuda:
                row = row.to(gpu_id)#.cuda()
                col = col.to(gpu_id)#.cuda()
                val = val.to(gpu_id)#cuda()

            preds = model.forward(row, col)
            loss = loss_function(preds, val)
            loss.backward()
            optimizer.step()
            
            #import pdb; pdb.set_trace()
            epoch_loss += loss.tolist()

        epoch_loss /= train_data_loader.dataset.shape[0]

        return epoch_loss

    # training model part
    print('################Training model###################')
    train_loss_list = []
    last_vali_hr_10 = None
    train_rmse_list = []
    print('parameters are: train ratio:{:f},batch_size:{:d}, epoches:{:d}, weight_decay:{:f}'.format(ratio, batch_size, epoches, weight_decay))
    print(model)

    # Load validatation data
    vali_u_item_dict = pickle.load(open('%s/valid_u_item_dict.pkl' % processed_data_path, 'rb'))
    vali_u_item_length_dict = pickle.load(open('%s/valid_u_item_length_dict.pkl' % processed_data_path, 'rb'))

    vali_row = []
    vali_col = []
    vali_user_list = []

    for u in vali_u_item_dict:
        vali_user_list.append(u)
        for item in vali_u_item_dict[u]:
            vali_row.append(u)
            vali_col.append(item)
    
    vali_row = Variable(torch.from_numpy(np.array(vali_row)).long())
    vali_col = Variable(torch.from_numpy(np.array(vali_col)).long())
    if cuda:
        vali_row = vali_row.to(gpu_id)#.cuda()
        vali_col = vali_col.to(gpu_id)#.cuda()

    # Evaluate the performance of the model based on the test dataset 
    print('################Testing trained model###################')
    # load testdata
    test_u_item_dict = pickle.load(open('%s/test_u_item_dict.pkl' % processed_data_path, 'rb'))
    test_u_item_length_dict = pickle.load(open('%s/test_u_item_length_dict.pkl' % processed_data_path, 'rb'))

    test_row = []
    test_col = []
    test_user_list = []

    for u in test_u_item_dict:
        test_user_list.append(u)
        for item in test_u_item_dict[u]:
            test_row.append(u)
            test_col.append(item)

    test_row = Variable(torch.from_numpy(np.array(test_row)).long())
    test_col = Variable(torch.from_numpy(np.array(test_col)).long())
    if cuda:
        test_row = test_row.to(gpu_id)#.cuda()
        test_col = test_col.to(gpu_id)#.cuda()

    best_hr_10, best_ndcg_10 = 0.0, 0.0

    # Start to train the model based on the train data, and implement the early stop based on the valid data
    for epoch in range(1, epoches+1):
        # construct train and vali loss list
        train_epoch_loss = train(epoch, train_data_loader)
        train_loss_list.append(train_epoch_loss)
        
        train_rmse = np.sqrt(train_epoch_loss)
        train_rmse_list.append(train_rmse)
        print(f'{time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime(time.time()))}, training epoch:{epoch}, training rmse:{"%.4f" % train_rmse}')

        vali_preds = model.predict(vali_row, vali_col)
        vali_hr_10, vali_ndcg_10 = getHrNdcg(vali_preds, vali_user_list, vali_u_item_dict, vali_u_item_length_dict)
        print(f'{time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime(time.time()))}, vali hr@10:{"%.4f" % vali_hr_10}, ndcg@10:{"%.4f" % vali_ndcg_10}')

        test_preds = model.predict(test_row, test_col)
        test_hr_10, test_ndcg_10 = getHrNdcg(test_preds, test_user_list, test_u_item_dict, test_u_item_length_dict)
        print(f'{time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime(time.time()))}, vali hr@10:{"%.4f" % test_hr_10}, ndcg@10:{"%.4f" % test_ndcg_10}')

        if test_hr_10 > best_hr_10:
            best_hr_10 = test_hr_10
            best_ndcg_10 = test_ndcg_10

        #'''
        if epoch > 2:
            if last_vali_hr_10 and last_vali_hr_10 > vali_hr_10:
                break
            else:
                last_vali_hr_10 = vali_hr_10
        #'''

    return best_hr_10, best_ndcg_10