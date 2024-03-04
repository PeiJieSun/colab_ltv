import torch
import torch.nn as nn
from numpy.random import RandomState

class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu"):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.activation.lower() == "relu":
                mlp_modules.append(nn.ReLU())
            elif self.activation.lower() == "sigmoid":
                mlp_modules.append(nn.Sigmoid())
            elif self.activation.lower() == "tanh":
                mlp_modules.append(nn.Tanh())
            elif self.activation.lower() == "leakyrelu":
                mlp_modules.append(nn.LeakyReLU())

        self.mlp_layers = nn.Sequential(*mlp_modules)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20, layers=[20, 10], dropout=0.0, is_sparse=False, mf_train=True, mlp_train=True):
        super(NeuMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.mf_train = mf_train
        self.mlp_train = mlp_train
        self.random_state = RandomState(1)

        self.user_mf_embeddings = nn.Embedding(n_users, n_factors, sparse=is_sparse)
        self.user_mf_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, n_factors)).float()

        self.item_mf_embeddings = nn.Embedding(n_items, n_factors, sparse=is_sparse)
        self.item_mf_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_items, n_factors)).float()

        self.user_mlp_embeddings = nn.Embedding(n_users, layers[0], sparse=is_sparse)
        self.user_mlp_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, layers[0])).float()

        self.item_mlp_embeddings = nn.Embedding(n_items, layers[0], sparse=is_sparse)
        self.item_mlp_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_items, layers[0])).float()

        self.mlp = MLPLayers(layers, dropout)

        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(n_factors + layers[-1], 1)
        elif self.mf_train:
            self.predict_layer = nn.Linear(n_factors, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(layers[-1], 1)

    def forward(self, users_index, items_index):
        if self.mf_train:
            user_mf_e = self.user_mf_embeddings(users_index)
            item_mf_e = self.item_mf_embeddings(items_index)
            mf_output = torch.mul(user_mf_e, item_mf_e)
        
        if self.mlp_train:
            user_mlp_e = self.user_mlp_embeddings(users_index)
            item_mlp_e = self.item_mlp_embeddings(items_index)
            mlp_output = self.mlp(torch.cat((user_mlp_e, item_mlp_e), dim=-1))

        if self.mf_train and self.mlp_train:
            output = torch.cat((mf_output, mlp_output), dim=-1)
        elif self.mf_train:
            output = mf_output
        elif self.mlp_train:
            output = mlp_output

        prediction = self.predict_layer(output)
        
        return prediction