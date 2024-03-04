import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from .fm_layer import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear

class NFM(nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.

    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, config):
        super().__init__()

        n_factors = config['n_factors']
        mlp_dims = config['mlp_dims']
        dropouts = config['dropouts']

        field_dims = [config['n_users'], config['n_items']]

        self.embedding = FeaturesEmbedding(field_dims, n_factors)
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(n_factors),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(n_factors, mlp_dims, dropouts[1])
        
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, users_feat, items_feat):
        x = torch.cat([users_feat.view(-1, 1), items_feat.view(-1, 1)], dim=1)
        cross_term = self.fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term)
        return x.squeeze(1)

    def predict(self, users_feat, items_feat):
        return self.forward(users_feat, items_feat)
