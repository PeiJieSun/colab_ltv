import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torchfm.layer import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding

class WideAndDeep(nn.Module):
    """
    A pytorch implementation of wide and deep learning.

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, n_factors, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, n_factors)
        self.embed_output_dim = len(field_dims) * n_factors
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, users_feat, items_feat):
        x = torch.cat([users_feat, items_feat], dim=1)
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))

    def predict(self, users_feat, items_feat):
        return self.forward(users_feat, items_feat)
