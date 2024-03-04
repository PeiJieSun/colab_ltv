# Refer the following link: https://github.com/shenweichen/DeepCTR/blob/master/deepctr/models/autoint.py 
# AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_heads)])

    def forward(self, inputs):
        attention_outputs = [F.softmax(attention_head(inputs)) for attention_head in self.attention_heads]
        output = torch.cat(attention_outputs, dim=-1)
        return output

class AutoInt(nn.Module):
    def __init__(self, field_dims, num_heads, num_layers, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, num_heads)
        self.linear = FeaturesLinear(field_dims)
        self.attention_layers = nn.ModuleList([MultiHeadSelfAttention(num_heads, num_heads) for _ in range(num_layers)])
        self.residual_connections = nn.ModuleList([nn.Linear(num_heads, num_heads) for _ in range(num_layers)])
        self.mlp = MultiLayerPerceptron(num_heads, mlp_dims, dropouts[1])
        
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, users_feat, items_feat):
        x = torch.cat([users_feat, items_feat], dim=1)
        x = self.embedding(x)
        for i in range(self.num_layers):
            attention_output = self.attention_layers[i](x)
            x = F.relu(self.residual_connections[i](attention_output) + x)
        x = self.linear(x) + self.mlp(x)
        return torch.sigmoid(x.squeeze(1))

    def predict(self, users_feat, items_feat):
        return self.forward(users_feat, items_feat)
