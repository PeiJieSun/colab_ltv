import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.loss import BPRLoss, EmbLoss
from numpy.random import RandomState

class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss
        
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dataset, n_factors=20):
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.n_factors = n_factors
        self.random_state = RandomState(1)
        
        # define layers and loss
        
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.n_factors)
        self.user_embedding.weight.data = torch.from_numpy(0.1 * self.random_state.rand(self.n_users, self.n_factors)).float()
        
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.n_factors)
        self.item_embedding.weight.data = torch.from_numpy(0.1 * self.random_state.rand(self.n_items, self.n_factors)).float()
        
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat(dataset).to(self.device)

        # parameters initialization
        #self.apply(xavier_uniform_initialization)
        #self.other_parameter_name = ["restore_user_e", "restore_item_e"]
    
    def get_norm_adj_mat(self, dataset, adj_type = 'pre'):
        r"""Get the normalized interaction matrix of users and items via scipy.sparse.csr_matrix

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """

        user_np, item_np = dataset['user_id'], dataset['item_id']
        ratings = dataset['ratings']
        #ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.n_users + self.n_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
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
    
    # ... rest of the class methods remain the same ...
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, users_index, items_index):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )

        user_h1 = user_all_embeddings[users_index]
        item_h1 = item_all_embeddings[items_index]

        R_h = (user_h1 * item_h1).sum(1)
        return R_h

    def predict(self, users_index, items_index):
        preds = self.forward(users_index, items_index)
        return preds
