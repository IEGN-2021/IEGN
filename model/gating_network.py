import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class IEGN(nn.Module):
    def __init__(self, num_users, num_items, model_args, device):
        super(IEGN, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.args = model_args

        # init args
        self.L = self.args.L
        self.T = self.args.T
        self.dims = self.args.d

        # add parameters
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.learning_rate = self.args.learning_rate
        self.l2 = self.args.l2
        self.drop_ratio = self.args.drop

        activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': F.tanh, 'sigm': F.sigmoid}



        # graph
        self.weight_size = eval(self.args.layer_size)
        self.n_layers = len(self.weight_size)
        self.adj_matrix = self.args.adj_matrix
        self.node_dropout = 0.1
        self.mess_dropout = self.args.mess_dropout
        self.n_fold = 100

        """
        Init the weight of user-item.
         """
        self.embedding_dict, self.weight_dict = self._init_weights()

        """
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_matrix).to(device)

        # activation function
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.L)]

        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, self.dims)) for i in lengths])


        # fully-connected layer
        self.fc1_dim_h = self.n_h * len(lengths)
        self.fc1 = nn.Linear(self.fc1_dim_h, self.dims)
        self.W2 = nn.Embedding(self.num_items, 2*self.dims)
        self.b2 = nn.Embedding(self.num_items, 1)
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.time_embeddings = nn.Embedding(2*self.num_items, self.dims).to(device)
        self.time_embeddings.weight.data.normal_(0, 1.0 / self.time_embeddings.embedding_dim)

        self.instance_gate_item = Variable(torch.zeros(self.dims, self.dims).type(torch.FloatTensor), requires_grad=True).to(device)#(50,1)
        self.instance_gate_user = Variable(torch.zeros(self.dims, self.dims).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)

        self.feature_gate_item = nn.Linear(self.dims, self.dims).to(device)

        self.feature_gate_user = nn.Linear(self.dims, self.dims).to(device)

        self.cache_x = None

        self.instance_item = Variable(torch.zeros(2*self.dims,self.dims).type(torch.FloatTensor),
                                           requires_grad=True).to(device)
        self.instance_item = torch.nn.init.xavier_uniform_(self.instance_item)


        self.instance_user = Variable(torch.zeros(self.dims, self.dims).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_user = torch.nn.init.xavier_uniform_(self.instance_user)

        self.instance_b = Variable(torch.zeros(1, self.dims).type(torch.FloatTensor), requires_grad=True).to(
        device)
        self.instance_b = torch.nn.init.xavier_uniform_(self.instance_b)

    def forward(self, item_seq, user_ids, items_to_predict, batch_times, for_pred=False):

        A_hat = self.sparse_dropout(self.sparse_norm_adj, self.node_dropout,
                                    self.sparse_norm_adj._nnz())
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            sum_embeddings = torch.matmul(ego_embeddings, self.weight_dict['W_gc_%d' % k])
            # add fusion gated
            user_embeddings_fusion = torch.matmul(ego_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                     + self.weight_dict['b_bi_%d' % k]

            fusion_weight = torch.sigmoid(user_embeddings_fusion + sum_embeddings)
            bi_embeddings = torch.mul(fusion_weight, side_embeddings)
            ego_embeddings = bi_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)

        light_out = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(light_out, [self.num_users, self.num_items])


        user_emb = u_g_embeddings[user_ids].squeeze(1)
        time_embs = self.time_embeddings(batch_times)

        user_time_features = time_embs + self.feature_gate_user(user_emb).unsqueeze(1)  # [10,5,50]#torch.sigmoid()
        # avg-pooling

        item_embeddings = i_g_embeddings[item_seq]  #
        item_embeddings_sque = item_embeddings.unsqueeze(1)
        item_time_features = user_time_features.unsqueeze(1) + item_embeddings_sque  #

        # Convolutional Layers
        out, out_h = None, None

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:

                conv_out = self.ac_conv(conv(item_time_features).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)
        # Fully-connected Layers
        out = out_h
        # apply dropout

        out = self.dropout(out)

        # fully-connected layer

        z = self.ac_fc(self.fc1(out))

        x = torch.cat([z, user_emb], 1)


        w222 = i_g_embeddings[items_to_predict]

        b2 = self.b2(items_to_predict)
        if not for_pred:
            w22 = torch.matmul(self.instance_item, torch.transpose(w222, 2, 1))

            w2 = torch.transpose(w22, 2, 1)
            results = []
            for i in range(items_to_predict.size(1)):
                w2i = w2[:, i, :]
                b2i = b2[:, i, 0]
                result = (x * w2i).sum(1) + b2i
                results.append(result)
            res = torch.stack(results, 1)
        else:
            w2 = torch.matmul(w222, torch.transpose(self.instance_item,1,0))
            w2 = w2.squeeze()
            b2 = b2.squeeze()

            res = x.mm(w2.t()) + b2

        return res


    def _init_weights(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.num_users,self.dims))),

            'item_emb': nn.Parameter(initializer(torch.empty(self.num_items, self.dims)))

        })


        weight_dict = nn.ParameterDict()
        layers = [self.dims] + self.weight_size
        for k in range(self.n_layers):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                                    layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                                    layers[k + 1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(torch.empty(1, layers[k + 1])))})

        return embedding_dict, weight_dict
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.uint8)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))





