import torch
import torch.nn as nn
import torch.nn.functional as F
from mflow.models.basic import GraphLinear, GraphConv, ActNorm, ActNorm2D


# class AffineCoupling(nn.Module):
#     def __init__(self, in_channel, hidden_channels, affine=True):  # filter_size=512,  --> hidden_channels =(512, 512)
#         super(AffineCoupling, self).__init__()
#
#         self.affine = affine
#         self.layers = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         # self.norms_in = nn.ModuleList()
#         last_h = in_channel // 2
#         if affine:
#             vh = tuple(hidden_channels) + (in_channel,)
#         else:
#             vh = tuple(hidden_channels) + (in_channel // 2,)
#
#         for h in vh:
#             self.layers.append(nn.Conv2d(last_h, h, kernel_size=3, padding=1))
#             self.norms.append(nn.BatchNorm2d(h))  #, momentum=0.9 may change norm later, where to use norm? for the residual? or the sum
#             # self.norms.append(ActNorm(in_channel=h, logdet=False)) # similar but not good
#             last_h = h
#
#     def forward(self, input):
#         in_a, in_b = input.chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)
#
#         if self.affine:
#             # log_s, t = self.net(in_a).chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)
#             s, t = self._s_t_function(in_a)
#             out_b = (in_b + t) * s   #  different affine bias , no difference to the log-det # (2,6,32,32) More stable, less error
#             # out_b = in_b * s + t
#             logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)
#
#         else:  # add coupling
#             # net_out = self.net(in_a)
#             _, t = self._s_t_function(in_a)
#             out_b = in_b + t
#             logdet = None
#
#         return torch.cat([in_a, out_b], 1), logdet
#
#     def reverse(self, output):
#         out_a, out_b = output.chunk(2, 1)
#
#         if self.affine:
#             s, t = self._s_t_function(out_a)
#             in_b = out_b / s - t  # More stable, less error   s must not equal to 0!!!
#             # in_b = (out_b - t) / s
#         else:
#             _, t = self._s_t_function(out_a)
#             in_b = out_b - t
#
#         return torch.cat([out_a, in_b], 1)
#
#     def _s_t_function(self, x):
#         h = x
#         for i in range(len(self.layers)-1):
#             h = self.layers[i](h)
#             h = self.norms[i](h)
#             # h = torch.tanh(h)  # tanh may be more stable?
#             h = torch.relu(h)  #
#         h = self.layers[-1](h)
#
#         s = None
#         if self.affine:
#             # residual net for doubling the channel. Do not use residual, unstable
#             log_s, t = h.chunk(2, 1)
#             # s = torch.sigmoid(log_s + 2)  # (2,6,32,32) # s != 0 and t can be arbitrary : Why + 2??? more stable, keep s != 0!!! exp is not stable
#             s = torch.sigmoid(log_s) # works good when actnorm
#             # s = torch.tanh(log_s) # can not use tanh
#             # s = torch.sign(log_s) # lower reverse error if no actnorm, similar results when have actnorm
#
#         else:
#             t = h
#         return s, t


class AffineCoupling(nn.Module):  # delete
    def __init__(self, in_channel, hidden_channels, affine=True, mask_swap=False):  # filter_size=512,  --> hidden_channels =(512, 512)
        super(AffineCoupling, self).__init__()

        self.affine = affine
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mask_swap=mask_swap
        # self.norms_in = nn.ModuleList()
        last_h = in_channel // 2
        if affine:
            vh = tuple(hidden_channels) + (in_channel,)
        else:
            vh = tuple(hidden_channels) + (in_channel // 2,)

        for h in vh:
            self.layers.append(nn.Conv2d(last_h, h, kernel_size=3, padding=1))
            self.norms.append(nn.BatchNorm2d(h))  # , momentum=0.9 may change norm later, where to use norm? for the residual? or the sum
            # self.norms.append(ActNorm(in_channel=h, logdet=False)) # similar but not good
            last_h = h

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)

        if self.mask_swap:
            in_a, in_b = in_b, in_a

        if self.affine:
            # log_s, t = self.net(in_a).chunk(2, 1)  # (2,12,32,32) --> (2,6,32,32), (2,6,32,32)
            s, t = self._s_t_function(in_a)
            out_b = (in_b + t) * s   #  different affine bias , no difference to the log-det # (2,6,32,32) More stable, less error
            # out_b = in_b * s + t
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)
        else:  # add coupling
            # net_out = self.net(in_a)
            _, t = self._s_t_function(in_a)
            out_b = in_b + t
            logdet = None

        if self.mask_swap:
            result = torch.cat([out_b, in_a], 1)
        else:
            result = torch.cat([in_a, out_b], 1)

        return result, logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.mask_swap:
            out_a, out_b = out_b, out_a

        if self.affine:
            s, t = self._s_t_function(out_a)
            in_b = out_b / s - t  # More stable, less error   s must not equal to 0!!!
            # in_b = (out_b - t) / s
        else:
            _, t = self._s_t_function(out_a)
            in_b = out_b - t

        if self.mask_swap:
            result = torch.cat([in_b, out_a], 1)
        else:
            result = torch.cat([out_a, in_b], 1)

        return result

    def _s_t_function(self, x):
        h = x
        for i in range(len(self.layers)-1):
            h = self.layers[i](h)
            h = self.norms[i](h)
            # h = torch.tanh(h)  # tanh may be more stable?
            h = torch.relu(h)  #
        h = self.layers[-1](h)

        s = None
        if self.affine:
            # residual net for doubling the channel. Do not use residual, unstable
            log_s, t = h.chunk(2, 1)
            # s = torch.sigmoid(log_s + 2)  # (2,6,32,32) # s != 0 and t can be arbitrary : Why + 2??? more stable, keep s != 0!!! exp is not stable
            s = torch.sigmoid(log_s)  # works good when actnorm
            # s = torch.tanh(log_s) # can not use tanh
            # s = torch.sign(log_s) # lower reverse error if no actnorm, similar results when have actnorm
        else:
            t = h
        return s, t


class GraphAffineCoupling(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(GraphAffineCoupling, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine

        self.hidden_dim_gnn = hidden_dim_dict['gnn']
        self.hidden_dim_linear = hidden_dim_dict['linear']

        self.net = nn.ModuleList()
        self.norm = nn.ModuleList()
        last_dim = in_dim
        for out_dim in self.hidden_dim_gnn:  # What if use only one gnn???
            self.net.append(GraphConv(last_dim, out_dim))
            self.norm.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            # self.norm.append(ActNorm2D(in_dim=n_node, logdet=False))
            last_dim = out_dim

        self.net_lin = nn.ModuleList()
        self.norm_lin = nn.ModuleList()
        for out_dim in self.hidden_dim_linear:  # What if use only one gnn???
            self.net_lin.append(GraphLinear(last_dim, out_dim))
            self.norm_lin.append(nn.BatchNorm1d(n_node))  # , momentum=0.9 Change norm!!!
            # self.norm_lin.append(ActNorm2D(in_dim=n_node, logdet=False))
            last_dim = out_dim

        if affine:
            self.net_lin.append(GraphLinear(last_dim, in_dim*2))
        else:
            self.net_lin.append(GraphLinear(last_dim, in_dim))

        self.scale = nn.Parameter(torch.zeros(1))  # nn.Parameter(torch.ones(1)) #
        mask = torch.ones(n_node, in_dim)
        mask[masked_row, :] = 0  # masked_row are kept same, and used for _s_t for updating the left rows
        self.register_buffer('mask', mask)

    def forward(self, adj, input):
        masked_x = self.mask * input
        s, t = self._s_t_function(adj, masked_x)  # s must not equal to 0!!!
        if self.affine:
            out = masked_x + (1-self.mask) * (input + t) * s
            # out = masked_x + (1-self.mask) * (input * s + t)
            logdet = torch.sum(torch.log(torch.abs(s)).view(input.shape[0], -1), 1)  # possibly wrong answer
        else:  # add coupling
            out = masked_x + t*(1-self.mask)
            logdet = None
        return out, logdet

    def reverse(self, adj, output):
        masked_y = self.mask * output
        s, t = self._s_t_function(adj, masked_y)
        if self.affine:
            input = masked_y + (1 - self.mask) * (output / s - t)
            # input = masked_x + (1 - self.mask) * ((output-t) / s)
        else:
            input = masked_y + (1 - self.mask) * (output - t)
        return input

    def _s_t_function(self, adj, x):
        # adj: (2,4,9,9)  x: # (2,9,5)
        s = None
        h = x
        for i in range(len(self.net)):
            h = self.net[i](adj, h)  # (2,1,9,hidden_dim)
            h = self.norm[i](h)
            # h = torch.tanh(h)  # tanh may be more stable
            h = torch.relu(h)  # use relu!!!

        for i in range(len(self.net_lin)-1):
            h = self.net_lin[i](h)  # (2,1,9,hidden_dim)
            h = self.norm_lin[i](h)
            # h = torch.tanh(h)
            h = torch.relu(h)

        h = self.net_lin[-1](h)
        # h =h * torch.exp(self.scale*2)

        if self.affine:
            log_s, t = h.chunk(2, dim=-1)
            #  x = sigmoid(log_x+bias): glow code Top 1 choice, keep s away from 0, s!!!!= 0  always safe!!!
            # And get the signal from added noise in the  input
            # s = torch.sigmoid(log_s + 2)
            s = torch.sigmoid(log_s)  # better validity + actnorm

            # s = torch.tanh(log_s)  # Not stable when s =0 for synthesis data, but works well for real data in best case....
            # s = torch.sign(s)

            # s = torch.sign(log_s)

            # s = F.softplus(log_s) # negative nll
            # s = torch.sigmoid(log_s)  # little worse than +2, # *self.scale #!!! # scale leads to nan results
            # s = torch.tanh(log_s+2) # not that good
            # s = torch.relu(log_s) # nan results
            # s = log_s  # nan results
            # s = torch.exp(log_s)  # nan results
        else:
            t = h
        return s, t


def test_AffineCoupling():
    from mflow.models.model import rescale_adj
    torch.manual_seed(0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4

    # x = torch.ones((bs, nodes, ch), dtype=torch.float32)  # 2 for duplicated flow for transforming whole info
    adj = torch.randint(0, 2, (bs, num_edge_type, nodes, nodes), dtype=torch.float32)
    # adj = rescale_adj(adj)

    gc = AffineCoupling(in_channel=4, hidden_channels={512,512}, affine=True)

    out = gc(adj)
    print('adj.shape:', adj.shape)
    # print('out', out.shape)  # (bs, out_ch)
    print(out[0].shape, out[1].shape)

    r = gc.reverse(out[0])
    print(r.shape)
    print(r)
    print('torch.abs(r-adj).mean():', torch.abs(r - adj).mean())


def test_GraphAffineCoupling():
    from mflow.models.model import rescale_adj
    torch.manual_seed(0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4

    # x = torch.ones((bs, nodes, ch), dtype=torch.float32)  # 2 for duplicated flow for transforming whole info
    x = torch.randint(0, 2, (bs, nodes, ch), dtype=torch.float32)
    adj = torch.randint(0, 2, (bs, num_edge_type, nodes, nodes), dtype=torch.float32)
    adj = rescale_adj(adj)

    in_dim = ch  # 5
    hidden_dim_dict = {'gnn': [8, 64], 'linear':[8]}
    gc = GraphAffineCoupling(nodes, in_dim, hidden_dim_dict, masked_row=range(0, nodes, 2), affine=True)
        # (num_nodes=nodes, num_relations=num_edge_type, num_features=ch, mask=mask,
        #                        batch_norm=True,
        #                        num_masked_cols=1, ch_list=[256, 256])

    out = gc(adj, x)
    print('in', x.shape, adj.shape)
    # print('out', out.shape)  # (bs, out_ch)
    print(out[0].shape, out[1].shape)
    print(out)

    r = gc.reverse(adj, out[0])
    print(r)

    print(r.shape)
    print('torch.abs(r-x).mean():', torch.abs(r - x).mean())


if __name__ == '__main__':
    # test_AdditiveAdjCoupling()
    # test_AdditiveNodeFeatureCoupling()
    # test_GraphAffineCoupling()
    test_AffineCoupling()


