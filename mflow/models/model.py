import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mflow.models.hyperparams import Hyperparameters
from mflow.models.glow import Glow, GlowOnGraph


def gaussian_nll(x, mean, ln_var, reduce='sum'):
    """Computes the negative log-likelihood of a Gaussian distribution.

    Given two variable ``mean`` representing :math:`\\mu` and ``ln_var``
    representing :math:`\\log(\\sigma^2)`, this function computes in
    elementwise manner the negative log-likelihood of :math:`x` on a
    Gaussian distribution :math:`N(\\mu, S)`,

    .. math::

        -\\log N(x; \\mu, \\sigma^2) =
        \\log\\left(\\sqrt{(2\\pi)^D |S|}\\right) +
        \\frac{1}{2}(x - \\mu)^\\top S^{-1}(x - \\mu),

    where :math:`D` is a dimension of :math:`x` and :math:`S` is a diagonal
    matrix where :math:`S_{ii} = \\sigma_i^2`.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'sum'``, loss values are summed up.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        mean (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing mean of a Gaussian distribution, :math:`\\mu`.
        ln_var (:class:`~chainer.Variable` or :ref:`ndarray`): A variable
            representing logarithm of variance of a Gaussian distribution,
            :math:`\\log(\\sigma^2)`.
        reduce (str): Reduction option. Its value must be either
            ``'sum'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable representing the negative log-likelihood.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum'``, the output variable holds a scalar value.

    """
    if reduce not in ('sum', 'no'):
        raise ValueError(
            "only 'sum' and 'no' are valid for 'reduce', but '%s' is "
            'given' % reduce)

    x_prec = torch.exp(-ln_var)  # 324
    x_diff = x - mean  # (256,324) - (324,) --> (256,324)
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + math.log(2 * (math.pi))) / 2 - x_power
    if reduce == 'sum':
        return loss.sum()
    else:
        return loss


def rescale_adj(adj, type='all'):
    # Previous paper didn't use rescale_adj.
    # In their implementation, the normalization sum is: num_neighbors = F.sum(adj, axis=(1, 2))
    # In this implementation, the normaliztion term is different
    # raise NotImplementedError
    # (256,4,9, 9):
    # 4: single, double, triple, and bond between disconnected atoms (negative mask of sum of previous)
    # 1-adj[i,:3,:,:].sum(dim=0) == adj[i,4,:,:]
    # usually first 3 matrices have no diagnal, the last has.
    # A_prime = self.A + sp.eye(self.A.shape[0])
    if type == 'view':
        out_degree = adj.sum(dim=-1)
        out_degree_sqrt_inv = out_degree.pow(-1)
        out_degree_sqrt_inv[out_degree_sqrt_inv == float('inf')] = 0
        adj_prime = out_degree_sqrt_inv.unsqueeze(-1) * adj  # (256,4,9,1) * (256, 4, 9, 9) = (256, 4, 9, 9)
    else:  # default type all
        num_neighbors = adj.sum(dim=(1, 2)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv[:, None, None, :] * adj
    return adj_prime


def logit_pre_process(x, a=0.05, bounds=0.9):
    """Dequantize the input image `x` and convert to logits.

    See Also:
        - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
        - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1

    Args:
        x (torch.Tensor): Input image.

    Returns:
        y (torch.Tensor): Dequantized logits of `x`.
    """
    y = (1-a) * x + a * torch.rand_like(x)
    y = (2 * y - 1) * bounds
    y = (y + 1) / 2
    y = y.log() - (1. - y).log()

    # Save log-determinant of Jacobian of initial transform
    ldj = F.softplus(y) + F.softplus(-y) \
        - F.softplus(torch.tensor(math.log(1. - bounds) - math.log(bounds)))
    sldj = ldj.flatten(1).sum(-1)

    return y, sldj


class MoFlow(nn.Module):
    def __init__(self, hyper_params: Hyperparameters):
        super(MoFlow, self).__init__()
        self.hyper_params = hyper_params  # hold all the parameters, easy for save and load for further usage

        # More parameters derived from hyper_params for easy use
        self.b_n_type = hyper_params.b_n_type
        self.a_n_node = hyper_params.a_n_node
        self.a_n_type = hyper_params.a_n_type
        self.b_size = self.a_n_node * self.a_n_node * self.b_n_type
        self.a_size = self.a_n_node * self.a_n_type
        self.noise_scale = hyper_params.noise_scale
        if hyper_params.learn_dist:
            self.ln_var = nn.Parameter(torch.zeros(1))  # (torch.zeros(2))  2 is worse than 1
        else:
            self.register_buffer('ln_var', torch.zeros(1))  # self.ln_var = torch.zeros(1)

        self.bond_model = Glow(
            in_channel=hyper_params.b_n_type,  # 4,
            n_flow=hyper_params.b_n_flow,  # 10, # n_flow 10-->20  n_flow=20
            n_block=hyper_params.b_n_block,  # 1,
            squeeze_fold=hyper_params.b_n_squeeze,  # 3,
            hidden_channel=hyper_params.b_hidden_ch,  # [128, 128],
            affine=hyper_params.b_affine,  # True,
            conv_lu=hyper_params.b_conv_lu  # 0,1,2
        )  # n_flow=9, n_block=4

        self.atom_model = GlowOnGraph(
            n_node=hyper_params.a_n_node,  # 9,
            in_dim=hyper_params.a_n_type,  # 5,
            hidden_dim_dict={'gnn': hyper_params.a_hidden_gnn, 'linear': hyper_params.a_hidden_lin},  # {'gnn': [64], 'linear': [128, 64]},
            n_flow=hyper_params.a_n_flow,  # 27,
            n_block=hyper_params.a_n_block,  # 1,
            mask_row_size_list=hyper_params.mask_row_size_list,  # [1],
            mask_row_stride_list=hyper_params.mask_row_stride_list,  # [1],
            affine=hyper_params.a_affine  # True
        )

    def forward(self, adj, x, adj_normalized):
        """
        :param adj:  (256,4,9,9)
        :param x: (256,9,5)
        :return:
        """
        h = x  # (256,9,5)
        # add uniform noise to node feature matrices
        # + noise didn't change log-det. 1. change to logit transform 2. *0.9 ---> *other value???
        if self.training:
            if self.noise_scale == 0:
                h = h/2.0 - 0.5 + torch.rand_like(x) * 0.4  #/ 2.0  similar to X + U(0, 0.8)   *0.5*0.8=0.4
            else:
                h = h + torch.rand_like(x) * self.noise_scale  # noise_scale default 0.9
            # h, log_det_logit_x = logit_pre_process(h) # to delete
        h, sum_log_det_jacs_x = self.atom_model(adj_normalized, h)
        # sum_log_det_jacs_x = sum_log_det_jacs_x + log_det_logit_x  # to delete

        # add uniform noise to adjacency tensors
        if self.training:
            if self.noise_scale == 0:
                adj = adj/2.0 - 0.5 + torch.rand_like(adj) * 0.4  #/ 2.0
            else:
                adj = adj + torch.rand_like(adj) * self.noise_scale  # (256,4,9,9) noise_scale default 0.9
            # adj, log_det_logit_adj = logit_pre_process(adj)  # to delete
        adj_h, sum_log_det_jacs_adj = self.bond_model(adj)
        # sum_log_det_jacs_adj = log_det_logit_adj + sum_log_det_jacs_adj  # to delete
        out = [h, adj_h]  # combine to one tensor later bs * dim tensor

        return out, [sum_log_det_jacs_x, sum_log_det_jacs_adj]

    def reverse(self, z, true_adj=None):  # change!!! z[0] --> for z_x, z[1] for z_adj, a list!!!
        """
        Returns a molecule, given its latent vector.
        :param z: latent vector. Shape: [B, N*N*M + N*T]    (100,369) 369=9*9 * 4 + 9*5
            B = Batch size, N = number of atoms, M = number of bond types,
            T = number of atom types (Carbon, Oxygen etc.)
        :param true_adj: used for testing. An adjacency matrix of a real molecule
        :return: adjacency matrix and feature matrix of a molecule
        """
        batch_size = z.shape[0]  # 100,  z.shape: (100,369)

        with torch.no_grad():
            z_x = z[:, :self.a_size]  # (100, 45)
            z_adj = z[:, self.a_size:]  # (100, 324)

            if true_adj is None:
                h_adj = z_adj.reshape(batch_size, self.b_n_type, self.a_n_node, self.a_n_node)  # (100,4,9,9)
                h_adj = self.bond_model.reverse(h_adj)

                if self.noise_scale == 0:
                    h_adj = (h_adj + 0.5) * 2
                # decode adjacency matrix from h_adj
                adj = h_adj
                adj = adj + adj.permute(0, 1, 3, 2)
                adj = adj / 2
                adj = adj.softmax(dim=1)  # (100,4!!!,9,9) prob. for edge 0-3 for every pair of nodes
                max_bond = adj.max(dim=1).values.reshape(batch_size, -1, self.a_n_node, self.a_n_node)  # (100,1,9,9)
                adj = torch.floor(adj / max_bond)  # (100,4,9,9) /  (100,1,9,9) -->  (100,4,9,9)
            else:
                adj = true_adj

            h_x = z_x.reshape(batch_size, self.a_n_node, self.a_n_type)
            adj_normalized = rescale_adj(adj).to(h_x)
            h_x = self.atom_model.reverse(adj_normalized, h_x)
            if self.noise_scale == 0:
                h_x = (h_x + 0.5) * 2
            # h_x = torch.sigmoid(h_x)  # to delete for logit
        return adj, h_x  # (100,4,9,9), (100,9,5)

    def log_prob(self, z, logdet):  # z:[(256,45), (256,324)] logdet:[(256,),(256,)]
        # If I din't use self.ln_var, then I can parallel the code!
        z[0] = z[0].reshape(z[0].shape[0],-1)
        z[1] = z[1].reshape(z[1].shape[0], -1)

        logdet[0] = logdet[0] - self.a_size * math.log(2.)  # n_bins = 2**n_bit = 2**1=2
        logdet[1] = logdet[1] - self.b_size * math.log(2.)
        if len(self.ln_var) == 1:
            ln_var_adj = self.ln_var * torch.ones([self.b_size]).to(z[0])  # (324,)
            ln_var_x = self.ln_var * torch.ones([self.a_size]).to(z[0])  # (45)
        else:
            ln_var_adj = self.ln_var[0] * torch.ones([self.b_size]).to(z[0])  # (324,) 0 for bond
            ln_var_x = self.ln_var[1] * torch.ones([self.a_size]).to(z[0])  # (45) 1 for atom
        nll_adj = torch.mean(
            torch.sum(gaussian_nll(z[1], torch.zeros(self.b_size).to(z[0]), ln_var_adj, reduce='no'), dim=1)
            - logdet[1])
        nll_adj = nll_adj / (self.b_size * math.log(2.))  # the negative log likelihood per dim with log base 2

        nll_x = torch.mean(torch.sum(
            gaussian_nll(z[0], torch.zeros(self.a_size).to(z[0]), ln_var_x, reduce='no'),
            dim=1) - logdet[0])
        nll_x = nll_x / (self.a_size * math.log(2.))  # the negative log likelihood per dim with log base 2
        if nll_x.item() < 0:
            print('nll_x:{}'.format(nll_x.item()))

        return [nll_x, nll_adj]

    def save_hyperparams(self, path):
        self.hyper_params.save(path)


if __name__ == '__main__':
    hyperparams = Hyperparameters(
        b_n_type=4,
        b_n_flow=2,
        b_n_block=1,
        b_n_squeeze=3,
        b_hidden_ch=[128,128],
        b_affine=True,
        b_conv_lu=True,
        # For Atom
        a_n_node=9,
        a_n_type=5,
        a_hidden_gnn=[64],
        a_hidden_lin=[128,64],
        a_n_flow=27,
        a_n_block=1,
        mask_row_size_list=[1],
        mask_row_stride_list=[1],
        a_affine=True,
        # General
        path=None,
        learn_dist=True,
        seed=1)
    hyperparams.print()
    with torch.autograd.set_detect_anomaly(True):
        model = MoFlow(hyperparams)
        bs = 2
        x = torch.ones((bs, 9, 5), dtype=torch.float32)
        adj = torch.ones((bs, 4, 9, 9), dtype=torch.float32)
        output = model(adj, x, adj)
        print(output)
        print("Test forward:", output[0][0].shape,  output[0][1].shape)

        # z: [(256, 45), (256, 324)]
        # logdet: [(256,), (256,)]
        # z = [torch.randn(2, 45), torch.randn(2, 324)]
        # logdet = [torch.zeros(2), torch.zeros(2)]
        # nll_x, nll_adj = model.log_prob(z, logdet)

        nll_x, nll_adj = model.log_prob(output[0], output[1])
        o = nll_x + nll_adj
        print("Test log_prob:", nll_x, nll_adj, o)
        o.backward()

        z = torch.randn(2, 369)
        r_out = model.reverse(z)
        print("Test reverse:", r_out[0].shape, r_out[1].shape)




