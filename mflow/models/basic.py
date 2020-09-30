import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la
logabs = lambda x: torch.log(torch.abs(x))


def conv_mask():
    pass


# Basic invertible layers
class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class ActNorm2D(nn.Module):
    def __init__(self, in_dim, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_dim, 1))
        self.scale = nn.Parameter(torch.ones(1, in_dim, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvRotationLU(nn.Module):
    def __init__(self, dim):
        super(InvRotationLU, self).__init__()
        # (9*9)  * (bs*9*5)
        weight = np.random.randn(dim, dim)  # (9,9)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)  # a Permutation matrix
        w_l = torch.from_numpy(w_l)  # L matrix from PLU
        w_s = torch.from_numpy(w_s)  # diagnal of the U matrix from PLU
        w_u = torch.from_numpy(w_u)  # u - dianal of the U matrix from PLU

        self.register_buffer('w_p', w_p)  # (12,12)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))  # (12,12) upper 1 with 0 diagnal
        self.register_buffer('l_mask', torch.from_numpy(l_mask))  # (12,12) lower 1 with 0 diagnal
        self.register_buffer('s_sign', torch.sign(w_s))  # (12,)  # sign of the diagnal of the U matrix from PLU
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))  # (12,12) 1 diagnal
        self.w_l = nn.Parameter(w_l)  # (12,12)
        self.w_s = nn.Parameter(logabs(w_s))  # (12, )
        self.w_u = nn.Parameter(w_u)  # (12,12)

    def forward(self, input):
        bs, height, width = input.shape  #    (bs, 9, 5)

        weight = self.calc_weight()  #  9,9

        # out = F.conv2d(input, weight)  # (2,12,32,32), (12,12,1,1) --> (2,12,32,32)
        # logdet = height * width * torch.sum(self.w_s)

        out = torch.matmul(weight, input)  # (1, 9,9) * (bs, 9, 5) --> (bs, 9, 5)
        logdet = width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        # weight = torch.matmul(torch.matmul(self.w_p, (self.w_l * self.l_mask + self.l_eye)),
        #              ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s))))
        # weight = self.w_p
        return weight.unsqueeze(0)  # weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()
        # return weight.inverse() @ output
        return torch.matmul(weight.inverse(), output)
        # return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))  #np.linalg.det(weight.data.numpy())


class InvRotation(nn.Module):
    def __init__(self, dim):
        super().__init__()

        weight = torch.randn(dim, dim)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(0)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, height, width = input.shape

        # out = F.conv2d(input, self.weight)
        out = self.weight @ input
        logdet = (
            width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return self.weight.squeeze().inverse().unsqueeze(0) @ output


# Basic non-invertible layers in coupling _s_t_function, or for transforming Gaussian distribution
class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)  # in:512, out:12
        self.conv.weight.data.zero_()  # (12,512,3,3)
        self.conv.bias.data.zero_()  # 12
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))  # (1,12,1,1)

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)  # input: (2,512,32,32) --> (2,512,34,34)
        out = self.conv(out)  # (2,12,32,32)
        out = out * torch.exp(self.scale * 3)  # (2,12,32,32) * (1,12,1,1) = (2,12,32,32)

        return out


# Basic non-invertible layers in coupling _s_t_function,
class GraphLinear(nn.Module):
    """Graph Linear layer.
        This function assumes its input is 3-dimensional. Or 4-dim or whatever, only last dim are changed
        Differently from :class:`nn.Linear`, it applies an affine
        transformation to the third axis of input `x`.
        Warning: original Chainer.link.Link use i.i.d. Gaussian initialization as default,
        while default nn.Linear initialization using init.kaiming_uniform_

    .. seealso:: :class:`nn.Linear`
    """
    def __init__(self, in_size, out_size, bias=True):
        super(GraphLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size, bias) # Warning: differential initialization from Chainer

    def forward(self, x):
        """Forward propagation.
            Args:
                x (:class:`chainer.Variable`, or :class:`numpy.ndarray`\
                or :class:`cupy.ndarray`):
                    Input array that should be a float array whose ``ndim`` is 3.

                    It represents a minibatch of atoms, each of which consists
                    of a sequence of molecules. Each molecule is represented
                    by integer IDs. The first axis is an index of atoms
                    (i.e. minibatch dimension) and the second one an index
                    of molecules.

            Returns:
                :class:`chainer.Variable`:
                    A 3-dimeisional array.

        """
        # h = x
        # s0, s1, s2 = h.shape
        # h = h.reshape(-1, s2)  # shape: (s0*s1, s2)
        # h = self.linear(h)
        # h = h.reshape(s0, s1, self.out_size)
        h = x
        h = h.reshape(-1, x.shape[-1])  # shape: (s0*s1, s2)
        h = self.linear(h)
        h = h.reshape(tuple(x.shape[:-1] + (self.out_size,)))
        return h


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_edge_type=4):
        """

        :param in_channels:   e.g. 8
        :param out_channels:  e.g. 64
        :param num_edge_type:  e.g. 4 types of edges/bonds
        """
        super(GraphConv, self).__init__()

        self.graph_linear_self = GraphLinear(in_channels, out_channels)
        self.graph_linear_edge = GraphLinear(in_channels, out_channels * num_edge_type)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def forward(self, adj, h):
        """
        graph convolution over batch and multi-graphs
        :param h: shape: (256,9, 8)
        :param adj: shape: (256,4,9,9)
        :return:
        """
        mb, node, ch = h.shape # 256, 9, 8
        # --- self connection, apply linear function ---
        hs = self.graph_linear_self(h)  # (256,9, 8) --> (256,9, 64)
        # --- relational feature, from neighbor connection ---
        # Expected number of neighbors of a vertex
        # Since you have to divide by it, if its 0, you need to arbitrarily set it to 1
        m = self.graph_linear_edge(h)  # (256,9, 8) --> (256,9, 64*4), namely (256,9, 256)
        m = m.reshape(mb, node, self.out_ch, self.num_edge_type)  # (256,9, 256) --> (256,9, 64, 4)
        m = m.permute(0, 3, 1, 2)  # (256,9, 64, 4) --> (256, 4, 9, 64)
        # m: (batchsize, edge_type, node, ch)
        # hr: (batchsize, edge_type, node, ch)
        hr = torch.matmul(adj, m)  # (256,4,9,9) * (256, 4, 9, 64) = (256, 4, 9, 64)
        # hr: (batchsize, node, ch)
        hr = hr.sum(dim=1)  # (256, 4, 9, 64) --> (256, 9, 64)
        return hs + hr  #


def test_ZeroConv2d():
    in_channel = 1
    out_channel = 2

    x = torch.ones(2, 1, 5, 5)
    net = ZeroConv2d(in_channel, out_channel)
    y = net(x)
    print('x.shape:', x.shape)
    print(x)
    print('y.shape', y.shape)
    print(y)


def test_actnorm():
    in_channel = 1
    out_channel = 2

    x = torch.ones(2, 1, 3, 3)
    net = ActNorm(in_channel)
    y = net(x)
    print('x.shape:', x.shape)
    print(x)
    print('y.shape', y[0].shape)
    print(y[0])


if __name__ == '__main__':
    torch.manual_seed(0)
    test_actnorm()
    # test_ZeroConv2d()
    #
    # nodes = 4
    # ch = 5
    #
    # x = torch.ones((nodes, ch), dtype=torch.float32)
    #
    # units = [64, 128]
    #
    # mlp = MLP(units=units, in_size=ch)
    # print('in', x.shape)
    # out = mlp(x)
    # print('out', out.shape)  # (bs, out_ch)
    # print(out)