import torch
import torch.nn as nn

from mflow.models.basic import ZeroConv2d, ActNorm, InvConv2dLU, InvConv2d, InvRotationLU, InvRotation, ActNorm2D
from mflow.models.coupling import AffineCoupling, GraphAffineCoupling


class Flow(nn.Module):
    def __init__(self, in_channel, hidden_channels, affine=True, conv_lu=2, mask_swap=False):
        super(Flow, self).__init__()

        # More stable to support more flows
        self.actnorm = ActNorm(in_channel)

        if conv_lu == 0:
            self.invconv = InvConv2d(in_channel)
        elif conv_lu == 1:
            self.invconv = InvConv2dLU(in_channel)
        elif conv_lu == 2:
            self.invconv = None
        else:
            raise ValueError("conv_lu in {0,1,2}, 0:InvConv2d, 1:InvConv2dLU, 2:none-just swap to update in coupling")

        # May add more parameter to further control net in the coupling layer
        self.coupling = AffineCoupling(in_channel, hidden_channels, affine=affine, mask_swap=mask_swap)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        # out = input
        # logdet = 0
        if self.invconv:
            out, det1 = self.invconv(out)
        else:
            det1 = 0
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        if self.invconv:
            input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


# class Flow2(nn.Module): # delete
#     def __init__(self, in_channel, hidden_channels, affine=True, conv_lu=True, mask_swap=0):
#         super(Flow2, self).__init__()
#
#         # More stable to support more flows
#         self.actnorm = ActNorm(in_channel)  # Delete ActNorm first, What I need is to norm adj, rather than along batch dim
#
#         # if conv_lu:
#         #     self.invconv = InvConv2dLU(in_channel)
#         #
#         # else:
#         #     self.invconv = InvConv2d(in_channel)
#
#         # May add more parameter to further control net in the coupling layer
#         self.coupling = AffineCoupling(in_channel, hidden_channels, affine=affine, mask_swap=mask_swap)
#
#     def forward(self, input):
#         out, logdet = self.actnorm(input)
#         # out = input
#         # logdet = 0
#         # out, det1 = self.invconv(out)
#         det1 = 0
#         out, det2 = self.coupling(out)
#
#         logdet = logdet + det1
#         if det2 is not None:
#             logdet = logdet + det2
#
#         return out, logdet
#
#     def reverse(self, output):
#         input = self.coupling.reverse(output)
#         # input = self.invconv.reverse(input)
#         input = self.actnorm.reverse(input)
#
#         return input


class FlowOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, masked_row, affine=True):
        super(FlowOnGraph, self).__init__()
        self.n_node = n_node
        self.in_dim = in_dim
        self.hidden_dim_dict = hidden_dim_dict
        self.masked_row = masked_row
        self.affine = affine
        # self.conv_lu = conv_lu
        self.actnorm = ActNorm2D(in_dim=n_node)  # May change normalization later, column norm, or row norm
        # self.invconv = InvRotationLU(n_node) # Not stable for inverse!!! delete!!!
        self.coupling = GraphAffineCoupling(n_node, in_dim, hidden_dim_dict, masked_row, affine=affine)

    def forward(self, adj, input):  # (2,4,9,9) (2,2,9,5)
        # if input are two channel identical, normalized results are 0
        # change other normalization for input
        out, logdet = self.actnorm(input)
        # out = input
        # logdet = torch.zeros(1).to(input)
        # out, det1 = self.invconv(out)
        det1 = 0
        out, det2 = self.coupling(adj, out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, adj, output):
        input = self.coupling.reverse(adj, output)
        # input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input) # change other normalization for input
        return input


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, squeeze_fold, hidden_channels, affine=True, conv_lu=2):  # in_channel: 3, n_flow: 32
        super(Block, self).__init__()
        # squeeze_fold = 3 for qm9 (bs,4,9,9), squeeze_fold = 2 for zinc (bs, 4, 38, 38)
        #                          (bs,4*3*3,3,3)                        (bs,4*2*2,19,19)
        self.squeeze_fold = squeeze_fold
        squeeze_dim = in_channel * self.squeeze_fold * self.squeeze_fold

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            if conv_lu in (0, 1):
                self.flows.append(Flow(squeeze_dim, hidden_channels,
                                       affine=affine, conv_lu=conv_lu, mask_swap=False))
            else:
                self.flows.append(Flow(squeeze_dim, hidden_channels,
                                       affine=affine, conv_lu=2, mask_swap=bool(i % 2)))

        # self.prior = ZeroConv2d(squeeze_dim, squeeze_dim*2)

    def forward(self, input):
        out = self._squeeze(input)
        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        out = self._unsqueeze(out)
        return out, logdet  # , log_p, z_new

    def reverse(self, output):  # , eps=None, reconstruct=False):
        input = self._squeeze(output)

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        unsqueezed = self._unsqueeze(input)
        return unsqueezed

    def _squeeze(self, x):
        """Trade spatial extent for channels. In forward direction, convert each
        1x4x4 volume of input into a 4x1x1 volume of output.

        Args:
            x (torch.Tensor): Input to squeeze or unsqueeze.
            reverse (bool): Reverse the operation, i.e., unsqueeze.

        Returns:
            x (torch.Tensor): Squeezed or unsqueezed tensor.
        """
        # b, c, h, w = x.size()
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold

        squeezed = x.view(b_size, n_channel, height // fold,  fold,  width // fold,  fold)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4).contiguous()
        out = squeezed.view(b_size, n_channel * fold * fold, height // fold, width // fold)
        return out

    def _unsqueeze(self, x):
        assert len(x.shape) == 4
        b_size, n_channel, height, width = x.shape
        fold = self.squeeze_fold
        unsqueezed = x.view(b_size, n_channel // (fold * fold), fold, fold, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
        out = unsqueezed.view(b_size, n_channel // (fold * fold), height * fold, width * fold)
        return out


# class Block2(nn.Module): # delete
#     def __init__(self, in_channel, n_flow, squeeze_fold, hidden_channels, affine=True, conv_lu=True):  # in_channel: 3, n_flow: 32
#         super(Block2, self).__init__()
#         # squeeze_fold = 3 for qm9 (bs,4,9,9), squeeze_fold = 2 for zinc (bs, 4, 38, 38)
#         #                          (bs,4*3*3,3,3)                        (bs,4*2*2,19,19)
#         self.squeeze_fold = squeeze_fold
#         squeeze_dim = in_channel * self.squeeze_fold * self.squeeze_fold
#
#         self.flows = nn.ModuleList()
#         for i in range(n_flow):
#             self.flows.append(Flow2(squeeze_dim, hidden_channels, affine=affine, conv_lu=conv_lu, mask_type=i % 2))
#
#         self.prior = ZeroConv2d(squeeze_dim, squeeze_dim*2)
#
#     def forward(self, input):
#         out = self._squeeze(input)
#         logdet = 0
#
#         for flow in self.flows:
#             out, det = flow(out)
#             logdet = logdet + det
#
#         out = self._unsqueeze(out)
#         return out, logdet  # , log_p, z_new
#
#     def reverse(self, output):  # , eps=None, reconstruct=False):
#         input = self._squeeze(output)
#
#         for flow in self.flows[::-1]:
#             input = flow.reverse(input)
#
#         unsqueezed = self._unsqueeze(input)
#         return unsqueezed
#
#     def _squeeze(self, x):
#         """Trade spatial extent for channels. In forward direction, convert each
#         1x4x4 volume of input into a 4x1x1 volume of output.
#
#         Args:
#             x (torch.Tensor): Input to squeeze or unsqueeze.
#             reverse (bool): Reverse the operation, i.e., unsqueeze.
#
#         Returns:
#             x (torch.Tensor): Squeezed or unsqueezed tensor.
#         """
#         # b, c, h, w = x.size()
#         assert len(x.shape) == 4
#         b_size, n_channel, height, width = x.shape
#         fold = self.squeeze_fold
#
#         squeezed = x.view(b_size, n_channel, height // fold,  fold,  width // fold,  fold)
#         squeezed = squeezed.permute(0, 1, 3, 5, 2, 4).contiguous()
#         out = squeezed.view(b_size, n_channel * fold * fold, height // fold, width // fold)
#         return out
#
#     def _unsqueeze(self, x):
#         assert len(x.shape) == 4
#         b_size, n_channel, height, width = x.shape
#         fold = self.squeeze_fold
#         unsqueezed = x.view(b_size, n_channel // (fold * fold), fold, fold, height, width)
#         unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3).contiguous()
#         out = unsqueezed.view(b_size, n_channel // (fold * fold), height * fold, width * fold)
#         return out


class BlockOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, n_flow, mask_row_size=1, mask_row_stride=1, affine=True):  #, conv_lu=True):
        """

        :param n_node:
        :param in_dim:
        :param hidden_dim:
        :param n_flow:
        :param mask_row_size: number of rows to be masked for update
        :param mask_row_stride: number of steps between two masks' firs row
        :param affine:
        """
        # in_channel=2 deleted. in_channel: 3, n_flow: 32
        super(BlockOnGraph, self).__init__()
        assert 0 < mask_row_size < n_node
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            start = i * mask_row_stride
            masked_row =[r % n_node for r in range(start, start+mask_row_size)]
            self.flows.append(FlowOnGraph(n_node, in_dim, hidden_dim_dict, masked_row=masked_row, affine=affine))
        # self.prior = ZeroConv2d(2, 4)

    def forward(self, adj, input):
        out = input
        logdet = 0
        for flow in self.flows:
            out, det = flow(adj, out)
            logdet = logdet + det
            # it seems logdet is not influenced
        return out, logdet

    def reverse(self, adj, output):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(adj, input)
        return input


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, squeeze_fold, hidden_channel, affine=True, conv_lu=2): # in_channel: 3, n_flow:32, n_block:4
        super(Glow, self).__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel  # 3
        for i in range(n_block):
            self.blocks.append(Block(n_channel, n_flow, squeeze_fold, hidden_channel, affine=affine, conv_lu=conv_lu)) # 3,6,12
            # self.blocks.append(
            #     Block2(n_channel, n_flow, squeeze_fold, hidden_channel, affine=affine, conv_lu=conv_lu))  # delete

    def forward(self, input):
        logdet = 0
        out = input

        for block in self.blocks:
            out, det = block(out)
            logdet = logdet + det

        return out, logdet

    def reverse(self, z):  # _list, reconstruct=False):
        h = z
        for i, block in enumerate(self.blocks[::-1]):
            h = block.reverse(h)

        return h


class GlowOnGraph(nn.Module):
    def __init__(self, n_node, in_dim, hidden_dim_dict, n_flow, n_block,
                 mask_row_size_list=[2], mask_row_stride_list=[1], affine=True):  # , conv_lu=True): # in_channel: 2 default
        super(GlowOnGraph, self).__init__()

        assert len(mask_row_size_list) == n_block or len(mask_row_size_list) == 1
        assert len(mask_row_stride_list) == n_block or len(mask_row_stride_list) == 1
        if len(mask_row_size_list) == 1:
            mask_row_size_list = mask_row_size_list * n_block
        if len(mask_row_stride_list) == 1:
            mask_row_stride_list = mask_row_stride_list * n_block
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            mask_row_size = mask_row_size_list[i]
            mask_row_stride = mask_row_stride_list[i]
            self.blocks.append(BlockOnGraph(n_node, in_dim, hidden_dim_dict, n_flow, mask_row_size, mask_row_stride, affine=affine))

    def forward(self, adj, x):
        # adj (bs, 4,9,9), xx:(bs, 9,5)
        logdet = 0
        out = x
        for block in self.blocks:
            out, det = block(adj, out)
            logdet = logdet + det

        return out, logdet

    def reverse(self, adj, z):
        # (bs, 4,9,9), zz: (bs, 9, 5)
        input = z
        for i, block in enumerate(self.blocks[::-1]):
            input = block.reverse(adj, input)

        return input


def test_Flow():
    from graflow.models.relgcn import rescale_adj
    torch.manual_seed(0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4

    x = torch.ones((bs, 2, nodes, ch), dtype=torch.float32)  # 2 for duplicated flow for transforming whole info
    adj = torch.randint(0, 2, (bs, num_edge_type, nodes, nodes), dtype=torch.float32)
    adj = rescale_adj(adj)

    gc = Flow(in_channel=4, hidden_channels=[512,512], affine=True, conv_lu=True)

    out = gc(adj)
    print('adj.shape:', adj.shape)
    # print('out', out.shape)  # (bs, out_ch)
    print('out[0].shape, out[1].shape', out[0].shape, out[1].shape)

    r = gc.reverse(out[0])
    print(r.shape)
    print('torch.abs(r-adj).mean():', torch.abs(r - adj).mean())


def test_Block():
    from graflow.models.relgcn import rescale_adj
    torch.manual_seed(0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4

    x = torch.ones((bs, 2, nodes, ch), dtype=torch.float32)  # 2 for duplicated flow for transforming whole info
    adj = torch.randint(0, 2, (bs, num_edge_type, nodes, nodes), dtype=torch.float32)
    adj = rescale_adj(adj)

    gc = Block(in_channel=4, n_flow=60, squeeze_fold=3, hidden_channels=[128,128], affine=True, conv_lu=True)

    out = gc(adj)
    print('adj.shape:', adj.shape)
    # print('out', out.shape)  # (bs, out_ch)
    print('out[0].shape, out[1].shape', out[0].shape, out[1].shape)

    r = gc.reverse(out[0])
    # r = gc.reverse(out[0])
    print(r.shape)
    print('torch.abs(r-adj).mean():', torch.abs(r - adj).mean())


def test_Glow():
    from graflow.models.relgcn import rescale_adj
    torch.manual_seed(0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4

    x = torch.ones((bs, 2, nodes, ch), dtype=torch.float32)  # 2 for duplicated flow for transforming whole info
    adj = torch.randint(0, 2, (bs, num_edge_type, nodes, nodes), dtype=torch.float32)
    adj = rescale_adj(adj)

    gc = Glow(in_channel=4, n_flow=10, n_block=1, squeeze_fold=3, hidden_channel=[512, 512], affine=True, conv_lu=True)
    # More n_flow, more n_block, more hidden_channels all introducing more reconstruction error
    out = gc(adj)
    print('adj.shape:', adj.shape)
    # print('out', out.shape)  # (bs, out_ch)
    print('out[0].shape, out[1].shape', out[0].shape, out[1].shape)

    r = gc.reverse(out[0])
    print(r.shape)
    print('torch.abs(r-adj).mean():', torch.abs(r-adj).mean())
    # print('.nelement()')


def test_FlowOnGraph():
    from graflow.models.relgcn import rescale_adj
    torch.manual_seed(0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4

    # x = torch.randint(0,2, (bs, 1, nodes, ch), dtype=torch.float32)  # 2 for duplicated flow for transforming whole info
    # x = torch.cat([x,x], dim=1)
    x = torch.randint(0, 2, (bs,  nodes, ch), dtype=torch.float32)
    adj = torch.randint(0, 2, (bs, num_edge_type, nodes, nodes), dtype=torch.float32)
    adj = rescale_adj(adj)

    in_dim = ch
    hidden_dim_dict = {'gnn': [8,64], 'linear': [8]}

    gc = FlowOnGraph(nodes, in_dim, hidden_dim_dict, masked_row=range(0, nodes, 2), affine=True)
    out = gc(adj, x)
    print('in', x.shape, adj.shape)
    # print('out', out.shape)  # (bs, out_ch)
    print(out[0].shape, out[1].shape)

    r = gc.reverse(adj, out[0])
    print(r.shape)
    print('torch.abs(r-x).mean():', torch.abs(r - x).mean())
    print('End of test_FlowOnGraph')


def test_BlockOnGraph():
    from graflow.models.relgcn import rescale_adj
    torch.manual_seed(0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4

    # x = torch.randint(0, 2, (bs, 1, nodes, ch),
    #                   dtype=torch.float32)
    # x = torch.cat([x, x], dim=1)
    x = torch.randint(0, 2, (bs,  nodes, ch), dtype=torch.float32)
    x = torch.softmax(x, dim=2)
    # nm = x.sum(dim=1, keepdim=True)
    # nm = x.sum(dim=(1,2), keepdim=True)
    # nm[nm==0] = 1e-6
    # x = x/nm
    adj = torch.randint(0, 2, (bs, num_edge_type, nodes, nodes), dtype=torch.float32)
    adj = rescale_adj(adj)

    in_dim = ch
    hidden_dim_dict = {'gnn': [8,64], 'linear': [8]}
    gc = BlockOnGraph(nodes, in_dim, hidden_dim_dict, n_flow=30, mask_row_size=3, mask_row_stride=2, affine=True) # n_flow >= 10, large error, WHY???
    # n_flow=10, torch.abs(r-adj).mean(): tensor(0.0073, grad_fn=<MeanBackward0>) WHY???
    out = gc(adj, x)
    print('in', x.shape, adj.shape)
    # print('out', out.shape)  # (bs, out_ch)
    print(out[0].shape, out[1].shape)

    r = gc.reverse(adj, out[0])
    # r_m = r.mean(dim=-1, keepdim=True)
    # std_m = r.std(dim=-1, keepdim=True)
    # r = (r - r_m)/std_m
    print(r.shape)
    print('torch.abs(r-x).mean():', torch.abs(r - x).mean())


def test_GlowOnGraph():
    from graflow.models.relgcn import rescale_adj
    torch.manual_seed(0)
    bs = 2
    nodes = 9
    ch = 5
    num_edge_type = 4

    # x = torch.randint(0, 2, (bs, 1, nodes, ch), dtype=torch.float32)
    # x = torch.cat([x, x], dim=1)
    x = torch.randint(0, 2, (bs,  nodes, ch), dtype=torch.float32)
    adj = torch.randint(0, 2, (bs, num_edge_type, nodes, nodes), dtype=torch.float32)
    adj = rescale_adj(adj)

    # mask = torch.ones(nodes, nodes).type(torch.bool)
    # mask[:, 0] = False
    # out_ch = 128
    in_dim = ch
    hidden_dim_dict = {'gnn': [8,64], 'linear':[8]}
    # gc = BlockOnGraph(in_dim, hidden_dim, n_flow=3, split=False, affine=True, conv_lu=True)
    gc = GlowOnGraph(nodes, in_dim, hidden_dim_dict, n_flow=30, n_block=2, mask_row_size_list=[1, 2], mask_row_stride_list=[1, 2],  affine=True)

    out = gc(adj, x)
    print('in', x.shape, adj.shape)
    # print('out', out.shape)  # (bs, out_ch)
    print(out[0].shape, out[1].shape)

    r = gc.reverse(adj, out[0])
    print(r.shape)
    print('torch.abs(r-x).mean():', torch.abs(r - x).mean())


if __name__ == '__main__':
    # test_AdditiveAdjCoupling()
    # test_AdditiveNodeFeatureCoupling()
    # test_FlowOnGraph()
    # test_BlockOnGraph()
    # test_GlowOnGraph()

    test_Flow()
    test_Block()
    test_Glow()
