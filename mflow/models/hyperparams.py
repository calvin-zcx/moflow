import json
import os
import numpy as np
from tabulate import tabulate
import torch


class Hyperparameters:
    def __init__(self,
                 # For bond
                 b_n_type=4, b_n_flow=-1, b_n_block=-1, b_n_squeeze=-1, b_hidden_ch=None, b_affine=True, b_conv_lu=2,
                 # For atom
                 a_n_node=-1, a_n_type=-1, a_hidden_gnn=None, a_hidden_lin=None, a_n_flow=-1, a_n_block=1,
                 mask_row_size_list=None, mask_row_stride_list=None, a_affine=True,
                 # General
                 path=None, learn_dist=True, seed=1, noise_scale=0.6):
        """

        :param b_n_type: Number of bond types/channels
        :param b_n_flow: Number of masked glow coupling layers per block for bond tensor
        :param b_n_block: Number of glow blocks for bond tensor
        :param b_n_squeeze:  Squeeze divisor, 3 for qm9, 2 for zinc250k
        :param b_hidden_ch:Hidden channel list for bonds tensor, delimited list input
        :param b_affine:Using affine coupling layers for bonds glow
        :param b_conv_lu: Using L-U decomposition trick for 1-1 conv in bonds glow
        :param a_n_node: Maximum number of atoms in a molecule
        :param a_n_type: Number of atom types
        :param a_hidden_gnn:Hidden dimension list for graph convolution for atoms matrix, delimited list input
        :param a_hidden_lin:Hidden dimension list for linear transformation for atoms, delimited list input
        :param a_n_flow:Number of masked flow coupling layers per block for atom matrix
        :param a_n_block:Number of flow blocks for atom matrix
        :param mask_row_size_list: Mask row list for atom matrix, delimited list input
        :param mask_row_stride_list: Mask row stride  list for atom matrix, delimited list input
        :param a_affine: Using affine coupling layers for atom conditional graph flow
        :param path:
        :param learn_dist: learn the distribution of feature matrix
        :param noise_scale:
        """
        self.b_n_type = b_n_type  # 4
        self.b_n_flow = b_n_flow  # 10
        self.b_n_block = b_n_block  # 1
        self.b_n_squeeze = b_n_squeeze  # 3 or 2
        self.b_hidden_ch = b_hidden_ch  # [128,128]
        self.b_affine = b_affine  # True
        self.b_conv_lu = b_conv_lu  # True

        self.a_n_node = a_n_node  # 9
        self.a_n_type = a_n_type  # 5
        self.a_hidden_gnn = a_hidden_gnn  # [64]
        self.a_hidden_lin = a_hidden_lin  # [128, 64]
        self.a_n_flow = a_n_flow  # 27
        self.a_n_block = a_n_block  # 1
        self.mask_row_size_list = mask_row_size_list  # [9]
        self.mask_row_stride_list = mask_row_stride_list  # [True]
        self.a_affine = a_affine  # True

        self.path = path  # None
        self.learn_dist = learn_dist  # 1
        self.seed = seed  # 1
        self.noise_scale = noise_scale

        # load function in the initialization by path argument
        if path is not None:
            if os.path.exists(path) and os.path.isfile(path):
                with open(path, "r") as f:
                    obj = json.load(f)
                    for (key, value) in obj.items():
                        setattr(self, key, value)
            else:
                raise Exception("{} does not exist".format(path))

    def save(self, path):
        self.path = path
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, cls=NumpyEncoder)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        print(tabulate(rows))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.numpy().tolist()  # what if I use obj.detach().tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    hyper = Hyperparameters()
    hyper.save('test_saving_hyper.json')
