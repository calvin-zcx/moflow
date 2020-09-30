import os
import sys
# for linux env.
sys.path.insert(0,'..')
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
from data.data_frame_parser import DataFrameParser
from data.data_loader import NumpyTupleDataset
from data.smile_to_graph import GGNNPreprocessor


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='qm9',
                        choices=['qm9', 'zinc250k'],
                        help='dataset to be downloaded')
    parser.add_argument('--data_type', type=str, default='relgcn',
                        choices=['gcn', 'relgcn'],)
    args = parser.parse_args()
    return args


start_time = time.time()
args = parse()
data_name = args.data_name
data_type = args.data_type
print('args', vars(args))

if data_name == 'qm9':
    max_atoms = 9
elif data_name == 'zinc250k':
    max_atoms = 38
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))


if data_type == 'relgcn':
    # preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True, return_is_real_node=False)
    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
else:
    raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

data_dir = "."
os.makedirs(data_dir, exist_ok=True)

if data_name == 'qm9':
    print('Preprocessing qm9 data:')
    df_qm9 = pd.read_csv('qm9.csv', index_col=0)
    labels = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
              'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES1')
    result = parser.parse(df_qm9, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'zinc250k':
    print('Preprocessing zinc250k data')
    # dataset = datasets.get_zinc250k(preprocessor)
    df_zinc250k = pd.read_csv('zinc250k.csv', index_col=0)
    # Caution: Not reasonable but used in used in chain_chemistry\datasets\zinc.py:
    # 'smiles' column contains '\n', need to remove it.
    # Here we do not remove \n, because it represents atom N with single bond
    labels = ['logP', 'qed', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
    result = parser.parse(df_zinc250k, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

NumpyTupleDataset.save(os.path.join(data_dir, '{}_{}_kekulized_ggnp.npz'.format(data_name, data_type)), dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)) )

#
# # dataset_old = NumpyTupleDataset.load(os.path.join(data_dir, 'qm9_relgcn_kekulized_ggnp.npz'))
# dataset_new = NumpyTupleDataset.load(os.path.join(data_dir,
#                                                   '{}_{}_kekulized_ggnp_new.npz'.format(data_name, data_type)))
#
#
# dataset_old = NumpyTupleDataset.load(os.path.join(data_dir,
#                                                   '{}_{}_kekulized_ggnp.npz'.format(data_name, data_type)))
# print('len(dataset):', len(dataset))
# print('len(dataset_old):', len(dataset_old))
#
# # check consistency
# assert len(dataset) == len(dataset_old)
# for i in tqdm(range(len(dataset))):
#     if not (dataset[i][0] == dataset_old[i][0]).all():
#         print('dataset[i][0]', i, 0, smiles[i])
#     if not (dataset[i][1] == dataset_old[i][1]).all():
#         print('dataset[i][1]', i, 1, smiles[i])
#     if np.abs(dataset[i][2] - dataset_old[i][2]).mean() > 1e-5:
#         print('dataset[i][2]', i, 2, smiles[i])
#
# from rdkit import Chem
# smiles = r'Cc1ccccc1/C=N\n1cnnc1'
# def s_2_g(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     canonical_smiles, mol = preprocessor.prepare_smiles_and_mol(mol)
#     input_features = preprocessor.get_input_features(mol)
#     return input_features