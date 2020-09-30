import argparse
import os
import sys
# for linux env.
sys.path.insert(0,'..')
from distutils.util import strtobool

import pickle
import torch

import numpy as np

from data.data_loader import NumpyTupleDataset
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import transform_qm9, transform_zinc250k
from data.transform_zinc250k import zinc250_atomic_num_list, transform_fn_zinc250k
# from mflow.generate import generate_mols_along_axis
from mflow.models.hyperparams import Hyperparameters
from mflow.models.utils import check_validity, construct_mol, adj_to_smiles
from mflow.utils.model_utils import load_model, get_latent_vec, smiles_to_adj
from mflow.utils.molecular_metrics import MolecularMetrics
from mflow.models.model import MoFlow, rescale_adj
from mflow.utils.timereport import TimeReport
import mflow.utils.environment as env
from sklearn.linear_model import LinearRegression
import time
import functools
print = functools.partial(print, flush=True)


class MoFlowProp(nn.Module):
    def __init__(self, model:MoFlow, hidden_size):
        super(MoFlowProp, self).__init__()
        self.model = model
        self.latent_size = model.b_size + model.a_size
        self.hidden_size = hidden_size

        vh = (self.latent_size,) + tuple(hidden_size) + (1,)
        modules = []
        for i in range(len(vh)-1):
            modules.append(nn.Linear(vh[i], vh[i+1]))
            if i < len(vh) - 2:
                modules.append(nn.Tanh())
                # modules.append(nn.ReLU())
        self.propNN = nn.Sequential(*modules)

    def encode(self, adj, x):
        with torch.no_grad():
            self.model.eval()
            adj_normalized = rescale_adj(adj).to(adj)
            z, sum_log_det_jacs = self.model(adj, x, adj_normalized)  # z = [h, adj_h]
            h = torch.cat([z[0].reshape(z[0].shape[0], -1), z[1].reshape(z[1].shape[0], -1)], dim=1)
        return h, sum_log_det_jacs

    def reverse(self, z):
        with torch.no_grad():
            self.model.eval()
            adj, x = self.model.reverse(z, true_adj=None)
        return adj, x

    def forward(self, adj, x):
        h, sum_log_det_jacs = self.encode(adj, x)
        output = self.propNN(h)  # do I need to add nll of the unsupervised part? or just keep few epoch? see the results
        return output, h,  sum_log_det_jacs


def fit_model(model, atomic_num_list, data, data_prop,  device, property_name='qed',
              max_epochs=10, learning_rate=1e-3, weight_decay=1e-5):
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    model = model.to(device)
    model.train()
    # Loss and optimizer
    metrics = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    N = len(data.dataset)
    assert len(data_prop) == N
    iter_per_epoch = len(data)
    log_step = 20
    # batch_size = data.batch_size
    tr = TimeReport(total_iter = max_epochs * iter_per_epoch)
    if property_name == 'qed':
        col = 0 # [0,1]
    elif property_name == 'plogp':
        col = 1  # unbounded, normalized later???
    else:
        raise ValueError("Wrong property_name{}".format(property_name))

    for epoch in range(max_epochs):
        print("In epoch {}, Time: {}".format(epoch + 1, time.ctime()))
        for i, batch in enumerate(data):
            x = batch[0].to(device)   # (bs,9,5)
            adj = batch[1].to(device)   # (bs,4,9, 9)
            bs = x.shape[0]

            ps = i * bs
            pe = min((i+1)*bs, N)
            true_y = [[tt[col]] for tt in data_prop[ps:pe]]  #[[propf(mol)] for mol in true_mols]
            true_y = torch.tensor(true_y).float().cuda()
            # model and loss
            optimizer.zero_grad()

            y, z, sum_log_det_jacs = model(adj, x)

            loss = metrics(y, true_y)
            loss.backward()
            optimizer.step()
            tr.update()
            # Print log info
            if (i + 1) % log_step == 0:  # i % args.log_step == 0:
                print('Epoch [{}/{}], Iter [{}/{}], loss: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
                      format(epoch + 1, args.max_epochs, i + 1, iter_per_epoch,
                             loss.item(),
                             tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
                tr.print_summary()
            # How to validate??? set aside validation data and cal and print the loss
    tr.print_summary()
    tr.end()
    print("[fit_model Ends], Start at {}, End at {}, Total {}".
          format(time.ctime(start), time.ctime(), time.time()-start))
    return model


def optimize_mol(model:MoFlow, property_model:MoFlowProp, smiles, device, sim_cutoff, lr=2.0, num_iter=20,
             data_name='qm9', atomic_num_list=[6, 7, 8, 9, 0], property_name='qed', debug=True, random=False):
    if property_name == 'qed':
        propf = env.qed  # [0,1]
    elif property_name == 'plogp':
        propf = env.penalized_logp  # unbounded, normalized later???
    else:
        raise ValueError("Wrong property_name{}".format(property_name))
    model.eval()
    property_model.eval()
    with torch.no_grad():
        bond, atoms = smiles_to_adj(smiles, data_name)
        bond = bond.to(device)
        atoms = atoms.to(device)
        mol_vec, sum_log_det_jacs = property_model.encode(bond, atoms)
        if debug:
            adj_rev, x_rev = property_model.reverse(mol_vec)
            reverse_smiles = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            print(smiles, reverse_smiles)

            adj_normalized = rescale_adj(bond).to(device)
            z, sum_log_det_jacs = model(bond, atoms, adj_normalized)
            z0 = z[0].reshape(z[0].shape[0], -1)
            z1 = z[1].reshape(z[1].shape[0], -1)
            adj_rev, x_rev = model.reverse(torch.cat([z0, z1], dim=1))
            # val_res = check_validity(adj_rev, x_rev, atomic_num_list)
            reverse_smiles2 = adj_to_smiles(adj_rev.cpu(), x_rev.cpu(), atomic_num_list)
            train_smiles2 = adj_to_smiles(bond.cpu(), atoms.cpu(), atomic_num_list)
            print(train_smiles2, reverse_smiles2)

    mol = Chem.MolFromSmiles(smiles)
    fp1 = AllChem.GetMorganFingerprint(mol, 2)
    start = (smiles, propf(mol), None) # , mol)

    cur_vec = mol_vec.clone().detach().requires_grad_(True).to(device)  # torch.tensor(mol_vec, requires_grad=True).to(mol_vec)
    start_vec = mol_vec.clone().detach().requires_grad_(True).to(device)

    visited = []
    for step in range(num_iter):
        prop_val = property_model.propNN(cur_vec).squeeze()
        grad = torch.autograd.grad(prop_val, cur_vec)[0]
        # cur_vec = cur_vec.data + lr * grad.data
        if random:
            rad = torch.randn_like(cur_vec.data)
            cur_vec = start_vec.data + lr * rad / torch.sqrt(rad * rad)
        else:
            cur_vec = cur_vec.data + lr * grad.data / torch.sqrt(grad.data * grad.data)
        cur_vec = cur_vec.clone().detach().requires_grad_(True).to(device)  # torch.tensor(cur_vec, requires_grad=True).to(mol_vec)
        visited.append(cur_vec)

    hidden_z = torch.cat(visited, dim=0).to(device)
    adj, x = property_model.reverse(hidden_z)
    val_res = check_validity(adj, x, atomic_num_list, debug=debug)
    valid_mols = val_res['valid_mols']
    valid_smiles = val_res['valid_smiles']
    results = []
    sm_set = set()
    sm_set.add(smiles)
    for m, s in zip(valid_mols, valid_smiles):
        if s in sm_set:
            continue
        sm_set.add(s)
        p = propf(m)
        fp2 = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        if sim >= sim_cutoff:
            results.append((s, p, sim, smiles))
    # smile, property, similarity, mol
    results.sort(key=lambda tup: tup[1], reverse=True)
    return results, start


def smile_cvs_to_property(data_name='zinc250k'):
    if data_name == 'qm9':
        # Total: 133885	 Invalid: 0	 bad_plogp: 0 	 bad_qed: 0
        atomic_num_list = [6, 7, 8, 9, 0]
        filename = '../data/qm9.csv'
        colname = 'SMILES1'
    elif data_name == 'zinc250k':
        # Total: 249455	 Invalid: 0	 bad_plogp: 0 	 bad_qed: 0
        atomic_num_list = zinc250_atomic_num_list
        filename = '../data/zinc250k.csv'
        colname = 'smiles'

    df = pd.read_csv(filename)
    smiles = df[colname].tolist()
    n = len(smiles)
    # for index, row in df.iterrows():
    #     sm = row[colname]
    #     smiles.append(sm)
    f = open(data_name+'_property.csv', "w")
    f.write('qed,plogp,smile\n')
    results = []
    total = 0
    bad_qed = 0
    bad_plogp = 0
    invalid = 0
    for i, smile in enumerate(smiles):
        if i % 10000 == 0:
            print('In {}/{} line'.format(i, n))
        total += 1
        mol = Chem.MolFromSmiles(smile)
        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)
        if mol ==  None:
            print(i, smile)
            invalid += 1
            qed = -1
            plogp = -999
            smile2 = 'N/A'
            results.append((qed, plogp, smile, smile2))
            f.write('{},{},{}\n'.format(qed, plogp, smile))
            continue

        try:
            qed = env.qed(mol)
        except ValueError as e:
            bad_qed += 1
            qed = -1
            print(i + 1, Chem.MolToSmiles(mol, isomericSmiles=True), ' error in qed')

        try:
            plogp = env.penalized_logp(mol)
        except RuntimeError as e:
            bad_plogp += 1
            plogp = -999
            print(i + 1, Chem.MolToSmiles(mol, isomericSmiles=True), ' error in penalized_log')

        results.append((qed, plogp, smile, smile2))
        f.write('{},{},{}\n'.format(qed, plogp, smile))
        f.flush()
    f.close()

    results.sort(key=lambda tup: tup[0], reverse=True)
    f = open(data_name+'_property_sorted_qed.csv', "w")  #
    f.write('qed,plogp,smile\n')
    for r in results:
        qed, plogp, smile, smile2 = r
        f.write('{},{},{}\n'.format(qed, plogp, smile))
        f.flush()
    f.close()

    results.sort(key=lambda tup: tup[1], reverse=True)
    f = open(data_name+'_property_sorted_plogp.csv', "w")  #
    f.write('qed,plogp,smile\n')
    for r in results:
        qed, plogp, smile, smile2 = r
        f.write('{},{},{}\n'.format(qed, plogp, smile))
        f.flush()
    f.close()

    print('Dump done!')
    print('Total: {}\t Invalid: {}\t bad_plogp: {} \t bad_qed: {}\n'.format(total, invalid, bad_plogp, bad_qed))


def load_property_csv(data_name, normalize=True):
    """
    We use qed and plogp in zinc250k_property.csv which are recalculated by rdkit
    the recalculated qed results are in tiny inconsistent with qed in zinc250k.csv
    e.g
    zinc250k_property.csv:
    qed,plogp,smile
    0.7319008436872337,3.1399057164163766,CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
    0.9411116113894995,0.17238635659148804,C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1
    import rdkit
    m = rdkit.Chem.MolFromSmiles('CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1')
    rdkit.Chem.QED.qed(m): 0.7319008436872337
    from mflow.utils.environment import penalized_logp
    penalized_logp(m):  3.1399057164163766
    However, in oringinal:
    zinc250k.csv
    ,smiles,logP,qed,SAS
    0,CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1,5.0506,0.702012232801,2.0840945720726807
    1,C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1,3.1137,0.928975488089,3.4320038192747795

    0.7319008436872337 v.s. 0.702012232801
    and no plogp in zinc250k.csv dataset!
    """
    if data_name == 'qm9':
        # Total: 133885	 Invalid: 0	 bad_plogp: 0 	 bad_qed: 0
        filename = '../data/qm9_property.csv'
    elif data_name == 'zinc250k':
        # Total: 249455	 Invalid: 0	 bad_plogp: 0 	 bad_qed: 0
        filename = '../data/zinc250k_property.csv'

    df = pd.read_csv(filename)  # qed, plogp, smile
    if normalize:
        # plogp: # [-62.52, 4.52]
        m = df['plogp'].mean()  # 0.00026
        std = df['plogp'].std() # 2.05
        mn = df['plogp'].min()
        mx = df['plogp'].max()
        # df['plogp'] = 0.5 * (np.tanh(0.01 * ((df['plogp'] - m) / std)) + 1)  # [0.35, 0.51]
        # df['plogp'] = (df['plogp'] - m) / std
        lower = -10 # -5 # -10
        df['plogp'] = df['plogp'].clip(lower=lower, upper=5)
        df['plogp'] = (df['plogp'] - lower) / (mx-lower)

    tuples = [tuple(x) for x in df.values]
    print('Load {} done, length: {}'.format(filename, len(tuples)))
    return tuples


def write_similes(filename, data, atomic_num_list):
    """
    QM9: Total: 133885	 bad_plogp: 133885 	 bad_qed: 142   plogp is not applicable to the QM9 dataset
    zinc250k:
    :param filename:
    :param data:
    :param atomic_num_list:
    :return:
    """
    f = open(filename, "w")  # append mode
    results = []
    total = 0
    bad_qed = 0
    bad_plogp= 0
    invalid = 0
    for i, r in enumerate(data):
        total += 1
        x, adj, label = r
        mol0 = construct_mol(x, adj, atomic_num_list)
        smile = Chem.MolToSmiles(mol0, isomericSmiles=True)  # 'CC(C)(C)C1=CC=C2OC=C(CC(=O)NC3=CC=CC=C3F)C2=C1'
        mol = Chem.MolFromSmiles(smile)
        if mol == None:
            print(i, smile)
            invalid += 1
            qed = -1
            plogp = -999
            smile2 = 'N/A'
            results.append((qed, plogp, smile, smile2))
            f.write('{},{},{},{}\n'.format(qed, plogp, smile, smile2))
            continue

        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)  # 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'
        try:
            qed = env.qed(mol)
        except ValueError as e:
            bad_qed += 1
            qed = -1
            print(i+1, Chem.MolToSmiles(mol, isomericSmiles=True), ' error in qed')

        try:
            plogp = env.penalized_logp(mol)
        except RuntimeError as e:
            bad_plogp += 1
            plogp = -999
            print(i+1, Chem.MolToSmiles(mol, isomericSmiles=True), ' error in penalized_log')

        results.append((qed, plogp, smile, smile2))
        f.write('{},{},{},{}\n'.format(qed, plogp, smile, smile2))
        f.flush()
    f.close()


    results.sort(key=lambda tup: tup[0], reverse=True)
    fv = filename.split('.')
    f = open(fv[0]+'_sortedByQED.'+fv[1], "w")  # append mode
    for r in results:
        qed, plogp, smile, smile2 = r
        f.write('{},{},{},{}\n'.format(qed, plogp, smile, smile2))
        f.flush()
    f.close()

    results.sort(key=lambda tup: tup[1], reverse=True)
    fv = filename.split('.')
    f = open(fv[0] + '_sortedByPlogp.' + fv[1], "w")  # append mode
    for r in results:
        qed, plogp, smile, smile2 = r
        f.write('{},{},{},{}\n'.format(qed, plogp, smile, smile2))
        f.flush()
    f.close()

    print('Dump done!')
    print('Total: {}\t Invalid: {}\t bad_plogp: {} \t bad_qed: {}\n'.format(total, invalid, bad_plogp, bad_qed))


def test_smiles_to_tensor():
    mol_smiles = 'CC(=O)c1ccc(S(=O)(=O)N2CCCC[C@H]2C)cc1'  # 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'  # 'CCOC1=C(N)OC=C1' #'CCC1COC(C)CO1' # 'CCC1=NNN=C1CC' #'CCCC1=C(O)C=CO1' # 'CCCCC1=NC=NN1' # 'CCCNC1=COC=C1'  # or 'CCCNc1ccoc1' same results for smile # 'CCCNC1=COC=C1' #'CCCNC1=CC=CO1' #'CC1=C2C(=O)N(C)C12'
    mm = Chem.MolFromSmiles(mol_smiles)
    Chem.Kekulize(mm, clearAromaticFlags=True)  # use this after mol from simles
    print(Chem.MolToSmiles(mm))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCCC2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=True))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=True))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCCC2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=False))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=False))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCCC2C)C=C1
    print('Chem.AddHs(mm)')
    Chem.AddHs(mm)
    Chem.SanitizeMol(mm, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    print(Chem.MolToSmiles(mm))  # CC(=O)C1C=CC(=CC=1)S(=O)(=O)N1CCCCC1C
    print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=True))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=True))  # CC(=O)C1C=CC(=CC=1)S(=O)(=O)N1CCCCC1C
    print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=False))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=False))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCCC2C)C=C1

    # atoms == atoms2 == atoms3
    bond, atoms = smiles_to_adj('CC(=O)c1ccc(S(=O)(=O)N2CCCC[C@H]2C)cc1', 'zinc250k')
    print(atoms.max(2)[1])
    bond2, atoms2 = smiles_to_adj('CC(=O)C1=CC=C(S(=O)(=O)N2CCCCC2C)C=C1', 'zinc250k')
    print(atoms2.max(2)[1], (bond == bond2).all(), (atoms == atoms2).all())
    bond3, atoms3 = smiles_to_adj('CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1', 'zinc250k')
    print(atoms3.max(2)[1], (bond == bond3).all(), (atoms == atoms3).all())


def test_property_of_smile_vs_tensor(data_name, atomic_num_list):
    mol_smiles = 'COC1=CC=C(C2=CC(C3=CC=CC=C3)=CC(C3=CC=C(Br)C=C3)=[O+]2)C=C1'  # 'CC(=O)c1ccc(S(=O)(=O)N2CCCC[C@H]2C)cc1'  #'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'  # 'CCOC1=C(N)OC=C1' #'CCC1COC(C)CO1' # 'CCC1=NNN=C1CC' #'CCCC1=C(O)C=CO1' # 'CCCCC1=NC=NN1' # 'CCCNC1=COC=C1'  # or 'CCCNc1ccoc1' same results for smile # 'CCCNC1=COC=C1' #'CCCNC1=CC=CO1' #'CC1=C2C(=O)N(C)C12'
    mm = Chem.MolFromSmiles(mol_smiles)
    plogp = env.penalized_logp(mm)
    qed = env.qed(mm)
    print('{}: plogp: {}\tqed: {}'.format(mol_smiles, plogp, qed))

    adj, x = smiles_to_adj(mol_smiles, data_name=data_name)
    rev_mol_smiles = adj_to_smiles(adj, x, atomic_num_list)
    mm2 = Chem.MolFromSmiles(rev_mol_smiles[0])
    plogp = env.penalized_logp(mm2)
    qed = env.qed(mm2)
    print('{}: plogp: {}\tqed: {}'.format(rev_mol_smiles[0], plogp, qed))

    Chem.Kekulize(mm)  # , clearAromaticFlags=True)  # use this after mol from simles
    plogp = env.penalized_logp(mm)
    qed = env.qed(mm)
    print('plogp: {}\tqed: {}'.format(plogp, qed))
    mm3 = Chem.MolFromSmiles(Chem.MolToSmiles(mm))
    plogp = env.penalized_logp(mm3)
    qed = env.qed(mm3)
    print('plogp: {}\tqed: {}'.format(plogp, qed))
    print(Chem.MolToSmiles(mm))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCCC2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=True))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=True))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCCC2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=False))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=False))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCCC2C)C=C1

    print('Chem.AddHs(mm)')
    Chem.AddHs(mm)
    # Chem.SanitizeMol(mm, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    plogp = env.penalized_logp(mm)
    qed = env.qed(mm)
    print('plogp: {}\tqed: {}'.format(plogp, qed))
    print(Chem.MolToSmiles(mm))  # CC(=O)C1C=CC(=CC=1)S(=O)(=O)N1CCCCC1C
    print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=True))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=True))  # CC(=O)C1C=CC(=CC=1)S(=O)(=O)N1CCCCC1C
    print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=False))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1
    print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=False))  # CC(=O)C1=CC=C(S(=O)(=O)N2CCCCC2C)C=C1

    # atoms == atoms2 == atoms3
    bond, atoms = smiles_to_adj('COC1=CC=C(C2=CC(C3=CC=CC=C3)=CC(C3=CC=C(Br)C=C3)=[O+]2)C=C1', 'zinc250k')
    print(atoms.max(2)[1])
    bond2, atoms2 = smiles_to_adj('COC1C=CC(=CC=1)C1=CC(=CC(=[O+]1)C1C=CC(Br)=CC=1)C1C=CC=CC=1', 'zinc250k')
    print(atoms2.max(2)[1], (bond == bond2).all(), (atoms == atoms2).all())
    # bond3, atoms3 = smiles_to_adj('CC(=O)C1=CC=C(S(=O)(=O)N2CCCC[C@H]2C)C=C1', 'zinc250k')
    # print(atoms3.max(2)[1], (bond==bond3).all(), (atoms==atoms3).all())


def find_top_score_smiles(model, device, data_name, property_name, train_prop, topk, atomic_num_list, debug):
    start_time = time.time()
    if property_name == 'qed':
        col = 0
    elif property_name == 'plogp':
        col = 1
    print('Finding top {} score'.format(property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[col], reverse=True)  # qed, plogp, smile
    result_list = []
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, topk, time.time() - start_time))
        qed, plogp, smile = r
        results, ori = optimize_mol(model, property_model, smile, device, sim_cutoff=0, lr=.005, num_iter=100,
                                      data_name=data_name, atomic_num_list=atomic_num_list,
                                      property_name=property_name, random=False, debug=debug)
        result_list.extend(results)  # results: [(smile2, property, sim, smile1), ...]

    result_list.sort(key=lambda tup: tup[1], reverse=True)

    # check novelty
    train_smile = set()
    for i, r in enumerate(train_prop_sorted):
        qed, plogp, smile = r
        train_smile.add(smile)
        mol = Chem.MolFromSmiles(smile)
        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)
        train_smile.add(smile2)

    result_list_novel = []
    for i, r in enumerate(result_list):
        smile, score, sim, smile_original = r
        if smile not in train_smile:
            result_list_novel.append(r)

    # dump results
    f = open(property_name + '_discovered_sorted.csv', "w")
    for r in result_list_novel:
        smile, score, sim, smile_original = r
        f.write('{},{},{},{}\n'.format(score, smile, sim, smile_original))
        f.flush()
    f.close()
    print('Dump done!')


def constrain_optimization_smiles(model, device, data_name, property_name, train_prop, topk,
                                  atomic_num_list, debug, sim_cutoff=0.0):
    start_time = time.time()
    if property_name == 'qed':
        col = 0
    elif property_name == 'plogp':
        col = 1

    print('Constrained optimization of {} score'.format(property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[col]) #, reverse=True)  # qed, plogp, smile
    result_list = []
    nfail = 0
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, topk, time.time() - start_time))
        qed, plogp, smile = r
        results, ori = optimize_mol(model, property_model, smile, device, sim_cutoff=sim_cutoff, lr=.005, num_iter=100,
                                    data_name=data_name, atomic_num_list=atomic_num_list,
                                    property_name=property_name, random=False, debug=debug)
        if len(results) > 0:
            smile2, property2, sim, _ = results[0]
            plogp_delta = property2 - plogp
            if plogp_delta >= 0:
                result_list.append((smile2, property2, sim, smile, qed, plogp, plogp_delta))
            else:
                nfail += 1
                print('Failure:{}:{}'.format(i, smile))
        else:
            nfail += 1
            print('Failure:{}:{}'.format(i, smile))

    df = pd.DataFrame(result_list,
                      columns=['smile_new', 'prop_new', 'sim', 'smile_old', 'qed_old', 'plogp_old', 'plogp_delta'])

    print(df.describe())
    df.to_csv(property_name+'_constrain_optimization.csv', index=False)
    print('Dump done!')
    print('nfail:{} in total:{}'.format(nfail, topk))
    print('success rate: {}'.format((topk-nfail)*1.0/topk))


def plot_top_qed_mol():
    import cairosvg
    filename = 'qed_discovered_sorted_bytop2k.csv'
    df = pd.read_csv(filename)
    vmol = []
    vlabel = []
    for index, row in df.head(n=25).iterrows():
        score, smile, sim, smile_old = row
        vmol.append(Chem.MolFromSmiles(smile))
        vlabel.append('{:.3f}'.format(score))

    svg = Draw.MolsToGridImage(vmol, legends=vlabel, molsPerRow=5, #5,
                               subImgSize=(120, 120), useSVG=True)  # , useSVG=True

    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to="top_qed2.pdf")
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to="top_qed2.png")
    # print('Dump {}.png/pdf done'.format(filepath))

    # img = Draw.MolsToGridImage(vmol, legends=vlabel, molsPerRow=5,
    #                            subImgSize=(300, 300), useSVG=True)
    # print(img)


def plot_mol_constraint_opt():
    import cairosvg
    vsmiles = ['O=C(NCc1ccc2c3c(cccc13)C(=O)N2)c1ccc(F)cc1',
               'O=C(NCC1=Cc2c[nH]c(=O)c3cccc1c23)c1ccc(F)cc1']
    vmol = [Chem.MolFromSmiles(s) for s in vsmiles]
    vplogp = ['{:.2f}'.format(env.penalized_logp(mol)) for mol in vmol]

    # vhighlight = [vmol[0].GetSubstructMatch(Chem.MolFromSmiles('C2=C1C=CC=C3C1=C(C=C2)NC3')),
    #               vmol[1].GetSubstructMatch(Chem.MolFromSmiles('C4=CC6=C5C4=CC=CC5=C[N](=C6)[H]'))]
    svg = Draw.MolsToGridImage(vmol, legends=vplogp, molsPerRow=2,
                               subImgSize=(250, 100), useSVG=True)
                               #highlightAtoms=vhighlight)  # , useSVG=True

    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to="copt2.pdf")
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to="copt2.png")


def plot_mol_matrix():
    import cairosvg
    import seaborn as sns
    import matplotlib.pyplot as plt
    smiles = 'CN(C)C(=N)NC(=N)N'  #'CC(C)NC1=CC=CO1'  #'CC1=C(SC(=C1)C(=O)NCC2=NOC=C2)Br'
    bond, atoms = smiles_to_adj(smiles, 'qm9')
    bond = bond[0]
    atoms = atoms[0]

    # def save_mol_png(mol, filepath, size=(100, 100)):
    #     Draw.MolToFile(mol, filepath, size=size)

    Draw.MolToImageFile(Chem.MolFromSmiles(smiles), 'mol.pdf')
    # save_mol_png(Chem.MolFromSmiles(smiles), 'mol.png')
    svg = Draw.MolsToGridImage([Chem.MolFromSmiles(smiles)], legends=[], molsPerRow=1,
                               subImgSize=(250, 250), useSVG=True)
    # highlightAtoms=vhighlight)  # , useSVG=True

    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to="mol.pdf")
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to="mol.png")

    # sns.set()
    # ax = sns.heatmap(1-atoms)
    # with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(2, 3.4))
    # sns.palplot(sns.diverging_palette(240, 10, n=9))
    ax = sns.heatmap(atoms, linewidths=.5, ax=ax, annot_kws={"size": 18}, cbar=False,
                     xticklabels=False, yticklabels=False, square=True, cmap="vlag", vmin=-1, vmax=1, linecolor='black')
    # ,cmap=sns.diverging_palette(240, 10, n=9)) #"YlGnBu"  , square=True

    plt.show()
    fig.savefig('atom.pdf')
    fig.savefig('atom.png')


    for i, x in enumerate(bond):
        fig, ax = plt.subplots(figsize=(5, 5))
        # sns.palplot(sns.diverging_palette(240, 10, n=9))
        ax = sns.heatmap(x, linewidths=.5,  ax=ax,   annot_kws={"size": 18}, cbar=False,
                         xticklabels=False, yticklabels=False, square=True, cmap="vlag", vmin=-1, vmax=1, linecolor='black')
                              # ,cmap=sns.diverging_palette(240, 10, n=9)) #"YlGnBu"  , square=True

        plt.show()
        fig.savefig('bond{}.pdf'.format(i))
        fig.savefig('bond{}.png'.format(i))


if __name__ == '__main__':
    # plot_mol()
    # plot_mol_constraint_opt()
    # plot_mol_matrix()
    # plot_top_qed_mol()
    # exit(-1)
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='./results', required=True)
    parser.add_argument("--data_dir", type=str, default='../data')
    parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'],
                        help='dataset name')
    parser.add_argument("--snapshot_path", "-snapshot", type=str, required=True)
    parser.add_argument("--hyperparams_path", type=str, default='moflow-params.json', required=True)
    parser.add_argument("--property_model_path", type=str, default=None)
    # parser.add_argument('--molecule_file', type=str, default='qm9_relgcn_kekulized_ggnp.npz',
    #                     help='path to molecule dataset')

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-5,
                        help='L2 norm for the parameters')
    parser.add_argument('--hidden', type=str, default="",
                        help='Hidden dimension list for output regression')
    parser.add_argument('-x', '--max_epochs', type=int, default=5, help='How many epochs to run in total?')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU Id to use')

    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--img_format", type=str, default='svg')
    parser.add_argument("--property_name", type=str, default='qed', choices=['qed', 'plogp'])
    parser.add_argument('--additive_transformations', type=strtobool, default=False,
                        help='apply only additive coupling layers')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the gaussian distributions')

    parser.add_argument('--topk', type=int, default=500, help='Top k smiles as seeds')
    parser.add_argument('--debug', type=strtobool, default='true', help='To run optimization with more information')

    parser.add_argument("--sim_cutoff", type=float, default=0.00)
    #
    parser.add_argument('--topscore', action='store_true', default=False, help='To find top score')
    parser.add_argument('--consopt', action='store_true', default=False, help='To do constrained optimization')

    args = parser.parse_args()

    # Device configuration
    device = -1
    if args.gpu >= 0:
        # device = args.gpu
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    property_name = args.property_name.lower()
    # chainer.config.train = False
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)  # Load moflow model

    if args.hidden in ('', ','):
        hidden = []
    else:
        hidden = [int(d) for d in args.hidden.strip(',').split(',')]
    print('Hidden dim for output regression: ', hidden)
    property_model = MoFlowProp(model, hidden)
    # model.eval()  # Set model for evaluation

    if args.data_name == 'qm9':
        atomic_num_list = [6, 7, 8, 9, 0]
        transform_fn = transform_qm9.transform_fn
        valid_idx = transform_qm9.get_val_ids()
        molecule_file = 'qm9_relgcn_kekulized_ggnp.npz'
        # smile_cvs_to_property('qm9')
    elif args.data_name == 'zinc250k':
        atomic_num_list = zinc250_atomic_num_list
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        valid_idx = transform_zinc250k.get_val_ids()
        molecule_file = 'zinc250k_relgcn_kekulized_ggnp.npz'
        # smile_cvs_to_property('zinc250k')
    else:
        raise ValueError("Wrong data_name{}".format(args.data_name))

    # dataset = NumpyTupleDataset(os.path.join(args.data_dir, molecule_file), transform=transform_fn)  # 133885
    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, molecule_file), transform=transform_fn)

    print('Load {} done, length: {}'.format(os.path.join(args.data_dir, molecule_file), len(dataset)))
    assert len(valid_idx) > 0
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 224568 = 249455 - 24887
    n_train = len(train_idx)  # 120803 zinc: 224568
    train = torch.utils.data.Subset(dataset, train_idx)  # 120803
    test = torch.utils.data.Subset(dataset, valid_idx)  # 13082  not used for generation

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size)

    # print("loading hyperparamaters from {}".format(hyperparams_path))

    if args.property_model_path is None:
        print("Training regression model over molecular embedding:")
        prop_list = load_property_csv(args.data_name, normalize=True)
        train_prop = [prop_list[i] for i in train_idx]
        test_prop = [prop_list[i] for i in valid_idx]
        print('Prepare data done! Time {:.2f} seconds'.format(time.time() - start))
        property_model_path = os.path.join(args.model_dir, '{}_model.pt'.format(property_name))
        property_model = fit_model(property_model, atomic_num_list, train_dataloader, train_prop, device,
                                   property_name=property_name, max_epochs=args.max_epochs,
                                   learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        print("saving {} regression model to: {}".format(property_name, property_model_path))
        torch.save(property_model, property_model_path)
        print('Train and save model done! Time {:.2f} seconds'.format(time.time() - start))
    else:
        print("Loading trained regression model for optimization")
        prop_list = load_property_csv(args.data_name, normalize=False)
        train_prop = [prop_list[i] for i in train_idx]
        test_prop = [prop_list[i] for i in valid_idx]
        print('Prepare data done! Time {:.2f} seconds'.format(time.time() - start))
        property_model_path = os.path.join(args.model_dir, args.property_model_path)
        print("loading {} regression model from: {}".format(property_name, property_model_path))
        device = torch.device('cpu')
        property_model = torch.load(property_model_path, map_location=device)
        print('Load model done! Time {:.2f} seconds'.format(time.time() - start))

        property_model.to(device)
        property_model.eval()

        model.to(device)
        model.eval()

        # mol_smiles = r'C1=CC=C(C=C1)CCCCCCCCCCCCCCCCCCCCCCCCCCCCC'
        # #'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1' #'CC(C)N1N=CC2=N[C@H](c3ccc(-c4ccccn4)cc3)N[C@@H]21'
        # # print(adj_to_smiles(
        # #     np.expand_dims(dataset[33233][1], axis=0), np.expand_dims(dataset[33233][0], axis=0),
        # #     atomic_num_list))  # ['CCCNC1=COC=C1']
        # # mol_smiles ='N=C(N)C(=NO)C1CN1' #'CCOC1=C(N)OC=C1' #'CCC1COC(C)CO1' # 'CCC1=NNN=C1CC' #'CCCC1=C(O)C=CO1' # 'CCCCC1=NC=NN1' # 'CCCNC1=COC=C1'  # or 'CCCNc1ccoc1' same results for smile # 'CCCNC1=COC=C1' #'CCCNC1=CC=CO1' #'CC1=C2C(=O)N(C)C12'
        # # mm = Chem.MolFromSmiles(mol_smiles)
        # # Chem.Kekulize(mm, clearAromaticFlags=True)  # use this after mol from simles
        # # print(Chem.MolToSmiles(mm))  # CCCNc1ccoc1
        # # print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=True))  # CCCNc1ccoc1
        # # print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=True)) # CCCNc1ccoc1
        # # print(Chem.MolToSmiles(mm, isomericSmiles=True, canonical=False)) # CCCNc1ccoc1
        # # print(Chem.MolToSmiles(mm, isomericSmiles=False, canonical=False)) # CCCNc1ccoc1
        #
        # results, start = optimize_mol(model, property_model, mol_smiles, device, sim_cutoff=0, lr=0.01, num_iter=100,
        #                    data_name=args.data_name, atomic_num_list=atomic_num_list,
        #                               property_name=property_name, random=False)
        #
        # print(start)
        # print(results)

        if args.topscore:
            print('Finding top score:')
            find_top_score_smiles(model, device, args.data_name, property_name, train_prop, args.topk, atomic_num_list, args.debug)

        if args.consopt:
            print('Constrained optimization:')
            constrain_optimization_smiles(model, device, args.data_name, property_name, train_prop, args.topk,   # train_prop
                                      atomic_num_list, args.debug, sim_cutoff=args.sim_cutoff)

        print('Total Time {:.2f} seconds'.format(time.time() - start))

