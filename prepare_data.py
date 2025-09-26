import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

import networkx as nx

import torch.nn.functional as F

# import torch_geometric.data into environment
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric import nn as pygnn
import torch.nn as nn

from torch_geometric.nn import GATv2Conv


from torch.utils.data import random_split

from torch_geometric.loader import DataLoader

from utils import draw_smiles, my_draw_networkx_edge_labels

from pysmiles import read_smiles #Unused

from tqdm import tqdm

from time import time

from scipy.stats import linregress
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDKFingerprint
from torch_geometric.data import InMemoryDataset
from torch.nn.parameter import Parameter
#Random graphs
import random
from XLSTM import xLSTM

import torch
import numpy as np
from time import time
import copy
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem
import threading
from functools import wraps
class OneHotTransform:
    def __init__(self, untransformed):
        self.full_x = untransformed.x
        self.full_x_slices = untransformed.slices['x']
        self.full_edges = untransformed.edge_attr
        self.full_edge_slices = untransformed.slices["edge_index"]

        self.index = 0
        self.edge_index = 0

    def __call__(self, data: Data):
        tensors = []
        for i in range(self.full_x.shape[1]):
            unique, unique_indices = torch.unique(self.full_x[:, i], return_inverse=True)

            if len(unique) > 2:
                tensors.append(F.one_hot(unique_indices, len(unique)))
            elif len(unique) == 2:
                tensors.append(self.full_x[:, i].reshape(-1, 1))

        new_x = torch.cat(tensors, dim=1)

        start = self.full_x_slices[self.index]
        end = self.full_x_slices[self.index + 1]
        data.x = new_x[start:end]

        tensors = []
        for i in range(self.full_edges.shape[1]):
            unique, unique_indices = torch.unique(self.full_edges[:, i], return_inverse=True)

            if len(unique) > 2:
                tensors.append(F.one_hot(unique_indices, len(unique)))
            elif len(unique) == 2:
                tensors.append(self.full_edges[:, i].reshape(-1, 1))

        new_edges = torch.cat(tensors, dim=1)

        start = self.full_edge_slices[self.index]
        end = self.full_edge_slices[self.index + 1]
        data.edge_attr = new_edges[start:end]

        self.index += 1

        return data

def create_fingerprint(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    substructure_fingerprint = torch.tensor(list(RDKFingerprint(mol)) ,dtype=torch.float32)[None,:]
    pubchem_fingerprint =  torch.tensor(list(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=881)),dtype=torch.float32)[None,:]
    maccs_fingerprint = torch.tensor(list(MACCSkeys.GenMACCSKeys(mol)),dtype=torch.float32)[None,:]
    return substructure_fingerprint,pubchem_fingerprint,maccs_fingerprint

def save_single_molecule(data, save_dir):
    """
    将单个PyG Data对象保存为单独的npz文件
    参数:
        data: PyG Data对象
        save_dir: 保存目录路径
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 使用SMILES作为文件名（替换特殊字符）
    safe_smiles = data.smiles.replace("/", "_").replace("\\", "_")
    filename = os.path.join(save_dir, f"{safe_smiles}.npz")

    # 准备保存的数据字典
    mol_dict = {
        'x': data.x.numpy() if data.x is not None else None,
        'edge_index': data.edge_index.numpy() if data.edge_index is not None else None,
        'edge_attr': data.edge_attr.numpy() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None,
        'y': data.y.numpy() if data.y is not None else None,
        'smiles': data.smiles if hasattr(data, 'smiles') else None,
        'angle_index': data.angle_index.numpy() if hasattr(data,
                                                           'angle_index') and data.angle_index is not None else None,
        'angle_attr': data.angle_attr.numpy() if hasattr(data, 'angle_attr') and data.angle_attr is not None else None,
        'sub_f': data.sub_f.numpy() if hasattr(data, 'sub_f') and data.sub_f is not None else None,
        'pub_f': data.pub_f.numpy() if hasattr(data, 'pub_f') and data.pub_f is not None else None,
        'maccs_f': data.maccs_f.numpy() if hasattr(data, 'maccs_f') and data.maccs_f is not None else None
    }

    # 保存为npz文件
    np.savez_compressed(filename, **mol_dict)
    print(f"已保存: {filename}")


class TimeoutError(Exception):
    pass


def timeout(seconds):
    """Windows-compatible timeout decorator using threading"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            exception = None

            def target():
                nonlocal result, exception
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function timed out after {seconds} seconds")
            if exception is not None:
                raise exception
            return result
        return wrapper
    return decorator

def process_molecule_windows(data, timeout_seconds):
    """Windows-specific molecule processing with timeout"""
    try:
        smile = data.smiles

        # 1. SMILES Parsing (fast operation, no timeout needed)
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print(f"Invalid SMILES: {smile}")
            return None

        # 2. Add Hydrogens (fast operation)
        new_mol = Chem.AddHs(mol)

        # 3. 3D Conformation Generation with timeout
        @timeout(timeout_seconds)
        def generate_3d_conformation():
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=1)
            if not res:
                raise ValueError("3D embedding failed")
            return res

        try:
            generate_3d_conformation()
        except TimeoutError:
            print(f"3D conformation generation timed out for {smile}")
            return None

        # 4. Force Field Optimization with timeout
        @timeout(timeout_seconds)
        def optimize_conformation():
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol, maxIters=2000)
            if res[0][0] != 0:

                raise ValueError("Optimization failed")
            return res

        try:
            optimize_conformation()
        except TimeoutError:
            print(f"Conformation optimization timed out for {smile}")
            return None

        # 5. Remove Hydrogens
        new_mol = Chem.RemoveHs(new_mol)

        # 6. Geometric Features with timeout
        @timeout(timeout_seconds)
        def calculate_geometric_features():
            info = mol_to_geognn_graph_data_MMFF3d(new_mol)
            if info['bond_angle'].shape[0] == 0:
                raise ValueError("No angles found")
            return info

        try:
            mol_3d_info = calculate_geometric_features()
        except TimeoutError:
            print(f"Geometric feature calculation timed out for {smile}")
            return None

        return new_mol, mol_3d_info

    except Exception as e:
        print(f"Error processing {smile}: {str(e)}")
        return None
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# check if GPU is available and detectable. cpu is ok for this homework.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cuda')
# device=torch.device('cpu')
print('Using device:', device)
file_names = ["FreeSolv", "Lipo", "PCBA", "MUV", "ToxCast", "ClinTox", "tox21", "esol", "bbbp", "bace", "hiv", "sider"]
dataset_name = file_names[1]
dataset_untransformed = MoleculeNet('./data', dataset_name)
dataset = MoleculeNet('./data', dataset_name, pre_transform = OneHotTransform(dataset_untransformed))


from compound_tools import mol_to_geognn_graph_data_MMFF3d
add_dataset = []
# 创建保存目录
save_directory = f"data/{dataset_name}_molecules"
os.makedirs(save_directory,  exist_ok=True)
# In your main processing loop:
timeout_seconds = 200  # Adjust as needed
for data in dataset:
    print(data)
    try:
        # smile = data.smiles
        # mol = AllChem.MolFromSmiles(smile)
        # new_mol = Chem.AddHs(mol)
        # res = AllChem.EmbedMultipleConfs(new_mol)
        # res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        # new_mol = Chem.RemoveHs(new_mol)
        #
        # mol_3d_info = mol_to_geognn_graph_data_MMFF3d(new_mol)


        smile = data.smiles
        print(f"Processing {smile}...")
        result = process_molecule_windows(data, timeout_seconds=100)
        if result is None:
            continue

        new_mol, mol_3d_info = result

        if mol_3d_info['bond_angle'].shape[0] != 0:
            print("开始计算描述符")
            data.angle_index = torch.tensor(mol_3d_info['BondAngleGraph_edges'].T, dtype=torch.int64)
            data.angle_attr = torch.tensor(mol_3d_info ['bond_angle'], dtype=torch.float32).unsqueeze(1)
            finger_prints = create_fingerprint(data.smiles)
            data.sub_f = finger_prints[0]
            data.pub_f = finger_prints[1]
            data.maccs_f = finger_prints[2]
            add_dataset.append(data)
            # 保存单个分子文件
            print(data)
            save_single_molecule(data, save_directory)
        else:
            pass
    # except:
    except Exception as e:
        print(f"处理分子 {data.smiles if hasattr(data, 'smiles') else '未知'} 时出错: {str(e)}")

    pass
print(f"所有分子已保存到目录: {save_directory}")
# # 保存处理后的数据
# save_data_dict(add_dataset, f'data/mol_mord_{dataset_name}.npz')