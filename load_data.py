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

def load_molecule_file(filepath):
    """
    加载单个分子npz文件
    参数:
        filepath: .npz文件路径
    返回:
        PyG Data对象
    """
    loaded = np.load(filepath, allow_pickle=True)
    mol_dict = {key: loaded[key] for key in loaded.files}

    # 创建Data对象
    data = Data(
        x=torch.from_numpy(mol_dict['x']) if mol_dict['x'] is not None else None,
        edge_index=torch.from_numpy(mol_dict['edge_index']) if mol_dict['edge_index'] is not None else None,
        edge_attr=torch.from_numpy(mol_dict['edge_attr']) if 'edge_attr' in mol_dict and mol_dict[
            'edge_attr'] is not None else None,
        y=torch.from_numpy(mol_dict['y']) if mol_dict['y'] is not None else None
    )

    # 添加可选属性
    if 'angle_index' in mol_dict and mol_dict['angle_index'] is not None:
        data.angle_index = torch.from_numpy(mol_dict['angle_index'])
    if 'angle_attr' in mol_dict and mol_dict['angle_attr'] is not None:
        data.angle_attr = torch.from_numpy(mol_dict['angle_attr'])
    if 'sub_f' in mol_dict and mol_dict['sub_f'] is not None:
        data.sub_f = torch.from_numpy(mol_dict['sub_f'])
    if 'pub_f' in mol_dict and mol_dict['pub_f'] is not None:
        data.pub_f = torch.from_numpy(mol_dict['pub_f'])
    if 'maccs_f' in mol_dict and mol_dict['maccs_f'] is not None:
        data.maccs_f = torch.from_numpy(mol_dict['maccs_f'])
    if 'smiles' in mol_dict:
        data.smiles = str(mol_dict['smiles'])

    return data


def load_all_molecules(directory):
    """
    加载目录下的所有分子文件
    参数:
        directory: 包含.npz文件的目录
    返回:
        PyG Data对象列表
    """
    data_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            filepath = os.path.join(directory, filename)
            try:
                data = load_molecule_file(filepath)
                data_list.append(data)
            except Exception as e:
                print(f"加载文件 {filename} 时出错: {str(e)}")
    return data_list


# 保存所有分子
# (上面的主代码已经包含)

# 加载所有分子
file_names = ["FreeSolv", "Lipo", "PCBA", "MUV", "ToxCast", "ClinTox", "tox21", "esol", "bbbp", "bace", "hiv", "sider"]
dataset_name = file_names[8]
loaded_data = load_all_molecules(f"data/{dataset_name}_molecules")
print(f"成功加载 {len(loaded_data)} 个分子")

# 加载单个分子示例
sample_file = os.path.join(f"data/{dataset_name}_molecules", os.listdir(f"data/{dataset_name}_molecules")[0])
sample_data = load_molecule_file(sample_file)
print("示例分子:", sample_data.smiles, len(sample_data))