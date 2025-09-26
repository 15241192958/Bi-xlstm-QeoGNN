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
from torcheval.metrics.functional import r2_score
import optuna
import torch
import numpy as np
from time import time
import copy
import matplotlib.pyplot as plt


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

class MyModifiedDataset(InMemoryDataset):
    def __init__(self, original_dataset, modified_data_list):
        super().__init__(original_dataset.root)

        self.data, self.slices = self.collate(modified_data_list)


class GeoGraphBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, angle_dim, heads=1, dropout=0.6):
        super(GeoGraphBlock, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim,add_self_loops=False)
        self.relu = torch.nn.ReLU()
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout,
                               edge_dim=hidden_channels * heads,add_self_loops=False)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout,
                               edge_dim=hidden_channels * heads,add_self_loops=False)

        # self.conv_single = GATv2Conv(in_channels, in_channels, heads = 1, dropout = dropout, edge_dim = edge_dim)
        self.linear_1 = torch.nn.Linear(in_channels, 1)

        self.H_conv1 = GATv2Conv(edge_dim, hidden_channels, heads=heads, dropout=dropout, edge_dim=angle_dim,add_self_loops=False)
        self.H_conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout,
                                 edge_dim=angle_dim,add_self_loops=False)
        self.bond_encoder = nn.Linear(edge_dim, hidden_channels)
        self.G_normalize = nn.BatchNorm1d(hidden_channels)

    def forward(self, data):
        x, bond_index, bond_attr = data.x.float(), data.edge_index, data.edge_attr.float()
        angle_index, anlge_attr = data.angle_index, data.angle_attr

        x = self.conv1(x, bond_index, edge_attr=bond_attr)
        x = self.relu(x)
        # print(bond_attr.shape)
        # print(angle_index.shape)
        # print(torch.max(angle_index).item())
        # print(anlge_attr.shape)
        # exit()
        bond_attr = self.H_conv1(bond_attr, angle_index, anlge_attr)

        x = self.conv2(x, bond_index, edge_attr=bond_attr)
        bond_attr = self.H_conv2(bond_attr, angle_index, anlge_attr)
        x = self.relu(x)

        x = self.conv3(x, bond_index, edge_attr=bond_attr)
        x = self.relu(x)

        x = pygnn.pool.global_mean_pool(x, data.batch)
        x = self.G_normalize(x)

        return x


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        channel_weights = self.sigmoid(y)
        fused_features = torch.sum(x * channel_weights, dim=1)
        return fused_features
class FingerprintBlock(torch.nn.Module):
    def __init__(self, in_channels_list, hidden_channels,dropout,num_finger=3):
        super(FingerprintBlock, self).__init__()
        self.num_finger=num_finger
        self.lstm0 = nn.LSTM(input_size=in_channels_list[0],hidden_size=hidden_channels,num_layers=1,dropout=dropout)
        self.lstm1 = nn.LSTM(input_size=in_channels_list[1],hidden_size=hidden_channels,num_layers=1,dropout=dropout)
        self.lstm2 = nn.LSTM(input_size=in_channels_list[2],hidden_size=hidden_channels,num_layers=1,dropout=dropout)
        self.finger_attention=eca_layer(channel=num_finger)
        self.F_normalize=nn.BatchNorm1d(hidden_channels)
    def forward(self,data):
        sub_rep,_=self.lstm0(data.sub_f)
        pub_rep,_=self.lstm1(data.pub_f)
        maccs_rep,_=self.lstm2(data.maccs_f)
        fusion_finger=torch.cat([sub_rep[:,None,:],pub_rep[:,None,:],maccs_rep[:,None,:]],dim=1)
        final_finger=self.finger_attention(fusion_finger)
        final_finger=self.F_normalize(final_finger)
        return final_finger


class AttentionModel(nn.Module):
    def __init__(self, emb_dim, nhid, heads):
        super().__init__()
        fp_feats_num = emb_dim
        graph_feats_num = emb_dim
        self.heads = heads
        self.linear_fp = nn.Linear(fp_feats_num, nhid)
        self.q_r = nn.Parameter(torch.rand(nhid, self.heads))
        self.linear_graph = nn.Linear(graph_feats_num, nhid)
        self.q_s = nn.Parameter(torch.rand(nhid, self.heads))
        self.att_fusion = nn.Linear(self.heads * nhid, nhid)
        self.sim_fusion = nn.Linear(fp_feats_num + graph_feats_num, nhid)

    def forward(self, fp, graph):
        fp_feats, graph_feats = fp, graph

        fp_feats = torch.tanh(self.linear_fp(fp_feats))
        fp_alpha = torch.matmul(fp_feats, self.q_r)

        graph_feats = torch.tanh(self.linear_graph(graph_feats))
        graph_alpha = torch.matmul(graph_feats, self.q_s)

        alpha = torch.exp(fp_alpha) + torch.exp(graph_alpha)
        fp_alpha = torch.exp(fp_alpha) / alpha
        graph_alpha = torch.exp(graph_alpha) / alpha
        fusion_x = torch.cat(
            [fp_alpha[:, i].view(-1, 1) * fp_feats + graph_alpha[:, i].view(-1, 1) * graph_feats for i in
             range(self.heads)],
            dim=1)
        fusion_x = self.att_fusion(fusion_x)
        return fusion_x


class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, angle_dim, finger_dims, num_classes, heads=1,
                 dropout=0.2):
        super(Model, self).__init__()
        self.gnn = GeoGraphBlock(in_channels, hidden_channels, edge_dim, angle_dim, heads=heads, dropout=dropout)
        self.lstm = FingerprintBlock(finger_dims, hidden_channels, dropout=dropout)
        self.attention_fusion = AttentionModel(hidden_channels, hidden_channels, heads=heads)
        # 修改输出层为分类
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_classes)
            # torch.nn.Sigmoid()  # 多标签分类需要sigmoid
        )

    def forward(self, data):
        # 强制所有数据到同一设备
        data = data.to(device)
        graph_embedding = self.gnn(data)
        finger_embedding = self.lstm(data)
        final_emb = self.attention_fusion(finger_embedding, graph_embedding)
        logits = self.decoder(final_emb)  # 输出原始logits
        # print(f"模型输出形状: {final_emb.shape}")  # 应为[batch_size, num_labels]
        return logits


def train(model, loader, optimizer, epoch_no):
    model.train()
    epoch_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        logits = model(batch)

        # 多标签，确保标签是长整型
        # 确保标签形状正确 [batch_size, num_labels]
        targets = batch.y
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)  # 单标签转为多标签格式
        targets = targets.float()

        # 多标签，使用交叉熵损失
        # 使用BCE损失代替CrossEntropy
        # loss = nn.BCELoss()(logits, targets)  # 二元交叉熵
        # 使用BCEWithLogitsLoss（内置Sigmoid且数值稳定）
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, targets)
        # loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        optimizer.step()

        # 修正准确率计算维度
        predicted = (torch.sigmoid(logits) > 0.5).float()
        # 计算准确率 - 需要阈值处理
        # predicted = (logits > 0.5).float()  # 阈值设为0.5
        correct = (predicted == targets).all(dim=1).sum().item()  # 多标签匹配
        total_correct += correct
        total_samples += targets.size(0)

        epoch_loss += loss.item()

    accuracy = total_correct / max(total_samples, 1)  # 避免除零
    return epoch_loss / max(len(loader), 1), accuracy


@torch.no_grad()
def test(model, loader):
    model.eval()
    epoch_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            # targets = batch.y.long().squeeze()
            # 多标签同样转为float
            targets = batch.y
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            targets = targets.float()
            # loss = nn.CrossEntropyLoss()(logits, targets)
            # epoch_loss += loss.item()
            #
            # _, predicted = torch.max(logits, 2)
            # total_correct += (predicted == targets).sum().item()
            # total_samples += targets.size(0)
            # 使用相同的损失函数
            loss = nn.BCEWithLogitsLoss()(logits, targets)
            # loss = nn.BCELoss()(logits, targets)
            predicted = (torch.sigmoid(logits) > 0.5).float()
            # predicted = (logits > 0.5).float()
            total_correct += (predicted == targets).all(dim=1).sum().item()
            total_samples += targets.size(0)
            epoch_loss += loss.item()

    accuracy = total_correct / max(total_samples, 1)
    return epoch_loss / max(len(loader), 1), accuracy
def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    hidden_dimensions = trial.suggest_int('hidden_dim', 30, 60, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,drop_last=True)

    model = Model(
        in_dimensions,
        hidden_dimensions,
        edge_dimensions,
        anlge_dimensions,
        finger_dims,
        num_classes,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_epochs = 50
    all_val_loss = []
    all_val_acc = []
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, epoch)
        val_loss, val_acc = test(model, val_loader)
        all_val_acc.append(val_acc)  # 跟踪验证准确率

        trial.report(val_acc, epoch)  # 使用准确率作为报告指标

        if trial.should_prune():
            raise optuna.TrialPruned()
    print("最大的准确率为",max(all_val_acc), all_val_acc)
    return -max(all_val_acc)  # Optuna最小化目标，所以取负准确率
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# check if GPU is available and detectable. cpu is ok for this homework.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cuda')
# device=torch.device('cpu')
print('Using device:', device)
file_names = ["FreeSolv", "Lipo", "PCBA", "MUV", "ToxCast", "ClinTox", "tox21", "esol", "bbbp", "bace", "hiv", "sider"]
dataset_name = file_names[3]
num_classes =17
# 2：pcba 128类 ，标签有问题,暂时改成0，待解决；3：muv  17类,标签有问题,同pcba；4：toxcast 617类，报错,同pcba； ；5：clintox 2类,ok；6：tox21 12类，标签有问题,同pcba；11：sider 27类  数据没有下载完成

dataset_untransformed = MoleculeNet('./data', dataset_name)
dataset = MoleculeNet('./data', dataset_name, pre_transform = OneHotTransform(dataset_untransformed))


from compound_tools import mol_to_geognn_graph_data_MMFF3d
add_dataset = []
# for data in dataset:
#     try:
#         smile = data.smiles
#         mol = AllChem.MolFromSmiles(smile)
#         new_mol = Chem.AddHs(mol)
#         res = AllChem.EmbedMultipleConfs(new_mol)
#         res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
#         new_mol = Chem.RemoveHs(new_mol)
#
#         mol_3d_info = mol_to_geognn_graph_data_MMFF3d(new_mol)
#
#         if mol_3d_info['bond_angle'].shape[0] != 0:
#             data.angle_index = torch.tensor(mol_3d_info['BondAngleGraph_edges'].T, dtype=torch.int64)
#             data.angle_attr = torch.tensor(mol_3d_info ['bond_angle'], dtype=torch.float32).unsqueeze(1)
#             finger_prints = create_fingerprint(data.smiles)
#             data.sub_f = finger_prints[0]
#             data.pub_f = finger_prints[1]
#             data.maccs_f = finger_prints[2]
#             add_dataset.append(data)
#             print(data.size)
#         else:
#             pass
#     except:
#         pass
add_dataset = load_all_molecules(f"data/{dataset_name}_molecules")
print(f"成功加载 {len(add_dataset)} 个分子")

#
# # 加载单个分子示例
# sample_file = os.path.join(f"data/{dataset_name}_molecules", os.listdir(f"data/{dataset_name}_molecules")[0])
# sample_data = load_molecule_file(sample_file)
# print("示例分子:", sample_data.smiles)
dataset = MyModifiedDataset(dataset, add_dataset).to(device)


# 数据加载后验证
sample = dataset[0]
print(f"样本标签形状: {sample.y.shape}")  # 应为[num_labels]
print(f"标签值示例: {sample.y}")  # 值应在0-1之间

# 验证所有标签
for data in dataset:
    if torch.isnan(data.y).any():
        print(f"发现nan")
        # data.y = torch.nan_to_num(data.y, nan=0.0)
        data.y = data.y.replace(0, -1)
        data.y = data.y.fillna(0)
    assert data.y.dim() in [1, 2], "标签必须是1D或2D"
    assert torch.all(data.y >= -1) and torch.all(data.y <= 1), "标签值必须在0-1范围内"


finger_dims = [int(dataset[0].sub_f.shape[1]),int(dataset[0].pub_f.shape[1]),int(dataset[0].maccs_f.shape[1])]
dataset_size = len(dataset)
train_size = int(0.6 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Train dataset: {len(train_dataset)} graphs")
print(f"Val dataset: {len(val_dataset)} graphs")
print(f"Test dataset: {len(test_dataset)} graphs")

in_dimensions = dataset[0].x.shape[1]
edge_dimensions = dataset[0].edge_attr.shape[1]
anlge_dimensions = dataset[0].angle_attr.shape[1]
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(objective, n_trials=5)

print("Best trial:")
trial = study.best_trial
print(f"  Value (min val_rmse): {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params = study.best_params

final_params = {
    'batch_size': best_params['batch_size'],
    'lr': best_params['lr'],
    'hidden_dim': best_params['hidden_dim'],
    'dropout': best_params['dropout'],
    'weight_decay': best_params['weight_decay'],
    'num_epochs': 50
}

final_train_loader = DataLoader(
    train_dataset,
    batch_size=final_params['batch_size'],
    shuffle=True,
    drop_last=True  # Drop last batch if smaller than batch_size
)

final_val_loader = DataLoader(
    val_dataset,
    batch_size=final_params['batch_size'],
    shuffle=False
)

final_model = Model(
    in_dimensions,
    final_params['hidden_dim'],
    edge_dimensions,
    anlge_dimensions,
    finger_dims,
    num_classes,
    dropout=final_params['dropout']
).to(device)

final_optimizer = torch.optim.Adam(
    final_model.parameters(),
    lr=final_params['lr'],
    weight_decay=final_params['weight_decay']
)

final_train_loss = []
final_train_acc = []
final_val_loss = []
final_val_acc = []
best_val_rmse = float('inf')
best_model_weights = None

patience = 20
no_improve = 0

for epoch in range(final_params['num_epochs']):

    final_model.train()
    train_loss, train_acc = train(final_model, final_train_loader, final_optimizer, epoch)
    final_train_loss.append(train_loss)
    final_train_acc.append(train_acc)

    final_model.eval()
    with torch.no_grad():
        val_rmse, val_acc = test(final_model, final_val_loader)
    final_val_loss.append(val_rmse)
    final_val_acc.append(val_acc)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model_weights = copy.deepcopy(final_model.state_dict())
        no_improve = 0

        torch.save(final_model.state_dict(), f'best_model_epoch{epoch}.pth')
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f'Early stopping at epoch {epoch}')
        break
# 多标签评价指标还没有开根号
final_model.load_state_dict(best_model_weights)
print('\nFinal Performance:',dataset_name,f"Train dataset: {len(train_dataset)} graphs")
print(f'Train acc: {max(final_train_acc):.4f} | Train loss: {min(final_train_loss):.4f}')
print(f'Val acc: {max(final_val_acc):.4f} | Val loss: {min(final_val_loss):.4f}')


torch.save({
    'model_state_dict': final_model.state_dict(),
    'optimizer_state_dict': final_optimizer.state_dict(),
    'best_params': best_params
}, 'final_model.pth')


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(final_train_loss, label='Train')
plt.plot(final_val_loss, label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(final_train_acc, label='Train')
plt.plot(final_val_acc, label='Validation')
plt.title('acc Score')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.legend()
plt.tight_layout()
plt.show()