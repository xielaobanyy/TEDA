import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix
from torchvision import transforms, models
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np
import random
import logging
from datetime import datetime
import csv
from sklearn.metrics import cohen_kappa_score
# Experiment parameters
batch_size = 32
threads = 0
lr = 0.005
epochs = 50
log_interval = 10
wdecay = 1e-4
#dataset = '/data1/qsy/gcn/heartnew_zero_arrtributes'
model_name = 'gcn'  # 'gcn', 'unet'
device = 'cuda'  # 'cuda', 'cpu'
visualize = True
shuffle_nodes = False
print('torch', torch.__version__)
loss_fn = nn.CrossEntropyLoss(reduction='sum')
best_acc = 0.0
best_model_path = 'best.pt'
best_confusion_matrix_path = '/data1/qsy/gcn/logs_big/big_eassy/gcn-10.13ssc-Ablation/best_confusion_matrix.png'
results_dir = '/data1/qsy/gcn/logs_big/big_eassy/gcn-10.13ssc-Ablation'
# 确保目录存在
os.makedirs(results_dir, exist_ok=True)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def setup_logger(results_dir):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 文件日志
    fh = logging.FileHandler(os.path.join(results_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台日志（可选，保留原有打印效果）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

logger = setup_logger(results_dir)
logger.info("Logging initialized. Results dir: %s", results_dir)
# Data loader and reader

class DataReader():
    def __init__(self, data_dir, rnd_state=None, use_cont_node_attr=False, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, images_dir=None):
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.images_dir = images_dir
        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState(88) if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr

    def load_data(self):
        files = os.listdir(self.data_dir)
        data = {}
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0], nodes, graphs, fn=lambda s: int(s.strip()))
        data['adj_list'] = self.read_graph_adj(
            list(filter(lambda f: f.find('_A') >= 0, files))[0], 
            list(filter(lambda f: f.find('_edges') >= 0, files))[0], 
            nodes, 
            graphs
        )
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0], line_parse_fn=lambda s: int(float(s.strip()))))

        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0], nodes, graphs, fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            n = np.sum(adj)  # total sum of edges
            assert n % 2 == 0, n
            n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
            if not np.allclose(adj, adj.T):
                print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            features.append(np.array(data['features'][sample_id]))

        features_all = np.concatenate(features)
        features_min = features_all.min()
        features_dim = int(features_all.max() - features_min + 1)  # number of possible values

        features_onehot = []
        for i, x in enumerate(features):
            feature_onehot = np.zeros((len(x), features_dim))
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1
            if self.use_cont_node_attr:
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
            features_onehot.append(feature_onehot)

        if self.use_cont_node_attr:
            features_dim = features_onehot[0].shape[1]

        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets']  # graph class labels
        labels -= np.min(labels)  # to start from 0
        N_nodes_max = np.max(shapes)

        classes = np.unique(labels)
        n_classes = len(classes)
        N_graphs = len(labels)
        #train_ids, test_ids = self.split_ids(np.arange(N_graphs), test_ratio=self.test_ratio, val_ratio=self.val_ratio, rnd_state=self.rnd_state)
        #self.train_indices, self.val_indices = self.split_data(train_ids)  # 注意这里使用train_ids分割为训练和验证集

        assert N_graphs == len(data['adj_list']), "The number of graphs does not match the number of adjacency lists."
        assert N_graphs == len(features_onehot), "The number of graphs does not match the number of feature lists."

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)

        N_graphs = len(labels)  # number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        self.train_indices, self.val_indices, self.test_indices= self.split_data(np.arange(N_graphs), train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

        self.data = {
            'features_onehot': features_onehot,
            'adj_list': data['adj_list'],
            'targets': labels,
            'N_nodes_max': N_nodes_max,
            'n_classes': n_classes,
            'features_dim': features_dim,
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,            
            'test_indices': self.test_indices,
        }
        self.setup_image_paths()
        self.label_dict = self.create_label_dict()
        print("Data loading completed.")

    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs
    
    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

    def read_graph_adj(self, fpath_adj, fpath_edges, nodes, graphs):
        edges = self.parse_txt_file(fpath_adj, line_parse_fn=lambda s: s.strip().split(','))
        edge_weights = self.parse_txt_file(fpath_edges, line_parse_fn=lambda s: s.strip().split(','))

        adj_dict = {graph_id: np.zeros((len(graphs[graph_id]), len(graphs[graph_id]))) for graph_id in graphs}

        for edge in edge_weights:
            # 确保每个边数据都包含三个部分
            if len(edge) != 3:
                print(f"Skipping invalid edge data: {edge} - Incomplete edge information")
                continue
            
            try:
                node1 = int(edge[0]) - 1
                node2 = int(edge[1]) - 1
                weight = float(edge[2])  # 读取权重
            except ValueError as e:
                print(f"Invalid edge data: {edge} - Error: {e}")
                continue
            
            if node1 not in nodes or node2 not in nodes:
                print(f"Skipping edge with nodes not in graph: {edge}")
                continue

            graph_id = nodes[node1]
            if graph_id != nodes[node2]:
                print(f"Skipping edge with nodes in different graphs: {edge}")
                continue
            
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            if len(ind1) != 1 or len(ind2) != 1:
                print(f"Skipping edge with invalid node indices: {edge}")
                continue
            
            adj_dict[graph_id][ind1, ind2] = weight  # 使用权重填充邻接矩阵
            adj_dict[graph_id][ind2, ind1] = weight  # 确保无向图的对称性

        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return adj_list

    def setup_image_paths(self):
        if self.images_dir is not None:
            self.image_paths = []
            self.label_dict = {}
            image_subdirs = ['0', '1', '2', '3', '4']
            for subdir in image_subdirs:
                class_path = os.path.join(self.images_dir, subdir)
                for img_path in glob.glob(os.path.join(class_path, '*.jpg')):
                    img_num = int(os.path.splitext(os.path.basename(img_path))[0])
                    self.image_paths.append(img_path)
                    self.label_dict[img_num] = int(subdir)  # Assume folder name is the true label

            self.image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        else:
            self.image_paths = []

    def create_label_dict(self):
        label_dict = {}
        for path in self.image_paths:
            img_num = int(os.path.splitext(os.path.basename(path))[0])
            # Extract true label from directory name
            true_label = int(os.path.basename(os.path.dirname(path)))
            label_dict[img_num] = true_label
        return label_dict

    def split_ids(self, indices, test_ratio, val_ratio, rnd_state):
        # Shuffle the indices to ensure randomness
        rnd_state.shuffle(indices)
    
        # Calculate the number of samples for the test set
        test_size = int(len(indices) * test_ratio)
        test_indices = indices[:test_size]  # The first `test_size` indices are the test set
        remaining_indices = indices[test_size:]  # Remaining indices for training + validation
    
        # Calculate the number of samples for the validation set
        val_size = int(len(remaining_indices) * val_ratio)
        val_indices = remaining_indices[:val_size]  # The first `val_size` indices are the validation set
        train_indices = remaining_indices[val_size:]  # The rest are the training set
    
        return train_indices, val_indices, test_indices

    def split_data(self, indices, train_ratio=None, val_ratio=None, test_ratio=None):
        if train_ratio is None:
            train_ratio = self.train_ratio
        if val_ratio is None:
            val_ratio = self.val_ratio
        if test_ratio is None:
            test_ratio = self.test_ratio
    
        # Step 1: Shuffle the indices for randomness
        rnd_state = self.rnd_state if self.rnd_state else np.random.RandomState()
        rnd_state.shuffle(indices)
    
        # Step 2: Calculate the sizes of each split
        test_size = int(len(indices) * test_ratio)
        test_indices = indices[:test_size]
        remaining_indices = indices[test_size:]  # remaining for training + validation
    
        val_size = int(len(remaining_indices) * val_ratio)
        val_indices = remaining_indices[:val_size]
        train_indices = remaining_indices[val_size:]
    
        return train_indices, val_indices, test_indices

    
    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(os.path.join(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s.strip()) if line_parse_fn is not None else s.strip() for s in lines]
        return data


class GraphData(torch.utils.data.Dataset):
    def __init__(self, datareader, indices, split, image_paths, label_dict):
        self.datareader = datareader
        self.indices = indices
        self.split = split
        self.image_paths = image_paths
        self.label_dict = label_dict
        self.setup_data()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup_data(self):
        data = self.datareader.data
        self.labels = [self.datareader.data['targets'][i] for i in self.indices]
        self.adj_list = [data['adj_list'][i] for i in self.indices]
        self.features_onehot = [data['features_onehot'][i] for i in self.indices]
        self.N_nodes_max = data['N_nodes_max']
        self.n_classes = data['n_classes']
        self.features_dim = data['features_dim']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        global_index = self.indices[index]
        graph_features = self.features_onehot[index]
        adj_matrix = self.adj_list[index]
        img_num = self.indices[index] + 1
        true_label = self.label_dict[img_num]

        image = self.load_image(global_index)
        graph_features = torch.tensor(graph_features, dtype=torch.float).to(device)
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float).to(device)
        dataset_label = torch.tensor(true_label, dtype=torch.long).to(device)

        graph_data = (graph_features, adj_matrix)

        return graph_data, image, dataset_label, global_index

    def load_image(self, global_index):
        image_path = self.image_paths[global_index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.to(device)


def collate_graphs(batch):
    max_nodes = max(graph_data[0].shape[0] for graph_data, image, label, index in batch)

    graph_features_batch = []
    adj_matrix_batch = []
    labels_batch = []
    images_batch = []
    indices_batch = []

    for (graph_features, adj_matrix), image, label, index in batch:
        num_nodes = graph_features.shape[0]
        pad_size = max_nodes - num_nodes

        padded_features = F.pad(graph_features, (0, 0, 0, pad_size), 'constant', 0)
        padded_adj_matrix = F.pad(adj_matrix, (0, pad_size, 0, pad_size), 'constant', 0)

        graph_features_batch.append(padded_features)
        adj_matrix_batch.append(padded_adj_matrix)
        labels_batch.append(label)
        images_batch.append(image)
        indices_batch.append(index)

    graph_data_batch = (torch.stack(graph_features_batch), torch.stack(adj_matrix_batch), torch.stack(labels_batch))
    images_batch = torch.stack(images_batch)
    indices_batch = torch.tensor(indices_batch)

    return graph_data_batch, images_batch, indices_batch

#------------------------------------网络定义-------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torchvision import models

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation=None, adj_sq=False, scale_identity=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.adj_sq = adj_sq
        self.activation = activation
        self.scale_identity = scale_identity
        self.in_features = in_features  # Add this line
        self.out_features = out_features  # Add this line

    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        I = torch.eye(N, device=A.device)
        if self.scale_identity:
            I = 2 * I  # increase weight of self connections
        A_hat = A + I
        D_hat = (A_hat.sum(dim=1) + 1e-5).pow(-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A = data
        A = self.laplacian_batch(A)
        x = self.fc(torch.bmm(A, x))
        if self.activation is not None:
            x = self.activation(x)
        return (x, A)

class GCN(nn.Module):
    def __init__(self, in_features, out_features, filters=[64, 64, 64], dropout=0.2, adj_sq=False, scale_identity=False):
        super(GCN, self).__init__()
        self.gconv_layers = nn.ModuleList([
            GraphConv(in_features=filters[i-1] if i > 0 else in_features, 
                      out_features=f,
                      activation=nn.ReLU(),
                      adj_sq=adj_sq,
                      scale_identity=scale_identity) for i, f in enumerate(filters)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_features = filters[-1]

    def forward(self, data):
        x, adj_matrix = data
        for gconv in self.gconv_layers:
            x, _ = gconv((x, adj_matrix))
        x = x.mean(dim=1)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, model_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads

        self.query_transform = nn.Linear(input_dim, model_dim, bias=False)
        self.key_transform = nn.Linear(input_dim, model_dim, bias=False)
        self.value_transform = nn.Linear(input_dim, model_dim, bias=False)

        self.output_transform = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.query_transform(query).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = self.key_transform(key).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.value_transform(value).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim_per_head ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)

        output = self.output_transform(attention_output)
        return output

class FeatureFusionWithBiAttention(nn.Module):
    def __init__(self, input_dim_gcn, input_dim_resnet, output_dim, num_heads):
        super(FeatureFusionWithBiAttention, self).__init__()
        self.attention_gcn_to_resnet = MultiHeadAttention(num_heads, input_dim_gcn + input_dim_resnet, output_dim)
        self.attention_resnet_to_gcn = MultiHeadAttention(num_heads, input_dim_gcn + input_dim_resnet, output_dim)
        self.output_transform = nn.Linear(2 * output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, gcn_features, resnet_features):
        combined_features_gcn_to_resnet = torch.cat((gcn_features, resnet_features), dim=-1)
        combined_features_resnet_to_gcn = torch.cat((resnet_features, gcn_features), dim=-1)

        attention_output_gcn_to_resnet = self.attention_gcn_to_resnet(combined_features_gcn_to_resnet, combined_features_gcn_to_resnet, combined_features_gcn_to_resnet)
        attention_output_resnet_to_gcn = self.attention_resnet_to_gcn(combined_features_resnet_to_gcn, combined_features_resnet_to_gcn, combined_features_resnet_to_gcn)

        combined_attention_output = torch.cat((attention_output_gcn_to_resnet, attention_output_resnet_to_gcn), dim=-1)
        combined_attention_output = self.output_transform(combined_attention_output)
        combined_attention_output = combined_attention_output.view(combined_attention_output.size(0), -1)
        #print(combined_attention_output.shape)
        combined_attention_output = self.batch_norm(combined_attention_output)
        
        return combined_attention_output

class FeatureExchangeLayer(nn.Module):
    def __init__(self, gcn_features_dim, cnn_features_dim):
        super(FeatureExchangeLayer, self).__init__()
        self.routing_weights = nn.Parameter(torch.rand(1, 1, gcn_features_dim, cnn_features_dim))
        self.linear_transform = nn.Linear(cnn_features_dim, gcn_features_dim)

    def forward(self, gcn_features, cnn_features):
        batch_size, num_nodes, gcn_dim = gcn_features.shape
        cnn_dim = cnn_features.shape[1]

        # Adjust CNN features to match GCN features dimension
        adjusted_cnn_features = self.linear_transform(cnn_features).unsqueeze(1).expand(-1, num_nodes, -1)
        #print(f"Adjusted CNN features shape after linear transform and unsqueeze: {adjusted_cnn_features.shape}")

        # Apply routing weights
        routing_scores = torch.sigmoid(self.routing_weights).expand(batch_size, num_nodes, -1, -1)
        #print(f"Routing scores shape: {routing_scores.shape}")

        adjusted_cnn_features = adjusted_cnn_features.unsqueeze(3)  # Add an extra dimension for routing scores multiplication
        adjusted_cnn_features = (adjusted_cnn_features * routing_scores).sum(dim=-1)  # Sum along the last dimension
        #print(f"Adjusted CNN features shape after applying routing scores and summing: {adjusted_cnn_features.shape}")

        # Ensure the dimensions of adjusted_cnn_features match gcn_features
        assert adjusted_cnn_features.shape == gcn_features.shape, f"Shape mismatch: {adjusted_cnn_features.shape} vs {gcn_features.shape}"

        # Add adjusted CNN features to GCN features
        combined_features = gcn_features + adjusted_cnn_features
        return combined_features







class GCNWithExchange(GCN):
    def __init__(self, in_features, filters=[64, 128, 256, 512], exchange_layers=[]):
        super(GCNWithExchange, self).__init__(in_features=in_features, out_features=filters[-1], filters=filters)
        self.exchange_layers = nn.ModuleList(exchange_layers)

    def forward(self, data, cnn_features):
        x, adj_matrix = data
        for i, gconv in enumerate(self.gconv_layers):
            x, _ = gconv((x, adj_matrix))
            if i < len(self.exchange_layers) and self.exchange_layers[i] is not None:
                x = self.exchange_layers[i](x, cnn_features)
        x = x.mean(dim=1)
        return x

class CombinedModelWithExchange(nn.Module):
    def __init__(self, gcn_model, resnet_model, num_classes, fusion_output_dim, num_heads, exchange_indices):
        super(CombinedModelWithExchange, self).__init__()
        self.gcn = gcn_model
        self.resnet = resnet_model
        resnet_out_features = 512

        self.exchange_layers = nn.ModuleList([
            FeatureExchangeLayer(gcn_model.gconv_layers[i].out_features, resnet_out_features) 
            if i in exchange_indices else None for i in range(len(gcn_model.gconv_layers))
        ])

        self.gcn_with_exchange = GCNWithExchange(
            gcn_model.gconv_layers[0].in_features,
            [layer.out_features for layer in gcn_model.gconv_layers],
            exchange_layers=self.exchange_layers
        )

        self.fusion = FeatureFusionWithBiAttention(gcn_model.out_features, resnet_out_features, fusion_output_dim, num_heads)
        self.classifier = nn.Linear(512, num_classes)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, graph_data, image_data):
        resnet_output = self.resnet(image_data)
        gcn_output = self.gcn_with_exchange(graph_data, resnet_output)
        #print(gcn_output.shape)
        #print(resnet_output.shape)
        #combined_features = self.fusion(gcn_output, resnet_output)
        #print(combined_features.shape)
        combined_features = gcn_output + resnet_output
        output = self.classifier(combined_features)
        output = output.squeeze(1)
        return output

# Instantiate the GCN model
gcn_model = GCNWithExchange(in_features=7, filters=[64, 128, 256, 512])

# Instantiate and modify the ResNet model
resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = nn.Identity()

# Specify which layers to perform feature exchange
exchange_indices = [0, 1, 2]

# Instantiate the combined model
combined_model = CombinedModelWithExchange(gcn_model, resnet_model, num_classes=10, fusion_output_dim=128, num_heads=8, exchange_indices=exchange_indices)

# Set device
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
combined_model.to(device)

print("GCN and ResNet models are initialized and ready for training.")


#---------------------------训练部分---------------------------------
print('Loading data')
datareader = DataReader(data_dir='/data1/qsy/gcn/heartnew_full_arrtributes', images_dir='/data1/qsy/gcn/heartnew_4.8_guiyi_big_eassy')
print(datareader)
datareader.load_data()

train_dataset = GraphData(datareader, datareader.train_indices, 'train', datareader.image_paths, datareader.label_dict)
val_dataset = GraphData(datareader, datareader.val_indices, 'validation', datareader.image_paths, datareader.label_dict)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

#print f"Number of training samples: {len(train_dataset)}")
#print f"Number of validation samples: {len(val_dataset)}")

from sklearn.metrics import classification_report

optimizer = optim.AdamW(combined_model.parameters(), lr=lr, weight_decay=wdecay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

from sklearn.metrics import classification_report, precision_recall_fscore_support

def evaluate(loader, model, loss_fn, device, results_dir):
    model.eval()
    total_loss = 0.0
    targets, predictions = [], []
    class_correct = [0 for _ in range(loader.dataset.n_classes)]
    class_total =   [0 for _ in range(loader.dataset.n_classes)]

    with torch.no_grad():
        for _, ((graph_data, adj_matrix, labels), images, indices) in enumerate(loader):
            graph_data = graph_data.to(device)
            adj_matrix = adj_matrix.to(device)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model((graph_data, adj_matrix), images).squeeze(1)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            predictions.extend(preds.tolist())
            targets.extend(labels.tolist())

            for i in range(len(labels)):
                y, yhat = labels[i].item(), preds[i].item()
                class_correct[y] += int(yhat == y)
                class_total[y] += 1

    cm = confusion_matrix(targets, predictions)
    pd.DataFrame(
        cm,
        index=[f'Class {i}' for i in range(cm.shape[0])],
        columns=[f'Class {i}' for i in range(cm.shape[1])]
    ).to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))

    correct_predictions = sum(class_correct)
    total_samples = len(targets)
    overall_acc = correct_predictions / max(total_samples, 1)

    precision, recall, fscore, support = precision_recall_fscore_support(targets, predictions, average=None)
    weighted_precision = float(np.sum(precision * support) / max(np.sum(support), 1))
    # === 新增：计算 Cohen's Kappa 系数 ===
    kappa = cohen_kappa_score(targets, predictions)
    # 每类准确率打印到日志
    for i in range(loader.dataset.n_classes):
        if class_total[i] > 0:
            logger.info('Class %d Acc: %.2f%% (%d/%d)', i, 100.0 * class_correct[i] / class_total[i], class_correct[i], class_total[i])

    logger.info("Eval -> Overall Acc: %.4f | kappa: %.4f", overall_acc, kappa)

    return cm, targets, predictions, overall_acc, kappa



def print_classification_report(targets, predictions, title):
    report = classification_report(targets, predictions, target_names=[f'Class {i}' for i in range(len(np.unique(targets)))], digits=4)
    print(f"\n{title} Classification Report:\n{report}")

def save_classification_report(targets, predictions, results_dir, title="Validation"):
    report = classification_report(
        targets, predictions,
        target_names=[f'Class {i}' for i in range(len(np.unique(targets)))],
        digits=4
    )
    logger.info("\n%s Classification Report:\n%s", title, report)
    with open(os.path.join(results_dir, f"{title.lower()}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

#results_dir = '/data1/qsy/gcn/logs/combinednet_new_a=0_1/4'
os.makedirs(results_dir, exist_ok=True)

def train_model(train_loader, val_loader, test_loader, model, optimizer, loss_fn, epochs, device, results_dir, scheduler=None):
    best_acc = 0.0
    metrics_csv = os.path.join(results_dir, "metrics.csv")

    # 写 CSV 表头
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_acc", "kappa", "lr", "best_acc"])

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training", unit='batch')
        for ((graph_features, adj_matrix, labels), images, indices) in pbar:
            graph_features = graph_features.to(device)
            adj_matrix = adj_matrix.to(device)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model((graph_features, adj_matrix), images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        cm, val_targets, val_predictions, val_acc, kappa = evaluate(
            val_loader, model, loss_fn, device, results_dir
        )
        
        cm, test_targets, test_predictions, test_acc, test_kappa = evaluate(
            test_loader, model, loss_fn, device, results_dir
        )
        save_classification_report(test_targets, test_predictions, results_dir, title="Test")

        avg_train_loss = total_loss / max(len(train_loader), 1)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info("Epoch %d | Train Loss: %.4f | Val Acc: %.4f | kappa: %.4f | LR: %.6f",
                    epoch, avg_train_loss, val_acc, kappa, current_lr)

        # 记录到 CSV
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{val_acc:.6f}", f"{kappa:.6f}", f"{current_lr:.8f}", f"{max(best_acc, val_acc):.6f}"])

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))
            logger.info("** New best model saved. Best Val Acc: %.4f **", best_acc)
            logger.info("** Test Acc: %.4f **", test_acc)
            logger.info("** Test KAPPA: %.4f **", test_kappa)
        # 学习率调度（原来你创建了 scheduler 但没有 step）
        if scheduler is not None:
            scheduler.step()

    logger.info("Training finished. Best Val Acc: %.4f", best_acc)


if __name__ == "__main__":
    logger.info('Loading data')
    # 设置随机数种子
    setup_seed(88)
    datareader = DataReader(data_dir='/data1/qsy/gcn/heartnew_4.8', images_dir='/data1/qsy/gcn/heart_class_4.8_1409_resnet')
    datareader.load_data()

    train_dataset = GraphData(datareader, datareader.train_indices, 'train', datareader.image_paths, datareader.label_dict)
    val_dataset = GraphData(datareader, datareader.val_indices, 'validation', datareader.image_paths, datareader.label_dict)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    test_dataset = GraphData(datareader, datareader.test_indices, 'test', datareader.image_paths, datareader.label_dict)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    logger.info("Number of training samples: %d", len(train_dataset))
    logger.info("Number of validation samples: %d", len(val_dataset))

    gcn_model = GCNWithExchange(in_features=7, filters=[64, 128, 256, 512])
    combined_model = CombinedModelWithExchange(gcn_model, resnet_model, num_classes=5, fusion_output_dim=128, num_heads=8, exchange_indices=exchange_indices)
    combined_model.to(device)

    logger.info("\nCombined Model Structure:\n%s", combined_model)

    optimizer = torch.optim.Adam(combined_model.parameters(), lr=lr, weight_decay=wdecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_model(train_loader, val_loader, test_loader, combined_model, optimizer, loss_fn, epochs, device, results_dir, scheduler=scheduler)


