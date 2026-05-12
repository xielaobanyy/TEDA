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
