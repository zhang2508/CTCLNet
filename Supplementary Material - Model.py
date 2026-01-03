import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

class Network2DCNN(nn.Module):
    def __init__(self, FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH):
        super(Network2DCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(TAR_INPUT_DIMENSION, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc = nn.Linear(128, FEATURE_DIM)

    def forward(self, x):
        feature = self.cnn(x)
        return self.fc(feature)


class NetworkMLP(nn.Module):
    def __init__(self, FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH):
        super(NetworkMLP, self).__init__()
        input_dim = TAR_INPUT_DIMENSION*PATCH*PATCH
        hidden_dim1 = 32
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, FEATURE_DIM),
        )

    def forward(self, x):
        feature = self.mlp(x)
        return feature


class Network1DCNN(nn.Module):
    def __init__(self, FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH):
        super(Network1DCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128, FEATURE_DIM)

    def forward(self, x):
        x = x.mean(dim=(2, 3), keepdim=False)
        x = x.unsqueeze(1)
        feature = self.feature_extractor(x)
        output = self.fc(feature)
        return output


class Network3DCNN(nn.Module):
    def __init__(self, FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH):
        super(Network3DCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(20, 3, 3), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(10, 3, 3), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(5, 3, 3), padding=0),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32, FEATURE_DIM)

    def forward(self, x):
        x = x.unsqueeze(1)

        feature = self.feature_extractor(x)
        output = self.fc(feature)
        return output


class NetworkRNN(nn.Module):
    def __init__(self, FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH):
        super(NetworkRNN, self).__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(self.hidden_size * 2, FEATURE_DIM)

    def forward(self, x):
        x = x.mean(dim=(2, 3), keepdim=False)
        x = x.unsqueeze(2)
        output, h_n = self.rnn(x)
        fwd_h = h_n[-2, :, :]
        bwd_h = h_n[-1, :, :]
        h_n_concat = torch.cat((fwd_h, bwd_h), dim=1)
        output = self.fc(h_n_concat)
        return output


class NetworkLSTM(nn.Module):
    def __init__(self, FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH):
        super(NetworkLSTM, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(self.hidden_size * 2, FEATURE_DIM)

    def forward(self, x):
        x = x.mean(dim=(2, 3), keepdim=False)
        x = x.unsqueeze(2)
        output, (h_n, c_n) = self.lstm(x)
        fwd_h = h_n[-2, :, :]
        bwd_h = h_n[-1, :, :]
        h_n_concat = torch.cat((fwd_h, bwd_h), dim=1)
        output = self.fc(h_n_concat)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NetworkTransformer(nn.Module):
    def __init__(self, FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH):
        super(NetworkTransformer, self).__init__()
        d_model = 128
        nhead = 8
        num_layers = 6
        dim_feedforward = 256
        seq_len = PATCH * PATCH

        self.input_proj = nn.Linear(TAR_INPUT_DIMENSION, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, FEATURE_DIM)

    def forward(self, x):
        x = x.flatten(2)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)

        transformer_out = self.transformer_encoder(x)
        cls_out = transformer_out[:, 0, :]
        output = self.fc(cls_out)
        return output

try:
    from .mamba import Mamba
except:
    from mamba_ssm import Mamba

class NetworkMamba(nn.Module):
    def __init__(self, FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH):
        super(NetworkMamba, self).__init__()
        d_model = 32
        d_state = 16
        d_conv = 4
        expand = 2

        self.input_proj_conv = nn.Sequential(
            nn.Conv2d(TAR_INPUT_DIMENSION, d_model,1),
            nn.BatchNorm2d(d_model),
            nn.GELU()
        )

        self.mamba_layers = nn.Sequential(
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
        )

        self.fc = nn.Linear(d_model, FEATURE_DIM)

    def forward(self, x):
        x = self.input_proj_conv(x)
        x = x.reshape(x.shape[0],x.shape[1],-1).permute(0,2,1)
        x = self.mamba_layers(x)
        mamba_out = torch.mean(x,dim=1)
        output = self.fc(mamba_out)
        return output

class HybridMambaCNN(nn.Module):
    def __init__(self, FEATURE_DIM, TAR_INPUT_DIMENSION, PATCH):
        super(HybridMambaCNN, self).__init__()
        embed_dim = 32
        self.spectral_projection = nn.Sequential(
            nn.Conv2d(TAR_INPUT_DIMENSION, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        self.mamba_branch = nn.Sequential(
            Mamba(
                d_model=embed_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            ),

            nn.Linear(embed_dim, FEATURE_DIM),
        )

        self.cnn_branch = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, FEATURE_DIM, kernel_size=3),
            nn.BatchNorm2d(FEATURE_DIM),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        projected_x = self.spectral_projection(x)
        b, d, h, w = projected_x.shape
        mamba_input = projected_x.reshape(b, d, -1).permute(0, 2, 1)
        mamba_output = self.mamba_branch(mamba_input)
        mamba_feature = mamba_output.mean(dim=1)
        cnn_feature = self.cnn_branch(projected_x)
        combined_feature = mamba_feature + cnn_feature
        return combined_feature

