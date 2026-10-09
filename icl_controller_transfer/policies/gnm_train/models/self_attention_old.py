import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class MultiLayerDecoder(nn.Module):
    ## TODO: Add causal masking. If not, make this an encoder
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
    
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(feature_dim, input_dim)
        self.beta = nn.Linear(feature_dim, input_dim)
        self.input_dim = input_dim
        
    def forward(self, x, feature):
        gamma = self.gamma(feature)
        beta = self.beta(feature)
        return gamma * x + beta
    

class MultiLayerDecoderResidualGoal(nn.Module):
    ## TODO: Add causal masking. If not, make this an encoder
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoderResidualGoal, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True) for _ in range(num_layers)])
        self.films = nn.ModuleList([FiLM(embed_dim, embed_dim) for _ in range(num_layers)])
        # self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.num_layers = num_layers
        # self.concat_layers =  nn.ModuleList([nn.Linear(embed_dim*2, embed_dim) for _ in range(num_layers)])
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, seq, goal):
        assert goal.shape[0] == seq.shape[0]
        # expand goal to match seq
        if len(goal.shape) == 2:
            goal = goal.unsqueeze(1)
        goal = goal.repeat(1, seq.shape[1], 1)
        if self.positional_encoding: seq = self.positional_encoding(seq)
        for i in range(self.num_layers):
            # seq = torch.cat([seq, goal], dim=-1)
            seq = self.films[i](seq, goal)
            seq = self.sa_layers[i](seq)
        # currently, x is [batch_size, seq_len, embed_dim]
        out = seq.reshape(seq.shape[0], -1)
        for i in range(len(self.output_layers)):
            out = self.output_layers[i](out)
            out = F.relu(out)
        return out


class MultiLayerDecoderConcatGoal(nn.Module):
    ## TODO: Add causal masking. If not, make this an encoder
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoderResidualGoal, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True) for _ in range(num_layers)])
        self.concat_layers = nn.ModuleList([nn.Linear(2*embed_dim, embed_dim) for _ in range(num_layers)])
        # self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.num_layers = num_layers
        # self.concat_layers =  nn.ModuleList([nn.Linear(embed_dim*2, embed_dim) for _ in range(num_layers)])
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, seq, goal):
        assert goal.shape[0] == seq.shape[0]
        # expand goal to match seq
        if len(goal.shape) == 2:
            goal = goal.unsqueeze(1)
        goal = goal.repeat(1, seq.shape[1], 1)
        if self.positional_encoding: seq = self.positional_encoding(seq)
        for i in range(self.num_layers):
            seq = torch.cat([seq, goal], dim=-1)
            seq = self.concat_layers[i](seq)
            seq = self.sa_layers[i](seq)
        # currently, x is [batch_size, seq_len, embed_dim]
        out = seq.reshape(seq.shape[0], -1)
        for i in range(len(self.output_layers)):
            out = self.output_layers[i](out)
            out = F.relu(out)
        return out