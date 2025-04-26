from collections import OrderedDict
import torch
from torch import nn


def exist(x):
    return x is not None

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    
    def forward(self,x):
        return self.fn(x)+x

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, layers = 1, expansion_rate = 1):
        super().__init__()
        c = input_dim * expansion_rate 
        modules = [
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, c)
                ]
        for _ in range(layers): 
            modules.append(nn.Linear(c, c)),
            modules.append(nn.GELU()),
            modules.append(nn.Dropout(p=dropout)),
        
        modules.append(nn.Linear(c, output_dim))
        modules.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.layers(x)
    
class SpatialGatingUnit(nn.Module):
    def __init__(self,dim,len_sen):
        super().__init__()
        self.ln=nn.LayerNorm(dim)
        self.proj=nn.Conv1d(len_sen,len_sen,1)

        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)
    
    def forward(self,x):
        res,gate=torch.chunk(x,2,-1) #bs,n,d_ff
        ###Norm
        gate=self.ln(gate) #bs,n,d_ff
        ###Spatial Proj
        gate=self.proj(gate) #bs,n,d_ff

        return res*gate

class gMLP(nn.Module):
    def __init__(self,num_tokens=None,len_sen=49,dim=512,d_ff=1024,num_layers=6):
        super().__init__()
        self.num_layers=num_layers
        self.embedding=nn.Embedding(num_tokens,dim) if exist(num_tokens) else nn.Identity()

        self.gmlp=nn.ModuleList([Residual(nn.Sequential(OrderedDict([
            ('ln1_%d'%i,nn.LayerNorm(dim)),
            ('fc1_%d'%i,nn.Linear(dim,d_ff*2)),
            ('gelu_%d'%i,nn.GELU()),
            ('sgu_%d'%i,SpatialGatingUnit(d_ff,len_sen)),
            ('fc2_%d'%i,nn.Linear(d_ff,dim)),
        ])))  for i in range(num_layers)])



        self.to_logits=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_tokens),
            nn.Softmax(-1)
        )


    def forward(self,x):
        #embedding
        embeded=self.embedding(x)

        #gMLP
        y=nn.Sequential(*self.gmlp)(embeded)


        #to logits
        logits=self.to_logits(y)


        return logits

class SGU(nn.Module):
    def __init__(self, dim, sequence_len):
        super().__init__()
        gate_dim = dim // 2
        self.norm = nn.LayerNorm(gate_dim)
        self.proj = nn.Linear(sequence_len, sequence_len)

    def init_weights(self):
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)


    def forward(self, x):
        #print(x.shape)
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v).transpose(-1, -2)
        try:
            v = self.proj(v)
        except:
            d = int(v.shape[-1])
            proj = nn.Linear(d, d).to('cuda')
                                                                                                                                                    
            v = proj(v)
            
        return u * v.transpose(-1, -2)

class original_gMLP(nn.Module):
    def __init__(self, d_model, seq_len, activation = 'sig', mlp_ratio = 1, drop = 0.5):
        super().__init__() 
        self.norm = nn.LayerNorm(d_model)
        channel_dim = d_model * mlp_ratio
        self.proj_1 = nn.Linear(d_model, channel_dim)
        self.activation = nn.Sigmoid() if activation == 'sig' else nn.GELU()
        self.drop = nn.Dropout(drop, inplace=True)
        self.sgu = SGU(channel_dim, seq_len)
        self.proj_2 = nn.Linear(channel_dim//2, d_model)

    def forward(self, x):
        shorcut = x
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        #x = self.drop(x)
        x = self.sgu(x)
        x = self.proj_2(x)
        #x = self.drop(x)
        return x + shorcut

class VisionProjector(nn.Module):
    """
    Textual Inversion Phi network.
    Takes as input the visual features of an image and outputs the pseudo-work embedding.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim)

        self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.layers(x)

