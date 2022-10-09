import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet

import config as c
from freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, ReversibleGraphNet, OutputNode, \
    InputNode, Node

WEIGHT_DIR = './weights'
MODEL_DIR = './models'


#n_feat : 256 * 3
#n_coupling_blocks : 8
def nf_head(input_dim=c.n_feat):

    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': c.fc_internal, 'dropout': c.dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class DifferNet(nn.Module):
    def __init__(self, sd_dims):
        super(DifferNet, self).__init__()
        self.feature_extractor = alexnet(pretrained=True)
        self.nf = nf_head( len(sd_dims)*c.n_scales )
        self.sd_dims = sd_dims
        print('[DifferNet Init] sd_dims:', sd_dims)

    def get_features(self, x):
        return self.feature_extractor.features(x)

    def forward(self, x):
        y_cat = list()

        #n_scales = 3
        for s in range(c.n_scales):
            #print('n_scales:',s,',shape of x:',x.shape)
            x_scaled = F.interpolate(x, size=c.img_size[0] // (2 ** s)) if s > 0 else x
            #print('n_scales:',s,',shape of x_scaled:',x.shape)
            feat_s = self.feature_extractor.features(x_scaled)            

            #self.sd_dims
            feat_s = feat_s[:,self.sd_dims,:,:]

            #print('n_scales:',s,',shape of feat_s:',feat_s.shape)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))

        #print('shape of y_cat:',y_cat)
        y = torch.cat(y_cat, dim=1)
        #print('shape of y:',y.shape)
        z = self.nf(y)
        #print('shape of z:',z.shape)
        return z


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model
