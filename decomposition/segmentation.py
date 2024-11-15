import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
# from scene.grid import HashHexPlane


class Segmentation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Segmentation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64, 64, 64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))
        self.ratio = 0
        self.create_net()

    @property
    def get_aabb(self):
        return self.grid.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        print("Segmentation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)

    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            grid_out_dim = self.grid.feat_dim + (self.grid.feat_dim) * 2 
        else:
            grid_out_dim = self.grid.feat_dim

        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_seg = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 2))

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3], time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            hidden = torch.cat([grid_feature], -1)
        hidden = self.feature_out(hidden)
        return hidden
    
    @property
    def get_empty_ratio(self):
        return self.ratio
    
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:, :3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:, :3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    
    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        seg = self.pos_seg(hidden)
        return seg
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" in name:
                parameter_list.append(param)
        return parameter_list
    

class seg_network(nn.Module):
    def __init__(self, args) :
        super(seg_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth = args.defor_depth
        posbase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2 * timebase_pe + 1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_output))
        self.segmentation_net = Segmentation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point):
        return self.forward_dynamic(point)
    
    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        segs = self.segmentation_net(point_emb, scales_emb, rotations_emb, opacity, shs, None, times_sel)
        return segs
    
    def get_mlp_parameters(self):
        return self.segmentation_net.get_mlp_parameters() + list(self.timenet.parameters())
    
    def get_grid_parameters(self):
        return self.segmentation_net.get_grid_parameters()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight, gain=1)
            # init.constant_(m.bias, 0)


def poc_fre(input_data,poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb