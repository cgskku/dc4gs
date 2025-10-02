#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, find_optimal_x_polynomial

class DCCGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, b_strip=True):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if b_strip:
                symm = strip_symmetric(actual_covariance)
            else:
                symm = actual_covariance
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.xyz_gradient_accum_abs_only = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.dcc = torch.empty(0)
        self.split_samples = torch.empty(0)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.xyz_gradient_accum_abs_only,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.dcc,
            self.split_samples
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D,
        self.max_weight,
        xyz_gradient_accum,
        xyz_gradient_accum_abs,
        xyz_gradient_accum_abs_only,
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        dcc,
        split_samples) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.xyz_gradient_accum_abs_only = xyz_gradient_accum_abs_only
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.dcc = dcc
        self.split_samples = split_samples

    def capture_for_3DGS(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore_from_3DGS(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = torch.zeros((self.xyz_gradient_accum.shape[0],1),device="cuda")
        self.xyz_gradient_accum_abs_only = torch.zeros((self.xyz_gradient_accum.shape[0],1),device="cuda")
        self.max_weight = torch.zeros_like(self.max_radii2D)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.dcc = torch.zeros((self.xyz_gradient_accum.shape[0],3),device="cuda")
        self.split_samples = torch.zeros((self.xyz_gradient_accum.shape[0], 10, 5),device="cuda")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1, b_strip=True):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, b_strip)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        dcc = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.dcc = nn.Parameter(dcc.requires_grad_(True))
        split_samples = torch.zeros((self.get_xyz.shape[0], 10, 5),device="cuda")
        self.split_samples = nn.Parameter(split_samples.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_only = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reduce_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        dcc = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.dcc = nn.Parameter(dcc.requires_grad_(True))
        split_samples = torch.zeros((self.get_xyz.shape[0], 10, 5),device="cuda")
        self.split_samples = nn.Parameter(split_samples.requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_only = self.xyz_gradient_accum_abs_only[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]
        self.dcc = nn.Parameter(self.dcc[valid_points_mask])
        self.split_samples = nn.Parameter(self.split_samples[valid_points_mask])

    def initial_prune(self):
        pts_mask_1 = torch.max(self.get_scaling, dim=1).values > torch.mean(self.get_scaling)
        if len(self.get_scaling) < 500_0000:
            pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.quantile(
                self.get_scaling,
                0.999)
        else:
            pts_mask_2 = torch.max(self.get_scaling, dim=1).values > torch.mean(
                self.get_scaling) * 4

        selected_pts_mask = torch.logical_and(pts_mask_1, pts_mask_2)
        print("Initial pruning based on radius, GS num: ", sum(selected_pts_mask))
        self.prune_points(selected_pts_mask)


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_only = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        max_weight = torch.zeros((new_xyz.shape[0]), device="cuda")
        self.max_weight = torch.cat((self.max_weight,max_weight),dim=0)
        dcc = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.dcc = nn.Parameter(dcc.requires_grad_(True))
        split_samples = torch.zeros((self.get_xyz.shape[0], 10, 5),device="cuda")
        self.split_samples = nn.Parameter(split_samples.requires_grad_(True))
        
    
    def densify_and_split_axis_samples(self, grads, grad_threshold, scene_extent, split_samples, opt, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
         
        padded_split_samples = torch.zeros((n_init_points, split_samples.shape[1]), device="cuda")
        padded_split_samples[:split_samples.shape[0]] = split_samples
        
        
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)      
            
        if opt.only_dcc == False:
            if opt.const_split <= 0:
                selected_split_samples = padded_split_samples[selected_pts_mask]
                N_sample = selected_split_samples.shape[1]
                x_samples = torch.linspace(0, 1, N_sample+2, device='cuda')
                x_samples = x_samples[1:-1]
                y_samples  = selected_split_samples
                
                if opt.argmin_split:
                    min_idx = torch.argmin(y_samples, dim=1)
                    adapt_p = x_samples[min_idx]
                else:
                    adapt_p = find_optimal_x_polynomial(x_samples, y_samples, opt.without_bound)
                invalid_adapt_P = torch.where((adapt_p==0) | (adapt_p==1), True, False)
                n_invalid = invalid_adapt_P.sum()
                
                if n_invalid > 0:
                    print(f"invalid adapt_P : {n_invalid}, total: {invalid_adapt_P.shape[0]}, ratio: {n_invalid/invalid_adapt_P.shape[0]}")
                    
                    ####### handle invalid ########
                    if opt.invalid_argmin:
                        # argmin
                        argmin_invalid_adapt_p = torch.argmin(y_samples[invalid_adapt_P], dim=1)
                        adapt_p[invalid_adapt_P] = x_samples[argmin_invalid_adapt_p]
                        adapt_p = adapt_p.unsqueeze(1)
                        ####
                    else:
                        # exclude
                        refine_selected_pts_mask = ~invalid_adapt_P
                        adapt_p = adapt_p[refine_selected_pts_mask, None]
                        selected_pts_mask[selected_pts_mask==True] = refine_selected_pts_mask[:]
                    #############################
                if len(adapt_p.shape) == 1:
                    adapt_p = adapt_p.unsqueeze(1)
            else:
                adapt_p = opt.const_split
            
            centers = self.get_xyz[selected_pts_mask] 
            selected_scaling = self.get_scaling[selected_pts_mask]
            selected_rots = build_rotation(self._rotation[selected_pts_mask])
        
            principal_idx = torch.argmax(selected_scaling, dim=1)     
            sigma = selected_scaling[torch.arange(principal_idx.shape[0]), principal_idx, None]
        
            local_principal = torch.zeros(principal_idx.shape[0], 3, device='cuda')
            local_principal[torch.arange(principal_idx.shape[0]), principal_idx] = 1.0
            
            world_principal = torch.bmm(selected_rots, local_principal.unsqueeze(-1)).squeeze(-1)
            
            ############  position  ################# 
            
            # left large
            d_left = 6 * sigma * (1 - adapt_p)
            d_right = 6 * sigma * adapt_p
            
           ############################# 
            
            new_center_left = centers - (d_left / 2)* world_principal
            new_center_right = centers + (d_right / 2) * world_principal
            new_xyz = torch.cat([new_center_left, new_center_right], dim=0)

            ############  scaling  ################# 
            new_scaling_left = selected_scaling.clone()
            new_scaling_right = selected_scaling.clone()
            
            if opt.const_scale == 0:
                new_scaling_left = new_scaling_left * adapt_p
                new_scaling_right = new_scaling_right * (1-adapt_p)
            elif opt.const_scale > 0:
                new_scaling_left = new_scaling_left * opt.const_scale
                new_scaling_right = new_scaling_right * opt.const_scale     
                
            
            # left large
            new_scaling_left[torch.arange(new_scaling_left.shape[0]), principal_idx, None] = sigma * adapt_p
            new_scaling_right[torch.arange(new_scaling_right.shape[0]), principal_idx, None] = sigma * (1-adapt_p)
            
            #############################
            
            new_scaling = self.scaling_inverse_activation(torch.cat([new_scaling_left, new_scaling_right], dim=0))

            new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
            
            ############  opacity  ################# 
            
            if opt.const_opacity <= 0:
                if opt.inverse_op:
                    # left opacity small
                    new_left_opacity = self.get_opacity[selected_pts_mask].clone() * (1 - adapt_p)
                    new_right_opacity = self.get_opacity[selected_pts_mask].clone() * adapt_p
                else:
                    # left opacity large
                    new_left_opacity = self.get_opacity[selected_pts_mask].clone() * adapt_p
                    new_right_opacity = self.get_opacity[selected_pts_mask].clone() * (1 - adapt_p)
            else:
                new_left_opacity = self.get_opacity[selected_pts_mask].clone() * opt.const_opacity
                new_right_opacity = self.get_opacity[selected_pts_mask].clone() * opt.const_opacity
            
            #############################
            
            new_opacity = self.inverse_opacity_activation(torch.cat([new_left_opacity, new_right_opacity], dim=0))
        else:        
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
               
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # print("clone:", new_xyz.shape[0])
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, max_grad_abs, min_opacity, extent, max_screen_size, opt):
        
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        if opt.only_split == False:
            grads_abs = self.xyz_gradient_accum_abs / self.denom
            grads_abs[grads_abs.isnan()] = 0.0
        else:
            grads_abs_only = self.xyz_gradient_accum_abs_only / self.denom
            grads_abs_only[grads_abs_only.isnan()] = 0.0
        
        split_samples = self.split_samples[:, :, 0] / self.denom
        split_samples[split_samples.isnan()] = 1.0
        split_samples = split_samples.view(-1, split_samples.shape[1]//2, 2).sum(dim=2) / 2.0

        self.densify_and_clone(grads, max_grad, extent)
        
        if opt.only_split == False:
            self.densify_and_split_axis_samples(grads_abs, max_grad_abs, extent, split_samples, opt)
        else:
            self.densify_and_split_axis_samples(grads_abs_only, max_grad_abs, extent, split_samples, opt)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        

    def add_densification_stats(self, viewspace_point_tensor, update_filter, opt, iter=0, toy=False):        
        view_grad = viewspace_point_tensor.grad[update_filter, :2]
        vec_cnt = self.dcc.grad[update_filter, -1]
        
        unit_C = self.dcc.grad[update_filter, 0:2] / vec_cnt.view(-1, 1)
        unit_mean = torch.norm(unit_C, dim=1)
        unit_mean[unit_mean.isnan()] = 1.0
        unit_variance = 1.0 - unit_mean
        
        view_grad_norm = torch.norm(view_grad, dim=-1, keepdim=True)
        view_grad_abs_norm = torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1,
                                                             keepdim=True)
        self.xyz_gradient_accum[update_filter] += view_grad_norm
        
        if opt.only_split == False:
            self.xyz_gradient_accum_abs[update_filter] += (unit_variance.view(-1,1)) * view_grad_abs_norm
        else:
            self.xyz_gradient_accum_abs_only[update_filter] += view_grad_abs_norm
        
        split_samples = self.split_samples.grad[update_filter]
        split_mean_C = split_samples[:, :, :2] / split_samples[:, :, 2:3]
        split_mean_length = torch.norm(split_mean_C, dim=2)
        split_mean_length = torch.nan_to_num(split_mean_length, nan=0.0)
        split_var = 1.0 - split_mean_length
        
        split_grad_abs_norm = torch.norm(split_samples[:, :, 3:5], dim=-1)
        
        self.split_samples[update_filter, :, 0] += split_var * split_grad_abs_norm
        
        self.denom[update_filter] += 1