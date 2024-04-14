import torch
from torch.nn import functional as F
import numpy as np
#import scene.tools as tools
import pytorch3d


class grid():
    def __init__(self, model):
        self.grids = []


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


class deformation_net():
    def __init__(self, model, bounds=1.0):
        self.model = model
        aabb = torch.tensor([[bounds, bounds, bounds],
                             [-bounds, -bounds, -bounds]], device=model.device)
        self.aabb = torch.nn.Parameter(aabb, requires_grad=False)
        model.aabb = aabb
        self.grid = grid(model)

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ], dtype=torch.float32)
        self.aabb = torch.nn.Parameter(aabb, requires_grad=False)
        self.model.aabb = self.aabb
        print("model aabb: ", self.aabb)


class HexPlane(torch.nn.Module):

    def __init__(self, cfg, device="cuda", bounds=1.0):
        super().__init__()
        self.cfg = cfg
        self.label_n_comp = cfg.model.label_n_comp
        self.deform_n_comp = cfg.model.deform_n_comp

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]

        self.label_dim = cfg.model.max_part_num
        self.deform_dim = 9 * self.label_dim
        self.voxel_grid = cfg.model.voxel_grid
        self.time_grid = cfg.model.time_grid
        self.align_corners = cfg.model.align_corners
        self.init_scale = cfg.model.init_scale
        self.init_shift = cfg.model.init_shift
        self.device = device
        self.init_planes(self.device)
        self.get_voxel_upsample_list()
        self.deformation_net = deformation_net(self, bounds=bounds)

        self.deform_arg = {'gumbel': cfg.optim.gumbel, 'hard': cfg.optim.hard, 'tau': cfg.optim.tau, 'eval': cfg.optim.eval}

        #self.init_zero_deform()

        """self.label_n_comp = [24, 24, 24]
        self.deform_n_comp = 24
        self.label_dim = 10
        self.deform_dim = 9 * self.label_dim
        self.gridSize = [64, 64, 64]
        self.time_grid = 64
        self.align_corners = True
        self.init_scale = 0.1
        self.init_shift = 0.0
        self.device = "cuda"
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.init_planes(self.device)"""

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def init_planes(self, device):
        self.label_plane = self.init_one_triplane(
            self.label_n_comp, self.voxel_grid, device
        )
        self.deform_plane = self.init_one_plane(
            self.deform_n_comp, self.time_grid, device
        )
        self.label_basis_mat = torch.nn.Linear(sum(self.label_n_comp), self.label_dim, bias=False).to(device)
        self.deform_basis_mat = torch.nn.Linear(self.deform_n_comp, self.deform_dim, bias=False).to(device)

    def init_one_plane(self, n_component, gridSize, device):
        plane_coef = []
        plane_coef.append(
            torch.nn.Parameter(
                self.init_scale
                * torch.randn(
                    (1, n_component, gridSize, gridSize)
                )
                + self.init_shift
            )
        )
        return torch.nn.ParameterList(plane_coef).to(device)
    
    def init_one_triplane(self, n_component, gridSize, device):
        plane_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )
        return torch.nn.ParameterList(plane_coef).to(device)
    
    def init_one_hexplane(self, n_component, gridSize, device):
        plane_coef, line_time_coef = [], []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )
            line_time_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn((1, n_component[i], gridSize[vec_id], self.time_grid))
                    + self.init_shift
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_time_coef).to(device)
    
    def compute_labelfeature(self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor) -> torch.Tensor:

        xyz_sampled = normalize_aabb(xyz_sampled, self.aabb)

        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].

        plane_feat = []
        for idx_plane in range(len(self.label_plane)):
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.label_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        plane_feat = torch.stack(plane_feat)
        
        inter = plane_feat
        inter = inter.view(-1, inter.shape[-1])
        inter = self.label_basis_mat(inter.T)  # Feature Projection
        return inter
    
    def compute_deformfeature(self, source_time, target_time):
        B = source_time.shape[0]
        source_time = source_time.view(1, -1, 1, 1)
        target_time = target_time.view(1, -1, 1, 1)

        time_coord = torch.cat((source_time, target_time), -1).detach()
        #.view(1, -1, 1, 2) # (1, B, 1, 2)
        
        plane_feat = F.grid_sample(
            self.deform_plane[0],
            time_coord,
            align_corners=self.align_corners,
        )[0, :, :, 0] #.view(-1, 1) # (1, C, B, 1) -> (C, B)

        deform = self.deform_basis_mat(plane_feat.T) # (C, B) -> (B, C) -> (B, K*9)
        deform = deform.reshape(B, -1, 9).reshape(-1, 9) # (B, K*9) -> (B, K, 9) -> (B*K, 9)
        #rotation = tools.compute_rotation_matrix_from_ortho6d(deform[:, 0:6]).reshape(B, -1, 3, 3) # [B*K, 6] -> [B*K, 3, 3] -> [B, K, 3, 3]
        rotation = pytorch3d.transforms.rotation_6d_to_matrix(deform[:, 0:6]).reshape(B, -1, 3, 3) # [B*K, 6] -> [B*K, 3, 3] -> [B, K, 3, 3]
        translation = deform[:, 6:9].reshape(B, -1, 3) # [B*K, 3] -> [B, K, 3]

        return deform, rotation, translation
    
    def label_feat_to_label(self, label, deform_arg=None):
        deform_arg = self.deform_arg if deform_arg is None else deform_arg
        if deform_arg["eval"] is True:
            _, label_ind = torch.max(label, 1)
            label = torch.eye(label.shape[-1], dtype=label.dtype, device=label.device)[label_ind]
        else:
            if deform_arg["gumbel"]:
                label = torch.nn.functional.gumbel_softmax(label, tau=deform_arg["tau"], hard=deform_arg["hard"])
            else:
                label = torch.nn.functional.softmax(label / deform_arg["tau"], dim=-1)
        return label
    
    def compute_batch_deform_once(
        self, xyz_sampled: torch.Tensor, frame_time_source: torch.Tensor, frame_time_target: torch.Tensor, 
        gumbel=True, hard=False, tau=0.1, eval=False,
    ) -> torch.Tensor:
        assert frame_time_source.shape == frame_time_target.shape

        label_raw = self.compute_labelfeature(xyz_sampled, frame_time_source) # (N, K)

        _, rotation, translation = self.compute_deformfeature(frame_time_source, frame_time_target)
        xyz_sampled_deform = torch.matmul(rotation, xyz_sampled[:, None, :, None]).squeeze(3) + translation # [N, K, 3]

        if eval is True:
            _, label_ind = torch.max(label_raw, 1)
            label = torch.eye(label_raw.shape[-1], dtype=label_raw.dtype, device=label_raw.device)[label_ind]
        else:
            if gumbel:
                label = torch.nn.functional.gumbel_softmax(label_raw, tau=tau, hard=hard)
            else:
                label = torch.nn.functional.softmax(label_raw, dim=-1)
        
        xyz_sampled_deform = torch.sum(xyz_sampled_deform * label[:, :, None], 1)

        results = {}
        results['xyz_sampled_deform'] = xyz_sampled_deform
        results['label_raw'] = label_raw
        results['label'] = label
        results['rotation'] = rotation
        results['translation'] = translation

        return results
    
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity_emb=None, shs_emb=None, time_feature=None, time_emb=None):
        
        pts = rays_pts_emb[:, :3]
        scales = scales_emb[:, :3]
        rotations = rotations_emb[:, :4]
        opacity = opacity_emb[:, :1]
        shs = shs_emb
        
        frame_time_source = torch.zeros_like(time_feature)
        _, rotation, translation = self.compute_deformfeature(torch.zeros_like(time_feature), time_feature * 2.0 - 1.0)
        label_feat = self.compute_labelfeature(pts, frame_time_source) # [N, K]
        label = self.label_feat_to_label(label_feat) # [N, K]

        rotation_concat = torch.sum(rotation * label[:, :, None, None], 1) # [N, 3, 3]
        translation_concat = torch.sum(translation * label[:, :, None], 1) # [N, 3]
        rotation_concat_to_quat = pytorch3d.transforms.matrix_to_quaternion(rotation_concat)

        pts = torch.matmul(rotation_concat, pts[:, :, None]).squeeze(2) + translation_concat
        rotations = pytorch3d.transforms.quaternion_multiply(rotation_concat_to_quat, rotations)

        return pts, scales, rotations, opacity, shs
    
    def TV_loss_label(self, reg):
        total = 0
        for idx in range(len(self.label_plane)):
            total = total + reg(self.label_plane[idx])
        return total
    
    def TV_loss_deform(self, reg):
        total = reg(self.deform_plane[0])
        return total

    def get_optparam_groups(self, cfg, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.deform_plane,
                "lr": lr_scale * cfg.lr_deform_grid,
                "lr_org": cfg.lr_deform_grid,
                "name": "grid",
            },
            {
                "params": self.deform_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_deform_nn,
                "lr_org": cfg.lr_deform_nn,
            },
            {
                "params": self.label_plane,
                "lr": lr_scale * cfg.lr_label_grid,
                "lr_org": cfg.lr_label_grid,
                "name": "grid",
            },
            {
                "params": self.label_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_label_nn,
                "lr_org": cfg.lr_label_nn,
            },
        ]
        return grad_vars

    def get_labeloptparam_groups(self, cfg, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.label_plane,
                "lr": lr_scale * cfg.lr_label_grid,
                "lr_org": cfg.lr_label_grid,
                "name": "grid",
            },
            {
                "params": self.label_basis_mat.parameters(),
                "lr": lr_scale * cfg.lr_label_nn,
                "lr_org": cfg.lr_label_nn,
            },
        ]
        return grad_vars
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "mat" in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "mat" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_voxel_upsample_list(self):
        upsample_list = self.cfg.model.upsample_list
        voxel_grid_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(self.cfg.model.voxel_grid_init),
                        np.log(self.cfg.model.voxel_grid_final),
                        len(upsample_list) + 1,
                    )
                )
            ).long()
        ).tolist()[1:]
            
        time_grid_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(self.cfg.model.time_grid_init),
                        np.log(self.cfg.model.time_grid_final),
                        len(upsample_list) + 1,
                    )
                )
            ).long()
        ).tolist()[1:]
        self.voxel_grid_list = voxel_grid_list
        self.time_grid_list = time_grid_list
    
    @torch.no_grad()
    def up_sampling_planes(self, plane_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(res_target[mat_id_1], res_target[mat_id_0]),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
        return plane_coef
    
    @torch.no_grad()
    def up_sampling_plane(self, plane_coef, time_grid):
        for i in range(1):
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data,
                    size=(time_grid, time_grid),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            )
        return plane_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target, time_grid):
        self.label_plane = self.up_sampling_planes(
            self.label_plane, [res_target, res_target, res_target]
        )
        self.deform_plane = self.up_sampling_plane(
            self.deform_plane, time_grid
        )
        self.time_grid = time_grid
        self.voxel_grid = [res_target, res_target, res_target]

    def init_zero_deform(self):

        # Initialize the optimizer
        grad_vars = self.get_optparam_groups(self.cfg.optim, lr_scale=1.0)
        optimizer = torch.optim.Adam(grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2))
        l1_loss = torch.nn.L1Loss()

        total_loss = 1.0
        while total_loss > 1e-3:
            deform_all = self.deform_plane[0][0, :, :, :].reshape(-1, self.time_grid * self.time_grid)
            deform_all = self.deform_basis_mat(deform_all.T).reshape(self.time_grid * self.time_grid, -1, 9)
            gt_deform = torch.zeros_like(deform_all)
            gt_deform[:, :, 0] = 1.0
            gt_deform[:, :, 4] = 1.0
            total_loss = l1_loss(deform_all - gt_deform, torch.zeros_like(deform_all))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print("init deform loss: ", total_loss)

    """def init_gmm_seg(self):
        
        emptiness_source = self.emptiness_all[self.source_idx]
        xyzs_source = self.xyzs[emptiness_source, :]
        frame_time_source = torch.zeros((xyzs_source.shape[0], 1), device=self.xyzs.device)
        valid_xyzs = self.xyzs[emptiness_source, :].detach().cpu().numpy()

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=model.label_dim)
        label = kmeans.fit_predict(valid_xyzs)
        gt_label = torch.eye(model.label_dim, dtype=self.xyzs.dtype, device=self.xyzs.device)[label]

        # Initialize the optimizer
        beta1 = 0.9
        beta2 = 0.99
        grad_vars = model.get_optparam_groups(self.cfg, lr_scale=1.0)
        optimizer = torch.optim.Adam(grad_vars, betas=(beta1, beta2))

        time = self.time_set[self.source_idx] * torch.ones((1, 1), device=self.xyzs.device)
        frame_time = time.view((1, 1)).expand(self.xyzs.shape[0], 1)
        label_acc = 0.0
        iteration = 0
        while label_acc < 0.99:
            # Calculate the learning rate decay factor

            total_loss = 0.0
            for t in self.time_set:
                frame_time = t * torch.ones((xyzs_source.shape[0], 1), device=self.xyzs.device)
                label_features = model.compute_labelfeature(
                    xyzs_source, frame_time
                )
                pred_label = torch.nn.functional.softmax(label_features, 1)
                label_loss = - gt_label * torch.log(pred_label + 1e-8)
                label_loss = torch.sum(label_loss, 1).mean()
                _, pred_label_ind = torch.max(pred_label, 1)
                _, gt_label_ind = torch.max(gt_label, 1)
                label_acc = (pred_label_ind == gt_label_ind).float().mean()
                total_loss += label_loss 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iteration += 1"""

"""
class HexPlane(torch.nn.Module):

    def __init__(self, num_mixture=10):
        super().__init__()
        self.label_n_comp = [24, 24, 24]
        self.deform_n_comp = 24
        self.label_dim = num_mixture
        self.deform_dim = 9 * self.label_dim
        self.gridSize = [64, 64, 64]
        self.time_grid = 64
        self.align_corners = True
        self.init_scale = 0.1
        self.init_shift = 0.0
        self.device = "cuda"
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.init_planes(self.device)
        self.lr_org = 0.01

    def init_planes(self, device):
        self.label_plane, self.label_line_time = self.init_one_hexplane(
            self.label_n_comp, self.gridSize, device
        )
        self.deform_plane = self.init_one_plane(
            self.deform_n_comp, self.gridSize[0], device
        )
        self.label_basis_mat = torch.nn.Linear(self.label_n_comp[0], self.label_dim, bias=False).to(device)
        self.deform_basis_mat = torch.nn.Linear(self.deform_n_comp, self.deform_dim, bias=False).to(device)

    def init_one_plane(self, n_component, gridSize, device):
        plane_coef = []
        plane_coef.append(
            torch.nn.Parameter(
                self.init_scale
                * torch.randn(
                    (1, n_component, gridSize, gridSize)
                )
                + self.init_shift
            )
        )
        return torch.nn.ParameterList(plane_coef).to(device)
    
    def init_one_triplane(self, n_component, gridSize, device):
        plane_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )
        return torch.nn.ParameterList(plane_coef).to(device)
    
    def init_one_hexplane(self, n_component, gridSize, device):
        plane_coef, line_time_coef = [], []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn(
                        (1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])
                    )
                    + self.init_shift
                )
            )
            line_time_coef.append(
                torch.nn.Parameter(
                    self.init_scale
                    * torch.randn((1, n_component[i], gridSize[vec_id], self.time_grid))
                    + self.init_shift
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_time_coef).to(device)
    
    def compute_labelfeature(self, xyz_sampled: torch.Tensor, frame_time: torch.Tensor) -> torch.Tensor:
        plane_coord = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            .detach()
            .view(3, -1, 1, 2)
        )
        # line_time_coord: (3, B, 1, 2) coordinates for spatial-temporal planes, where line_time_coord[:, 0, 0, :] = [[t, z], [t, y], [t, x]].
        line_time_coord = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        line_time_coord = (
            torch.stack(
                (frame_time.expand(3, -1, -1).squeeze(-1), line_time_coord), dim=-1
            )
            .detach()
            .view(3, -1, 1, 2)
        )

        plane_feat, line_time_feat = [], []
        for idx_plane in range(len(self.label_plane)):
            # Spatial Plane Feature: Grid sampling on app plane[idx_plane] given coordinates plane_coord[idx_plane].
            plane_feat.append(
                F.grid_sample(
                    self.label_plane[idx_plane],
                    plane_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )
            # Spatial-Temoral Feature: Grid sampling on app line_time[idx_plane] plane given coordinates line_time_coord[idx_plane].
            line_time_feat.append(
                F.grid_sample(
                    self.label_line_time[idx_plane],
                    line_time_coord[[idx_plane]],
                    align_corners=self.align_corners,
                ).view(-1, *xyz_sampled.shape[:1])
            )

        plane_feat, line_time_feat = torch.stack(plane_feat), torch.stack(line_time_feat)
        
        inter = plane_feat * line_time_feat
        inter = torch.prod(inter, dim=0)
        inter = self.label_basis_mat(inter.T)  # Feature Projection
        return inter
    
    def compute_deformfeature(self, source_time, target_time):
        B = source_time.shape[0]
        source_time = source_time.view(1, -1, 1, 1)
        target_time = target_time.view(1, -1, 1, 1)

        time_coord = torch.cat((source_time, target_time), -1).detach()
        
        plane_feat = F.grid_sample(
            self.deform_plane[0],
            time_coord,
            align_corners=self.align_corners,
        )[0, :, :, 0] #.view(-1, 1) # (1, C, B, 1) -> (C, B)

        deform = self.deform_basis_mat(plane_feat.T) # (C, B) -> (B, C) -> (B, K*9)
        deform = deform.reshape(B, -1, 9).reshape(-1, 9) # (B, K*9) -> (B, K, 9) -> (B*K, 9)
        rotation = tools.compute_rotation_matrix_from_ortho6d(deform[:, 0:6]).reshape(B, -1, 3, 3) # [B*K, 6] -> [B*K, 3, 3] -> [B, K, 3, 3]
        translation = deform[:, 6:9].reshape(B, -1, 3) # [B*K, 3] -> [B, K, 3]

        if B == 1:
            deform = deform[0]
            rotation = rotation[0]
            translation = translation[0]

        results = {}
        results['deform'] = deform
        results['rotation'] = rotation
        results['translation'] = translation
        return results
    
    def compute_batch_deform_once(
        self, xyz_sampled: torch.Tensor, frame_time_source: torch.Tensor, frame_time_target: torch.Tensor, 
        gumbel=True, hard=False, tau=0.1, eval=False,
    ) -> torch.Tensor:
        assert frame_time_source.shape == frame_time_target.shape

        label_raw = self.compute_labelfeature(xyz_sampled, frame_time_source) # (N, K)

        once_results = self.compute_deformfeature(frame_time_source, frame_time_target)
        rotation = once_results['rotation'] # [N, K, 3, 3]
        translation = once_results['translation'] # [N, K, 3]
        xyz_sampled_deform = torch.matmul(rotation, xyz_sampled[:, None, :, None]).squeeze(3) + translation # [N, K, 3]

        if eval is True:
            _, label_ind = torch.max(label_raw, 1)
            label = torch.eye(label_raw.shape[-1], dtype=label_raw.dtype, device=label_raw.device)[label_ind]
        else:
            if gumbel:
                label = torch.nn.functional.gumbel_softmax(label_raw, tau=tau, hard=hard)
            else:
                label = torch.nn.functional.softmax(label_raw, dim=-1)
        
        xyz_sampled_deform = torch.sum(xyz_sampled_deform * label[:, :, None], 1)

        results = {}
        results['xyz_sampled_deform'] = xyz_sampled_deform
        results['label_raw'] = label_raw
        results['label'] = label
        results['rotation'] = rotation
        results['translation'] = translation

        return results
    
    def TV_loss_label(self, reg, reg2=None):
        total = 0
        if reg2 is None:
            reg2 = reg
        for idx in range(len(self.label_plane)):
            total = total + reg(self.label_plane[idx]) + reg2(self.label_line_time[idx])
        return total
    
    def TV_loss_deform(self, reg):
        total = reg(self.deform_plane[0])
        return total

    def get_optparam_groups(self, lr_scale=1.0):
        grad_vars = [
            {
                "params": self.deform_plane,
                "lr": lr_scale * self.lr_org,
                "lr_org": self.lr_org,
            },
            {
                "params": self.deform_basis_mat.parameters(),
                "lr": lr_scale * self.lr_org,
                "lr_org": self.lr_org,
            },
            {
                "params": self.label_plane,
                "lr": lr_scale * self.lr_org,
                "lr_org": self.lr_org,
            },
            {
                "params": self.label_line_time,
                "lr": lr_scale * self.lr_org,
                "lr_org": self.lr_org,
            },
            {
                "params": self.label_basis_mat.parameters(),
                "lr": lr_scale * self.lr_org,
                "lr_org": self.lr_org,
            },
        ]
        return grad_vars
"""
