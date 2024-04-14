import torch

@torch.no_grad()
#def get_state_at_time(pc, viewpoint_camera, seg_model, source_cam):    
def get_state_at_time(pc, viewpoint_camera, seg_model):    
    means3D = pc.get_xyz
    #source_time = torch.tensor(source_cam.time).to(means3D.device).repeat(means3D.shape[0], 1)
    source_time = torch.tensor(seg_model.time_source).to(means3D.device).repeat(means3D.shape[0], 1)
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = pc._scaling
    rotations = pc._rotation
    cov3D_precomp = None

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, rotations, opacity, shs, source_time)
    
    target_time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
    means3D_final, rotations_final, _ = seg_model(means3D_final, rotations_final, source_time, target_time, eval=False)

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    frame_time_source = torch.zeros_like(means3D_final[:, 0:1])
    label_feat = seg_model.compute_labelfeature(means3D_final, frame_time_source, xyz_normalize=True)
    _, label_final = torch.max(label_feat, 1)
            
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)

    return means3D_final, scales_final, rotations_final, opacity_final, shs_final, label_final