import numpy as np
import os
import trimesh
import torch
from scipy.spatial.transform import Rotation as R
from matplotlib import cm

from PIL import Image, ImageColor

import colorsys

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0: #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb


class TestVisualizer():
    def __init__(self, sampler_3d, time_step=3, vis_seg=True, vis_recon=True, save=True, folder_name=None, deform_arg=None, one_source=False):
        self.sampler_3d = sampler_3d
        self.time_step = time_step
        self.color_list = np.random.randint(255, size=(40, 3))
        #colormap = sorted(set(ImageColor.colormap.values()))
        #self.color_list = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
        self.vis_seg = vis_seg
        self.vis_recon = vis_recon
        self.one_source = one_source

        camera_rotation = np.eye(4)
        #rotation = R.from_euler('xyz', [110, 0, 0], degrees=True).as_matrix() # hellwarrior
        rotation = R.from_euler('xyz', [70, 0, -30], degrees=True).as_matrix() # hellwarrior
        #rotation = R.from_euler('xyz', [110, 0, -30], degrees=True).as_matrix() # hook
        #rotation = R.from_euler('xyz', [70, 0, 0], degrees=True).as_matrix() # gundam
        camera_rotation[0:3, 0:3] = rotation
        camera = trimesh.scene.Camera(fov=(np.pi/4.0, np.pi/4.0))
        transform = camera.look_at([[0, 0.0, 0.0]], rotation=camera_rotation, distance=4.0)
        #transform = camera.look_at([[0, 0.0, 0.0]], rotation=camera_rotation, distance=2.0) #gundam
        self.transform = transform

        camera_rotation = np.eye(4)
        #rotation = R.from_euler('xyz', [70, 0, 0], degrees=True).as_matrix() #leg
        rotation = R.from_euler('xyz', [70, 0, -30], degrees=True).as_matrix() # hook
        camera_rotation[0:3, 0:3] = rotation
        camera = trimesh.scene.Camera(fov=(np.pi/4.0, np.pi/4.0))
        transform = camera.look_at([[0, 0.0, 0.0]], rotation=camera_rotation, distance=4.0)
        self.transform_multi = transform

        self.deform_arg = {'eval': True} if deform_arg is None else deform_arg
        
        self.save = save
        self.folder_name = folder_name
        if save:
            os.makedirs(f"{self.folder_name}/reconstruction", exist_ok=True)
            os.makedirs(f"{self.folder_name}/reconstruction_reverse", exist_ok=True)
            os.makedirs(f"{self.folder_name}/segmentation", exist_ok=True)

    def vis(self, model, iter=0, interactive=False):
        if self.one_source:
            self.vis_one_source(model, iter=iter, interactive=interactive)
        else:
            self.vis_multi_source(model, iter=iter, interactive=interactive)

    def raw_label_to_vector(self, label_raw, deform_arg, remaining_idx):
        idx = remaining_idx  

        if idx is not None:
            label = label_raw[:, idx] # [N, IDX]
        else:
            label = label_raw

        if deform_arg["eval"] is True:
            _, label_ind = torch.max(label, 1)
            label = torch.eye(label.shape[-1], dtype=label.dtype, device=label.device)[label_ind]
        else:
            if deform_arg["gumbel"]:
                label = torch.nn.functional.gumbel_softmax(label, tau=deform_arg["tau"], hard=deform_arg["hard"])
            else:
                label = torch.nn.functional.softmax(label, dim=-1)

        return label

    def vis_one_source(self, model, iter=0, interactive=False):
        emptiness_all = self.sampler_3d.emptiness_all
        time_set = self.sampler_3d.time_set
        xyzs = self.sampler_3d.xyzs
        source_idx = self.sampler_3d.source_idx 
        color_list = self.color_list
        deform_arg = self.deform_arg
        transform = self.transform
        transform_multi = self.transform_multi
        folder_name = self.folder_name
        
        emptiness_source = emptiness_all[source_idx]
        time_source = time_set[source_idx]
        
        xyzs_source = xyzs[emptiness_source, :]
        label_feat = model.compute_labelfeature(xyzs_source, torch.zeros_like(xyzs_source[:, 0:1]))

        #self.color_list = np.random.randint(255, size=(label_feat.shape[1], 3))
        self.color_list = []
        for i in range(label_feat.shape[1]):
            self.color_list.append(id2rgb(i))
        color_list = self.color_list

        valid_xyzs = xyzs_source.view(-1, 3).detach().cpu().numpy()
        colors = np.zeros_like(valid_xyzs)
        _, pred_label_ind = torch.max(label_feat, 1)
        for i in range(valid_xyzs.shape[0]):
            colors[i] = color_list[pred_label_ind[i]] / 255.0

        pc = trimesh.PointCloud(valid_xyzs, colors=colors)
        scene = trimesh.Scene()
        scene.camera_transform = transform
        scene.add_geometry(pc)
        if interactive:
            scene.show()

        if self.save:
            file_name = f"{folder_name}/segmentation/segmentation_%d.png"%(iter)
            data = scene.save_image(resolution=(1024, 1024), visible=True)
            with open(file_name, 'wb') as f:
                f.write(data)
                f.close()

        for t in range(0, emptiness_all.shape[0], self.time_step):
            emptiness_target = emptiness_all[t]
            time_target = time_set[t]

            frame_time_source = time_source * torch.ones((*xyzs.shape[:-1], 1), device=xyzs.device)
            frame_time_target = time_target * torch.ones((*xyzs.shape[:-1], 1), device=xyzs.device)

            xyzs_deformed, _, _, _, _ = model.compute_batch_deform_once(xyzs[emptiness_source, :], frame_time_source[emptiness_source, :], 
                                                                                    frame_time_target[emptiness_source, :], **deform_arg)

            valid_xyzs = xyzs_deformed.view(-1, 3).detach().cpu().numpy()
            pc = trimesh.PointCloud(valid_xyzs, colors=colors)
            pc2 = trimesh.PointCloud(xyzs[emptiness_target, :].detach().cpu().numpy(), colors=[0.9, 0.9, 0.9])
            scene = trimesh.Scene()
            scene.camera_transform = transform_multi
            scene.add_geometry(pc)
            scene.add_geometry(pc2)
            if interactive:
                scene.show()

            if self.save:
                file_name = f"{folder_name}/reconstruction/reconstruction_%d_%d.png"%(iter, t)
                data = scene.save_image(resolution=(1024, 1024), visible=True)
                with open(file_name, 'wb') as f:
                    f.write(data)
                    f.close()
        
    def vis_multi_source(self, model, iter=0, interactive=False):
        emptiness_all = self.emptiness_all
        time_set = self.time_set
        xyzs = self.xyzs
        color_list = self.color_list
        deform_arg = self.deform_arg
        transform = self.transform
        transform_multi = self.transform_multi
        folder_name = self.folder_name

        """if vis_remaining_idx is not None:
            hsv = cm.get_cmap('hsv', len(vis_remaining_idx))
            for i in range(len(vis_remaining_idx)):
                color_list[i, :] = np.asarray(hsv(i / (len(vis_remaining_idx) - 1.0))[0:3]) * 255.0"""

        if self.vis_seg:
            for t in range(0, self.time_grid, self.time_step):
                time = time_set[t] * torch.ones((*xyzs.shape[:-1], 1), device=xyzs.device)
                valid_labels = model.compute_labelfeature(
                    xyzs[emptiness_all[t], :], time[emptiness_all[t], :]
                )

                valid_xyzs = xyzs[emptiness_all[t], :].view(-1, 3).detach().cpu().numpy()
                colors = np.zeros_like(valid_xyzs)

                scene = trimesh.Scene()
                scene.camera_transform = transform 

                _, pred_label_ind = torch.max(valid_labels, 1)
                print(torch.min(pred_label_ind), torch.max(pred_label_ind))
                for i in range(valid_xyzs.shape[0]):
                    colors[i] = color_list[pred_label_ind[i]] / 255.0

                pc = trimesh.PointCloud(valid_xyzs, colors=colors)
                scene.add_geometry(pc)
                if interactive:
                    scene.show()

                #file_name = f"results_{folder_name}/segmentation/segmentation_%d_%d.png"%(iter, t)
                file_name = f"{folder_name}/segmentation/segmentation_%d_%d.png"%(iter, t)
                data = scene.save_image(resolution=(1024, 1024), visible=True)
                with open(file_name, 'wb') as f:
                    f.write(data)
                    f.close()
                
                """for valid_idx in vis_remaining_idx:
                    sample_idx = np.argwhere(pred_label_ind.detach().cpu().numpy() == valid_idx)
                    if sample_idx.shape[0] == 0:
                        continue
                    sample_valid_xyzs = valid_xyzs[sample_idx[:, 0], :]
                    print(sample_valid_xyzs.shape)

                    scene = trimesh.Scene()
                    scene.camera_transform = transform
                    pc = trimesh.PointCloud(sample_valid_xyzs)
                    scene.add_geometry(pc)
                    if interactive:
                        scene.show()

                    file_name = f"results_{folder_name}/segmentation/segmentation_%d_%d_%d.png"%(iter, t, valid_idx)
                    data = scene.save_image(resolution=(1024, 1024), visible=True)
                    with open(file_name, 'wb') as f:
                        f.write(data)
                        f.close()"""

                """for tau in [0.1]:
                    valid_labels = torch.nn.functional.gumbel_softmax(valid_labels[:, vis_remaining_idx], tau=tau, hard=True)
                    _, pred_label_ind = torch.max(valid_labels, 1)
                    #print(torch.min(pred_label_ind), torch.max(pred_label_ind))
                    for i in range(valid_xyzs.shape[0]):
                        colors[i] = color_list[pred_label_ind[i]] / 255.0

                    pc = trimesh.PointCloud(valid_xyzs, colors=colors)
                    scene.add_geometry(pc)
                    if interactive:
                        scene.show()

                    file_name = f"results_{folder_name}/segmentation/segmentation_%d_%d.png"%(iter, t)
                    data = scene.save_image(resolution=(1024, 1024), visible=True)
                    with open(file_name, 'wb') as f:
                        f.write(data)
                        f.close()"""

        if self.vis_recon:
            for t in range(0, self.time_grid, self.time_step):
                for t2 in range(0, self.time_grid, self.time_step):
                #for t2 in range(self.time_grid // 2, self.time_grid // 2 + 1, 1):
                    emptiness_source = emptiness_all[t2]
                    emptiness_target = emptiness_all[t]
                    time_source = time_set[t2]
                    time_target = time_set[t]

                    frame_time_source = time_source * torch.ones((*xyzs.shape[:-1], 1), device=xyzs.device)
                    frame_time_target = time_target * torch.ones((*xyzs.shape[:-1], 1), device=xyzs.device)

                    once_results = model.compute_batch_deform_once(xyzs[emptiness_source, :], frame_time_source[emptiness_source, :], 
                                                                    frame_time_target[emptiness_source, :], idx=vis_remaining_idx, **deform_arg)
        
                    #once_results = model.compute_deform_once(xyzs[emptiness_source, :], time_source, time_target, idx=vis_remaining_idx, inverse=False, **deform_arg_eval)
                    xyzs_deformed = once_results['xyz_sampled_deform']
                    valid_labels = once_results['label_raw']
                    #valid_labels = once_results['label']

                    valid_xyzs = xyzs[emptiness_source, :].view(-1, 3).detach().cpu().numpy()
                    colors = np.zeros_like(valid_xyzs)
                    _, pred_label_ind = torch.max(valid_labels[:, vis_remaining_idx], 1)
                    #_, pred_label_ind = torch.max(valid_labels, 1)
                    for i in range(valid_xyzs.shape[0]):
                        colors[i] = color_list[pred_label_ind[i]] / 255.0
                    pc = trimesh.PointCloud(xyzs_deformed.detach().cpu().numpy(), colors=colors)
                    pc2 = trimesh.PointCloud(xyzs[emptiness_target, :].detach().cpu().numpy(), colors=[0.5, 0.5, 0.5])
                    scene = trimesh.Scene()
                    scene.camera_transform = transform 
                    scene.add_geometry(pc)
                    scene.add_geometry(pc2)
                    if interactive:
                        scene.show()

                    pc = trimesh.PointCloud(xyzs_deformed.detach().cpu().numpy() + np.array([[-1.0, 0.0, 0.0]]), colors=colors)
                    pc2 = trimesh.PointCloud(xyzs[emptiness_target, :].detach().cpu().numpy() + np.array([[1.0, 0.0, 0.0]]), colors=[0.5, 0.5, 0.5])
                    scene = trimesh.Scene()
                    scene.camera_transform = transform_multi
                    scene.add_geometry(pc)
                    scene.add_geometry(pc2)
                    if interactive:
                        scene.show()

                    if self.save:
                        #file_name = f"results_{folder_name}/reconstruction/reconstruction_%d_%d_%d.png"%(iter, t2, t)
                        file_name = f"{folder_name}/reconstruction/reconstruction_%d_%d_%d.png"%(iter, t2, t)
                        data = scene.save_image(resolution=(1024, 1024), visible=True)
                        with open(file_name, 'wb') as f:
                            f.write(data)
                            f.close()

                    """scene = trimesh.Scene()
                    scene.camera_transform = transform 
                    scene.add_geometry(pc)
                    if interactive:
                        scene.show()

                    if self.save:
                        file_name = f"results_{folder_name}/reconstruction/reconstruction_%d_%d_%d_1.png"%(iter, t2, t)
                        data = scene.save_image(resolution=(1024, 1024), visible=True)
                        with open(file_name, 'wb') as f:
                            f.write(data)
                            f.close()"""
                    
            for t2 in range(0, self.time_grid, self.time_step):
                #for t2 in range(0, self.time_grid, self.time_step):
                for t in range(self.time_grid // 2, self.time_grid // 2 + 1, 1):
                    emptiness_source = emptiness_all[t2]
                    emptiness_target = emptiness_all[t]
                    time_source = time_set[t2]
                    time_target = time_set[t]

                    frame_time_source = time_source * torch.ones((*xyzs.shape[:-1], 1), device=xyzs.device)
                    frame_time_target = time_target * torch.ones((*xyzs.shape[:-1], 1), device=xyzs.device)

                    once_results = model.compute_batch_deform_once(xyzs[emptiness_source, :], frame_time_source[emptiness_source, :], 
                                                                    frame_time_target[emptiness_source, :], idx=vis_remaining_idx, **deform_arg)
        
                    #once_results = model.compute_deform_once(xyzs[emptiness_source, :], time_source, time_target, idx=vis_remaining_idx, inverse=False, **deform_arg_eval)
                    xyzs_deformed = once_results['xyz_sampled_deform']
                    valid_labels = once_results['label_raw']
                    #valid_labels = once_results['label']

                    valid_xyzs = xyzs[emptiness_source, :].view(-1, 3).detach().cpu().numpy()
                    colors = np.zeros_like(valid_xyzs)
                    #_, pred_label_ind = torch.max(valid_labels[:, vis_remaining_idx], 1)
                    _, pred_label_ind = torch.max(valid_labels, 1)
                    for i in range(valid_xyzs.shape[0]):
                        colors[i] = color_list[pred_label_ind[i]] / 255.0
                    pc = trimesh.PointCloud(xyzs_deformed.detach().cpu().numpy(), colors=colors)
                    pc2 = trimesh.PointCloud(xyzs[emptiness_target, :].detach().cpu().numpy(), colors=[0.5, 0.5, 0.5])
                    scene = trimesh.Scene()
                    scene.camera_transform = transform 
                    scene.add_geometry(pc)
                    scene.add_geometry(pc2)
                    if interactive:
                        scene.show()

                    pc = trimesh.PointCloud(xyzs_deformed.detach().cpu().numpy() + np.array([[-1.0, 0.0, 0.0]]), colors=colors)
                    pc2 = trimesh.PointCloud(xyzs[emptiness_target, :].detach().cpu().numpy() + np.array([[1.0, 0.0, 0.0]]), colors=[0.5, 0.5, 0.5])
                    scene = trimesh.Scene()
                    scene.camera_transform = transform_multi
                    scene.add_geometry(pc)
                    scene.add_geometry(pc2)
                    if interactive:
                        scene.show()

                    if self.save:
                        #file_name = f"results_{folder_name}/reconstruction/reconstruction_%d_%d_%d.png"%(iter, t2, t)
                        file_name = f"{folder_name}/reconstruction/reconstruction_%d_%d_%d.png"%(iter, t2, t)
                        data = scene.save_image(resolution=(1024, 1024), visible=True)
                        with open(file_name, 'wb') as f:
                            f.write(data)
                            f.close()
                    
