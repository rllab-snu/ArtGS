import torch
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import copy
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"] = ["Times New Roman"]

from PIL import Image, ImageColor
import random
colormap = sorted(set(ImageColor.colormap.values()))
colormap_shuffle_copy = colormap[0:-1]
random.shuffle(colormap_shuffle_copy)
colormap[0:-1] = colormap_shuffle_copy
color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)

color_palette = np.random.randint(255, size=(43, 3)) / 255.0
color_palette[-1] = [1.0, 1.0, 1.0] 

def visualize_label_numpy(label, bg_white=True):
    """
    label: (H, W, K)
    """ 
    #colormap = sorted(set(ImageColor.colormap.values()))
    #color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)

    #label+print(np.max(label_map.numpy(), -1))
    #acc_map = np.max(label, -1) > 0

    if bg_white:
        acc = np.sum(label, axis=-1)
        label = np.concatenate((label, 1.0 - acc[..., None]), axis=-1)
        label = np.clip(label, 0, 1)
        background_label_idx = label.shape[-1]
        label = np.argmax(label, -1)
        label[label == background_label_idx - 1] = -1
    else:
        label = np.clip(label, 0, 1)
        label = np.argmax(label, -1)

    #label_pil = Image.fromarray(color_palette[label])
    label_pil = color_palette[label]
    return label_pil


@torch.no_grad()
def render_training_image_and_label_articulation(scene, gaussians, viewpoints, render_func, label_render_func, pipe, background, seg_model, stage, iteration, time_now, dataset_type, eval=False):
    def render(gaussians, viewpoint, path, scaling, cam_type):
        # scaling_copy = gaussians._scaling
        render_pkg = render_func(viewpoint, gaussians, pipe, background, seg_model, eval=eval, stage=stage, cam_type=cam_type)
        label1 = f"iteration: {iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"]
        render_pkg = label_render_func(viewpoint, gaussians, pipe, background, seg_model, eval=eval, stage=stage, cam_type=cam_type)
        segmentation = render_pkg["render"]
        #print(segmentation.shape)
        if dataset_type == "PanopticSports":
            gt_np = viewpoint["image"].permute(1, 2, 0).cpu().numpy()
        else:
            gt_np = viewpoint.original_image.permute(1, 2, 0).cpu().numpy()
        image_np = image.detach().permute(1, 2, 0).cpu().numpy()
        segmentation_np = segmentation.detach().permute(1, 2, 0).cpu().numpy()
        #print("before label: ", segmentation_np.shape)
        segmentation_np = visualize_label_numpy(segmentation_np)
        image_np = np.concatenate((gt_np, image_np, segmentation_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype("uint8")) 

        draw1 = ImageDraw.Draw(image_with_labels)
        font = ImageFont.truetype("./utils/TIMES.TTF", size=40)  
        text_color = (255, 0, 0) 
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)  # 右上角坐标

        draw1.text(label1_position, label1, fill=text_color, font=font)
        image_with_labels.save(path)
    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    image_path = os.path.join(render_base_path, "images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path, f"{iteration}_{idx}.jpg")
        render(gaussians,viewpoints[idx], image_save_path,scaling=1, cam_type=dataset_type)