from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class System_Config:
    seed: int = 20220401
    basedir: str = "./log"
    ckpt: Optional[str] = None
    progress_refresh_rate: int = 10
    vis_every: int = 10000
    add_timestamp: bool = True


@dataclass
class Data_Config:
    dataset_name: str = "dnerf" 
    num_sample_total: int = 15
    opacity_threshold: float = 0.01


@dataclass
class Model_Config:
    N_voxel_init: int = 64 * 64 * 64  # initial voxel number
    N_voxel_final: int = 200 * 200 * 200  # final voxel number
    voxel_size: List[int] = field(default_factory=lambda: [64, 64, 64])
    voxel_grid: List[int] = field(default_factory=lambda: [64, 64, 64])
    time_grid: int = 64

    # Plane Initialization
    label_n_comp: List[int] = field(default_factory=lambda: [24, 24, 24])
    deform_n_comp: int = 24
    max_part_num: int = 40
    init_scale: float = 0.1
    init_shift: float = 0.0

    # Sampling
    align_corners: bool = True

    time_grid_init: int = 16
    time_grid_final: int = 120 #24
    voxel_grid_init: int = 64 #200 #64
    voxel_grid_final: int = 200
    upsample_list: List[int] = field(default_factory=lambda: [2000, 4000, 6000]) #List[int] = field(default_factory=lambda: [1000, 2000, 3000])
    shrink_list: List[int] = field(default_factory=lambda: [2000, 4000, 6000, 8000])
    group_merge_list: List[int] = field(default_factory=lambda: [5000, 9000])

@dataclass
class Optim_Config:
    # Learning Rate
    lr_label_grid: float = 0.02
    lr_deform_grid: float = 0.02
    lr_label_nn: float = 0.001
    lr_deform_nn: float = 0.001

    # Optimizer, Adam deault
    beta1: float = 0.9
    beta2: float = 0.99
    lr_decay_type: str = "exp"  # choose from "exp" or "cosine" or "linear" 
    lr_decay_target_ratio: float = 0.01 #0.01
    lr_decay_step: int = -1
    lr_upsample_reset: bool = True

    gumbel: bool = False
    hard: bool = True
    tau: float = 1.0
    eval: bool = False

    turn_to_softmax: int = 2500

    loss_mode: str = "voxel"
    adj_loss_version: int = 4

    batch_size: int = 1
    n_iters: int = 12000
    ### original 10000

    sym_loss_weight: float = 1e-5
    tv_label_loss_weight: float = 1e-3 #1e-6 # 0.1# 0.0001 # voxel for 0.1
    ###################################### 1e-4
    tv_deform_loss_weight: float = 1e-5
    recon_loss_weight: float = 1e-4
    diag_loss_weight: float = 0.0001
    nst_loss_weight: float = 1e-4 #1.0 #1 #0.0001 #1.0 #001 #1.0
    ######################################### 0.001!!!!!!!!
    ### original value is 0.01 and adj_loss_version: 5
    ### o.1 is not bad
    num_nst_points: int = 10
    ######################### 50
    
    one_source: bool = True
    logfolder: str = "test"

    vis: bool = False
    vis_every: int = 50
    vis_time_step: int = 10


@dataclass
class Config:
    config: Optional[str] = None
    expname: str = "default"

    #render_catergory: str = "hellwarrior"
    #render_scene_idx: {"hellwarrior"}
    render_only: bool = False
    render_train: bool = False
    render_test: bool = True
    render_path: bool = False

    source_idx: int = -1
    train_render_scene_idx: int = 0
    test_render_scene_idx: int = 0

    systems: System_Config = System_Config()
    model: Model_Config = Model_Config()
    data: Data_Config = Data_Config()
    optim: Optim_Config = Optim_Config()
