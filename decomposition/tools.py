import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def point_features_to_volumes(points, volume_densities, point_features, grid_sizes=[100, 100, 100]):

    grid_sizes = torch.Tensor(grid_sizes).to(points.device)
    XYZ = (points.detach() + 1.0) * 0.5 * (grid_sizes - 1.0) 
    XYZ = torch.round(XYZ).long() # [K, N, 3]
    #print(XYZ.shape)
    X, Y, Z = XYZ.split(1, dim=-1) # [K, N], [K, N], [K, N]

    # linearized indices into the volume
    grid_sizes = grid_sizes.type_as(XYZ)
    idx_valid = (X * grid_sizes[1] + Y) * grid_sizes[2] + Z
    idx_valid = idx_valid[..., 0]
    #print(volume_densities.shape, idx_valid.shape, point_features.shape)

    # scatter add casts the votes into the weight accumulator
    # and the feature accumulator
    #volume_densities.scatter_add_(1, idx_valid, w_valid)
    volume_densities.scatter_add_(1, idx_valid, point_features)


"""def point_features_to_volumes(points, volume_densities, point_features, grid_sizes=[100, 100, 100]):

    grid_sizes = torch.Tensor(grid_sizes).to(points.device)
    XYZ = (points + 1.0) * 0.5 * (grid_sizes - 1.0)
    XYZ = torch.round(XYZ).long()
    print(XYZ.shape)
    X, Y, Z = XYZ.split(1, dim=1)

    grid_sizes = grid_sizes.type_as(XYZ)
    valid = (
        (0 <= X)
        * (X < grid_sizes[0])
        * (0 <= Y)
        * (Y < grid_sizes[1])
        * (0 <= Z)
        * (Z < grid_sizes[2])
    ).long()

    # linearized indices into the volume
    idx_valid = (X * grid_sizes[1] + Y) * grid_sizes[2] + Z
    w_valid = valid.type_as(volume_densities)
    idx_valid = idx_valid.transpose(0, 1)
    w_valid = w_valid.transpose(0, 1)
    print(volume_densities.shape, idx_valid.shape, w_valid.shape)

    # scatter add casts the votes into the weight accumulator
    # and the feature accumulator
    #volume_densities.scatter_add_(1, idx_valid, w_valid)
    volume_densities.scatter_add_(1, idx_valid, point_features)"""


#T_poses num*3
#r_matrix num*batch*3*3
def compute_batch_pose_from_rotation_matrix(T_pose, r_matrix):
    batch = r_matrix.shape[1]
    num = T_pose.shape[0]
    r_matrices = r_matrix.view(num, batch, 3, 3).contiguous().view(num * batch, 3, 3)
    src_poses = T_pose.view(num, 1, 3, 1).expand(num, batch, 3, 1).contiguous().view(num * batch, 3, 1)
    out_poses = torch.matmul(r_matrices, src_poses) # (num * batch, 3, 1)
        
    return out_poses.view(num, batch, 3)


#T_poses num*3
#r_matrix batch*3*3
def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    batch = r_matrix.shape[0]
    joint_num = T_pose.shape[0]
    r_matrices = r_matrix.view(batch,1, 3,3).expand(batch,joint_num, 3,3).contiguous().view(batch*joint_num,3,3)
    src_poses = T_pose.view(1,joint_num,3,1).expand(batch,joint_num,3,1).contiguous().view(batch*joint_num,3,1)
         
    out_poses = torch.matmul(r_matrices, src_poses) #(batch*joint_num)*3*1
        
    return out_poses.view(batch, joint_num,3)
    
# batch*n
def normalize_vector( v, return_mag =False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1) + 1e-8)# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


#in batch*6
#out batch*5
def stereographic_project(a):
    dim = a.shape[1]
    a = normalize_vector(a)
    out = a[:,0:dim-1]/(1-a[:,dim-1])
    return out
	


#in a batch*5, axis int
def stereographic_unproject(a, axis=None):
    """
	Inverse of stereographic projection: increases dimension by one.
	"""
    batch=a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a,2).sum(1) #batch
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1]+1).cuda()) #batch*6
    unproj = 2*a/(s2+1).view(batch,1).repeat(1,a.shape[1]) #batch*5
    if(axis>0):
        ans[:,:axis] = unproj[:,:axis] #batch*(axis-0)
    ans[:,axis] = (s2-1)/(s2+1) #batch
    ans[:,axis+1:] = unproj[:,axis:]	 #batch*(5-axis)		# Note that this is a no-op if the default option (last axis) is used
    return ans


#a batch*5
#out batch*3*3
def compute_rotation_matrix_from_ortho5d(a):
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2)+1, np.sqrt(2)+1, np.sqrt(2)]) #3
    proj_scale = torch.autograd.Variable(torch.FloatTensor(proj_scale_np).cuda()).view(1,3).repeat(batch,1) #batch,3
    
    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)#batch*4
    norm = torch.sqrt(torch.pow(u[:,1:],2).sum(1)) #batch
    u = u/ norm.view(batch,1).repeat(1,u.shape[1]) #batch*4
    b = torch.cat((a[:,0:2], u),1)#batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix


#quaternion batch*4
def compute_rotation_matrix_from_quaternion( quaternion):
    batch=quaternion.shape[0]
    
    
    quat = normalize_vector(quaternion).contiguous()
    
    qw = quat[...,0].contiguous().view(batch, 1)
    qx = quat[...,1].contiguous().view(batch, 1)
    qy = quat[...,2].contiguous().view(batch, 1)
    qz = quat[...,3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix
    
#axisAngle batch*4 angle, x,y,z
def compute_rotation_matrix_from_axisAngle( axisAngle):
    batch = axisAngle.shape[0]
    
    theta = torch.tanh(axisAngle[:,0])*np.pi #[-180, 180]
    sin = torch.sin(theta*0.5)
    axis = normalize_vector(axisAngle[:,1:4]) #batch*3
    qw = torch.cos(theta*0.5)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*3 (x,y,z)*theta
def compute_rotation_matrix_from_Rodriguez( rod):
    batch = rod.shape[0]
    
    axis, theta = normalize_vector(rod, return_mag=True)
    
    sin = torch.sin(theta)
    
    
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*3 a,b,c
def compute_rotation_matrix_from_hopf( hopf):
    batch = hopf.shape[0]
    
    theta = (torch.tanh(hopf[:,0])+1.0)*np.pi/2.0 #[0, pi]
    phi   = (torch.tanh(hopf[:,1])+1.0)*np.pi     #[0,2pi)
    tao   = (torch.tanh(hopf[:,2])+1.0)*np.pi     #[0,2pi)
    
    qw = torch.cos(theta/2)*torch.cos(tao/2)
    qx = torch.cos(theta/2)*torch.sin(tao/2)
    qy = torch.sin(theta/2)*torch.cos(phi+tao/2)
    qz = torch.sin(theta/2)*torch.sin(phi+tao/2)
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix
    

#euler batch*4
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)  
def compute_rotation_matrix_from_euler(euler):
    batch=euler.shape[0]
        
    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
    c2=torch.cos(euler[:,2]).view(batch,1)#batch*1 
    s2=torch.sin(euler[:,2]).view(batch,1)#batch*1 
    c3=torch.cos(euler[:,1]).view(batch,1)#batch*1 
    s3=torch.sin(euler[:,1]).view(batch,1)#batch*1 
        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix


#euler_sin_cos batch*6
#output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)  
def compute_rotation_matrix_from_euler_sin_cos(euler_sin_cos):
    batch=euler_sin_cos.shape[0]
    
    s1 = euler_sin_cos[:,0].view(batch,1)
    c1 = euler_sin_cos[:,1].view(batch,1)
    s2 = euler_sin_cos[:,2].view(batch,1)
    c2 = euler_sin_cos[:,3].view(batch,1)
    s3 = euler_sin_cos[:,4].view(batch,1)
    c3 = euler_sin_cos[:,5].view(batch,1)

        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    
    
    return theta


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_angle_from_r_matrices(m):
    
    batch=m.shape[0]
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    theta = torch.acos(cos)
    
    return theta
    
def get_sampled_rotation_matrices_by_quat(batch):
    #quat = torch.autograd.Variable(torch.rand(batch,4).cuda())
    quat = torch.autograd.Variable(torch.randn(batch, 4).cuda())
    matrix = compute_rotation_matrix_from_quaternion(quat)
    return matrix
    
def get_sampled_rotation_matrices_by_hpof(batch):
    
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,1, batch)*np.pi).cuda()) #[0, pi]
    phi   =  torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,2,batch)*np.pi).cuda())      #[0,2pi)
    tao   = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0,2,batch)*np.pi).cuda())      #[0,2pi)
    
    
    qw = torch.cos(theta/2)*torch.cos(tao/2)
    qx = torch.cos(theta/2)*torch.sin(tao/2)
    qy = torch.sin(theta/2)*torch.cos(phi+tao/2)
    qz = torch.sin(theta/2)*torch.sin(phi+tao/2)
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

#axisAngle batch*4 angle, x,y,z
def get_sampled_rotation_matrices_by_axisAngle( batch, return_quaternion=False):
    
    theta = torch.autograd.Variable(torch.FloatTensor(np.random.uniform(-1,1, batch)*np.pi).cuda()) #[0, pi] #[-180, 180]
    sin = torch.sin(theta)
    axis = torch.autograd.Variable(torch.randn(batch, 3).cuda())
    axis = normalize_vector(axis) #batch*3
    qw = torch.cos(theta)
    qx = axis[:,0]*sin
    qy = axis[:,1]*sin
    qz = axis[:,2]*sin
    
    quaternion = torch.cat((qw.view(batch,1), qx.view(batch,1), qy.view(batch,1), qz.view(batch,1)), 1 )
    
    # Unit quaternion rotation matrices computatation  
    xx = (qx*qx).view(batch,1)
    yy = (qy*qy).view(batch,1)
    zz = (qz*qz).view(batch,1)
    xy = (qx*qy).view(batch,1)
    xz = (qx*qz).view(batch,1)
    yz = (qy*qz).view(batch,1)
    xw = (qx*qw).view(batch,1)
    yw = (qy*qw).view(batch,1)
    zw = (qz*qw).view(batch,1)
    
    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    if(return_quaternion==True):
        return matrix, quaternion
    else:
        return matrix


import numpy
from chainer import cuda


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=numpy.int32,
                            distances_dtype=numpy.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)
    return indices, distances