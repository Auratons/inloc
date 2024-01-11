#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
sys.path.append('/home/kremeto1/inloc/functions/inLocCIIRC_utils/projectMesh')
import projectMesh
sys.path.append('/home/kremeto1/hololens_mapper')
from src.utils.UtilsMath import UtilsMath, renderDepth
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
import scipy.io as sio
import json
from pathlib import Path
from itertools import repeat
import pandas as pd
from pyntcloud import PyntCloud
import cv2

import pyrender
import trimesh
import numpy as np
from pyrender.constants import RenderFlags

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,graphics,utility,video'
os.environ["PYOPENGL_PLATFORM"] = "egl"
# When ran with SLURM on a multigpu node, scheduled on other than GPU0, we need
# to set this or we get an egl initialization error.
os.environ["EGL_DEVICE_ID"] = os.environ.get("SLURM_JOB_GPUS", "0").split(",")[0]


# In[ ]:


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(28, 28 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


# In[ ]:


splatting = 'pipeline-inloc-conv5-splatting'
pyrender = 'pipeline-inloc-conv5-pyrender'
lifted = 'pipeline-inloc-lifted-conv5-pyrender'

prefix = '/home/kremeto1/inloc/datasets/'
suffix = '/candidate_renders/IMG_0819/cutout_DUC_cutout_019_150_-30_color.png'

for path in Path(prefix + splatting + "/candidate_renders").iterdir():
    if path.is_dir():
        print(path)
        im = cv2.imread(f'/home/kremeto1/neural_rendering/datasets/raw/inloc/query/same_db_size_conditionally_rotated/{path.name}.JPG')
        for img in path.glob("*_color.png"):
            images = [cv2.imread(str(path / img.name)) for i in [splatting, pyrender, lifted]] + [im]
            display_images(images, cols=4)
    

# images = [cv2.imread(prefix + i + suffix) for i in [splatting, pyrender, lifted]]
# display_images(images, cols=3)


# In[ ]:


d = np.load('/home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_splatting/CSE3/000/cse_cutout_000_0_0_depth.png.npy')
d = np.clip(d * 255 / 100, 0, 255).astype(np.uint8)
cv2.imwrite('/home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_splatting/CSE3/000/cse_cutout_000_0_0_depth_uin8.png', d)


# In[ ]:


mat = sio.loadmat('/home/kremeto1/inloc/datasets/hagia-pyrender/densePE_top100_shortlist.mat')


# In[ ]:


query_path = str(mat['ImgList'][0][0][0][0])
candidate_paths = [str(i[0]) for i in mat['ImgList'][0][0][1][0]]
candidate_scores = mat['ImgList'][0][0][2][0]
candidate_transforms = [i for i in mat['ImgList'][0][0][3][0]]
query_path


# In[ ]:


# log_base = '/home/kremeto1/inloc/dvc/pipeline-hagia-pyrender/logs/inloc_algo_2022667_stats'
# log_base = '/home/kremeto1/inloc/dvc/pipeline-hagia-conv5-pyrender/logs/inloc_algo_2022668_stats'
log_base = '/home/kremeto1/inloc/dvc/pipeline-inloc-conv5-pyrender/logs/inloc_algo_2125436_stats'
import re
with open(f'{log_base}.txt', 'r') as f:
    lines = f.readlines()
    
new_lines = []
for line in lines:
    m = re.match('([0-9:]*)[ ]*PID S     TIME %CPU   RSS COMMAND', line)
    if m is not None:
        time = m.group(1)
    m = re.match('.*MATLAB.*', line)
    if m is not None:
        line = f"{time} {line.strip()}"
        line = re.sub('[ ]+', ' ', line)
        line = ' '.join(line.split(' ')[:9]) + '\n'
        new_lines.append(line)

with open(f'{log_base}_processed.txt', 'w') as f:
    f.writelines(new_lines)


# In[ ]:


ff = pd.read_csv(f'{log_base}_processed.txt', sep=' ', header=None, names=[f"col{i}" for i in range(9)])
f = ff.groupby(["col0"]).sum()
plt.figure(figsize=(18, 12))
(ff.groupby(["col0"]).sum()["col5"] / 1e6).plot()


# In[ ]:



def reverse_dict(d):
    return {v: k for k, v in d.items()}

input_mapping = '/nfs/projects/artwin/experiments/hololens_mapper/joined_dataset/mapping.txt'
input_root = Path('/nfs/projects/artwin/experiments/hololens_mapper/joined_dataset/train')

# Get mapping from reference images in joined dataset for nriw training to
# reference images in a concrete nriw training dataset from which the joined
# one was generated.
with open(input_mapping, 'r') as f:
    sub_mapping_1 = json.load(f)
sub_mapping_1 = reverse_dict(sub_mapping_1)

# Get roots of nriw training datasets from which a joined dataset was generated.
source_roots = {}
for path in [Path(k).parent.parent for k in sub_mapping_1.values()]:
    source_roots[str(path)] = 1
source_roots = list(source_roots.keys())

# Get mapping from partial nriw training datasets to source references
# generated by matlab from artwin panoramas
sub_mappings_2 = []
for mp in [Path(root) / "mapping.txt" for root in source_roots]:
    with open(mp, 'r') as f:
        lines = f.readlines()
        # Filter lines only for used source (train/val/test)
        lines = filter(lambda line: str(input_root.name).upper() in line, lines)
        lines = [str.join(" ", line.split(" ")[:-1]) for line in lines]  # Get rid of trailing TRAIN/DEV/TEST
        line_tuples = [tuple(line.split(" -> ")) for line in lines]  # (source, dest)
        sub_map = {}
        for wut in zip(line_tuples):
            v, k = wut[0]
            sub_map[str(Path(mp).parent / str(input_root.name) / f"{int(k):04n}_reference.png")] = f"/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/{str.join('_', v.split('_')[0:2])}/images/{v}"
        sub_mappings_2.append(sub_map)

# Get mapping from reference images in joined dataset for nriw training to
# source references generated by matlab from artwin panoramas
mapping = {}
for k, v in sub_mapping_1.items():
    for sm in sub_mappings_2:
        if v in sm:
            mapping[k] = sm[v]


# In[ ]:


str29 = '2019-09-28_08.31.29'
str53 = '2019-09-28_16.11.53'
ply29 = f'/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/{str29}/{str29}_simplified.ply'
ply53 = f'/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/{str53}/{str53}_simplified.ply'
f = 960/np.tan(30*np.pi/180)
u0 = 1920/2
v0 = 1080/2
k = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])
img_size =  np.array([1920, 1080])


# In[ ]:


def o3d_to_pyrenderer(mesh_or_pt):
    if isinstance(mesh_or_pt, o3d.geometry.PointCloud):
        points = np.asarray(mesh_or_pt.points).copy()
        colors = np.asarray(mesh_or_pt.colors).copy()
        mesh = pyrender.Mesh.from_points(points, colors)
    elif isinstance(mesh_or_pt, o3d.geometry.TriangleMesh):
        mesh = trimesh.Trimesh(
            np.asarray(mesh_or_pt.vertices),
            np.asarray(mesh_or_pt.triangles),
            vertex_colors=np.asarray(mesh_or_pt.vertex_colors),
        )
        mesh = pyrender.Mesh.from_trimesh(mesh)
    else:
        raise NotImplementedError()
    return mesh

def to_o3d(rgb, xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.copy().astype(np.float64).T)
    pcd.colors = o3d.utility.Vector3dVector(rgb.copy().astype(np.float64).T)
    return pcd


# In[ ]:


pos = sio.loadmat('/home/kremeto1/inloc/datasets/artwin-small-set/densePE_top100_shortlist.mat')
query_images = pos['ImgList'][0][0]
sorted_db_images = pos['ImgList'][0][0][1]
sorted_db_scores = pos['ImgList'][0][0][2]
sorted_matrices = pos['ImgList'][0][0][3]

# Loading the mesh / pointcloud
# mesh = load_ply(ply_path, voxel_size)
pcds = {}; rgbs = {}; scenes = {}; xyzs = {}

for str_ in [str29, str53]:
    pcd = o3d.io.read_point_cloud(
        f'/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/{str_}/{str_}_simplified.ply',
        remove_nan_points=True, remove_infinite_points=True, print_progress=True
    )
    rgb = np.asarray(pcd.colors).T
    rgb *= 255
    rgb = rgb.astype(np.uint8)
    xyz = np.asarray(pcd.points).T
    pcds[str_] = pcd
    rgbs[str_] = rgb
    xyzs[str_] = xyz
    # For artwin all images have the same dimensions, pre-creating EGL context
    # within renderer speeds up the rendering loop.
    scenes[str_] = pyrender.Scene(bg_color=[0.0,0.0,0.0])
    scenes[str_].add(o3d_to_pyrenderer(pcd))

utils_math = UtilsMath()


# In[ ]:


print(query_images[0][0])
mapping[query_images[0][0]]


# In[ ]:


# o3d.visualization.draw_geometries([pcds[str29]])


# In[ ]:


rndr = pyrender.OffscreenRenderer(1920, 1080, point_size=15)
flags = RenderFlags.FLAT | RenderFlags.RGBA | RenderFlags.DEPTH_ONLY


# In[ ]:


def invert_T(T):
    """Invert a 4x4 transformation matrix."""
    T_inv = np.eye(4)
    r, t = T[:3, :3], T[:3, -1]
    T_inv[:3, :3] = r.T
    T_inv[:3, -1] = -r.T @ t
    return T_inv

ROT_ALIGN_QUERY = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
], dtype=np.float32)


# In[ ]:


#%env DISPLAY=localhost:10.0


# In[ ]:


def a2h(pts):
    """Affine to homogeneous coordinates, (3, N) -> (4, N) shape."""
    return np.vstack((pts, np.ones((1, pts.shape[1]))))

def h2a(pts):
    """Homogeneous to affine coordinates, (4, N) -> (3, N) shape."""
    return (pts / pts[3,:])[:3,:]

def clip_view_frustrum(xyz, rgb, affine_transform):
    """Clip points and colors to view frustrum (without znear and zfar clipping)."""
    pairs = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    pts_cam = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]).T
    at = affine_transform.copy()
    pts_world = h2a(at @ a2h(pts_cam))
    t = affine_transform[:3, -1:]
    
    at[:, 1:3] *= -1
    
    pcd = o3d.geometry.PointCloud()
    pts = np.zeros((5 + 4 + 1000, 3), dtype=np.float64)
    pts[:4, :] = pts_world.astype(np.float64).T
    pts[4:8, :] = h2a(at @ a2h(pts_cam)).astype(np.float64).T
    pts[8, :] = t.squeeze().astype(np.float64).T
    pts[9:, :] = xyz[:,:1000].copy().astype(np.float64).T
    pcd.points = o3d.utility.Vector3dVector(pts)
    colors = np.array([[1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [0,1,0]] + list(repeat([0,0,1], 1000)), dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(np.asarray(pcd.points).shape)
    print(np.asarray(pcd.points).dtype)
    print(np.asarray(pcd.points).data.contiguous)
    # print(np.asarray(pcd.colors).shape)
    # print(np.asarray(pcd.colors).dtype)
    # print(np.asarray(pcd.colors).data.contiguous)
    
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("sync.ply", pcd)
    
    # cloud = PyntCloud(pd.DataFrame(
    # # same arguments that you are passing to visualize_pcl
    # data=np.hstack((pts, colors)),
    # columns=["x", "y", "z", "red", "green", "blue"]))

    # cloud.to_file("output.ply")

    Ndpts = xyz.shape[1]
    vpts = xyz - t
    D = np.ones(Ndpts, dtype=bool)
    t = t.squeeze()
    for j in range(0, 4):
        v2 = pts_world[:, pairs[0, j]] - t
        v1 = pts_world[:, pairs[1, j]] - t
        D = D & (((v2[0] * v1[1]) * vpts[2, :] - (v2[2] * v1[1]) * vpts[0, :] +
                (v2[2] * v1[0]) * vpts[1, :] - (v2[0] * v1[2]) * vpts[1, :] +
                (v2[1] * v1[2]) * vpts[0, :] - (v2[1] * v1[0]) * vpts[2, :]) > 0)
    return xyz[:, D], rgb[:, D]

def render_colmap_image(data, max_radius=5):
    """Render image so that it agrees with artwin reference images."""
    K   = data["K"]
    R   = data["R"]
    C   = data["C"]
    h   = data["h"]
    w   = data["w"]
    xyz = data["xyz"]
    rgb = data["rgb"]

    # project points
    camera2world = np.eye(4)
    camera2world[:3, :3] = R.T
    camera2world[:3, -1:] = C
    camera2world[:, 1:3] *= -1
    
    xyz, rgb = clip_view_frustrum(xyz.copy(), rgb.copy(), camera2world)

    xyz_camera_homogeneous = np.linalg.inv(camera2world) @ a2h(xyz)
    xyz_camera = h2a(xyz_camera_homogeneous)
    depth = np.linalg.norm(xyz_camera, axis=0)

    # focal_length = (float(K[0, 0]) + float(K[1, 1])) / 2
    # df = depth > focal_length
    # print(max(depth))
    # print(min(depth))
    # depth = depth[df]
    # xyz_camera = xyz_camera[:, df]

    uv_homogeneous = K @ xyz_camera
    uv = (uv_homogeneous / uv_homogeneous[2, :])[:2, :]
    u = uv[0, :]
    v = uv[1, :]

    # get data in image
    filtering = np.array((u >= 0) & (v >= 0) & (u < w) & (v < h) & (uv_homogeneous[2, :] < 0)).squeeze()
    depth_filtered = depth[filtering.squeeze()]
    # sort according to depth
    ordering = np.flip(np.argsort(depth_filtered))

    depth = depth_filtered[ordering.squeeze()]
    uv = uv[:, filtering][:, ordering]
    rgb = rgb[:, filtering][:, ordering]

    # Linear interpolation between points (min(depth), min_radius) and (max(depth), max_radius),
    # coefficients computation
    min_radius = 5
    k_ = (max_radius - min_radius) / (max(depth) - min(depth))
    q_ = min_radius - k_ * min(depth)
    # Linear interpolation itself + ensuring only odd radii are present (C++ impl requirement)
    radii = np.round((k_ * depth + q_ + 1) / 2).astype(np.uint16)

    # run cpp to render visibility information
    uv[0,:] = w - uv[0,:]  # Flip so that it agrees with reference images from artwin
    img = renderDepth.render_image(h, w, uv, radii, rgb)

    return img


# In[ ]:


for idx in range(0,1):
    T_query = np.concatenate([sorted_matrices[0][idx], np.array([[0, 0, 0, 1]])], axis=0)
    T_query_inv = invert_T(T_query.copy())

    # camera_pose = T_query_inv.copy()
    # r = camera_pose[:3, :3].copy()  # sorted_matrices[0][idx][:3, :3]
    # camera_pose[:, 1:3] *= -1
    # t = camera_pose[:3, -1:]  # sorted_matrices[0][idx][:3, 3].reshape((3, 1))
    # r = np.linalg.inv(k) @ r
    # t = np.linalg.inv(k) @ t

    dbpath = sorted_db_images[0][idx][0]
    if str29 in dbpath:
        rgb = rgbs[str29]
        xyz = xyzs[str29]
        scene = scenes[str29]
    else:
        rgb = rgbs[str53]
        xyz = xyzs[str53]
        scene = scenes[str53]

    camera = pyrender.IntrinsicsCamera(
        k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    )
    camera_pose = T_query_inv.copy()
    # camera_pose[:3, :3] = r
    # camera_pose[:3, -1:] = -r @ t
    camera_pose[:, 1:3] *= -1
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)

    # Offscreen rendering
    K = k
    R = r
    C = t
    h = 1080
    w = 1920
    rgb = rgb
    xyz = xyz
    max_radius=18
    
    # project points
    camera2world = np.eye(4)
    camera2world[:3, :3] = R.T
    camera2world[:3, -1:] = C
    camera2world[:, 1:3] *= -1
    
    xyz, rgb = clip_view_frustrum(xyz.copy(), rgb.copy(), T_query_inv)
    o3d.io.write_point_cloud("sync.ply", to_o3d(rgb, xyz))
    # o3d.visualization.draw_geometries([to_o3d(rgb, xyz)])

    xyz_camera_homogeneous = np.linalg.inv(camera2world) @ a2h(xyz)
    xyz_camera = h2a(xyz_camera_homogeneous)
    depth = np.linalg.norm(xyz_camera, axis=0)

    # focal_length = (float(K[0, 0]) + float(K[1, 1])) / 2
    # df = depth > focal_length
    # print(max(depth))
    # print(min(depth))
    # depth = depth[df]
    # xyz_camera = xyz_camera[:, df]

    uv_homogeneous = K @ xyz_camera
    uv = (uv_homogeneous / uv_homogeneous[2, :])[:2, :]
    u = uv[0, :]
    v = uv[1, :]

    # get data in image
    filtering = np.array((u >= 0) & (v >= 0) & (u < w) & (v < h) & (uv_homogeneous[2, :] < 0)).squeeze()
    depth_filtered = depth[filtering.squeeze()]
    # sort according to depth
    ordering = np.flip(np.argsort(depth_filtered))

    depth = depth_filtered[ordering.squeeze()]
    uv = uv[:, filtering][:, ordering]
    rgb = rgb[:, filtering][:, ordering]

    # Linear interpolation between points (min(depth), min_radius) and (max(depth), max_radius),
    # coefficients computation
    min_radius = 5
    k_ = (max_radius - min_radius) / (max(depth) - min(depth))
    q_ = min_radius - k_ * min(depth)
    # Linear interpolation itself + ensuring only odd radii are present (C++ impl requirement)
    radii = np.round((k_ * depth + q_ + 1) / 2).astype(np.uint16)

    # run cpp to render visibility information
    uv[0,:] = w - uv[0,:]  # Flip so that it agrees with reference images from artwin
    rgb_rendering = renderDepth.render_image(h, w, uv, radii, rgb)

    depth_rendering = rndr.render(scene, flags=flags)
    scene.remove_node(camera_node)
    print(f"Processed {dbpath}")


# In[ ]:


Image.fromarray(rgb_rendering)


# In[ ]:


Image.open('/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/images/2019-09-28_08.31.29_00004_x30_z240_reference.png')
# query = '/nfs/projects/artwin/experiments/hololens_mapper/joined_dataset/test/0000_reference.png'


# In[ ]:


def get_central_crop(img, crop_height=512, crop_width=512):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    assert len(img.shape) == 3, (
        "input image should be either a 2D or 3D matrix,"
        " but input was of shape %s" % str(img.shape)
    )
    height, width, ch = img.shape
    # assert height >= crop_height and width >= crop_width, (
    #     "input image cannot " "be smaller than the requested crop size"
    # )
    if height >= crop_height and width >= crop_width:
        st_y = (height - crop_height) // 2
        st_x = (width - crop_width) // 2
        return np.squeeze(img[st_y : st_y + crop_height, st_x : st_x + crop_width, :])
    else:
        new_img = np.zeros((crop_height, crop_height, ch), dtype=np.float32)
        new_img[st_y : st_y + height, st_x : st_x + width, :] = img
        return np.squeeze(new_img)


# In[ ]:


plt.imshow(np.array(Image.open('/home/kremeto1/neural_rendering/datasets/raw/inloc/query/iphone7/IMG_0938.JPG')).astype(np.float32)/255)


# In[ ]:


rendered_img = np.array(Image.open('/home/kremeto1/neural_rendering/datasets/raw/inloc/query/iphone7/IMG_0938.JPG')).astype(np.float32)
print(rendered_img.shape)
rendered_img = get_central_crop(rendered_img, 1600, 1600)
print(rendered_img.shape)
plt.imshow(rendered_img/255)


# In[ ]:


def load_ply(ply_path, voxel_size):
    # Loading the mesh / pointcloud
    m = trimesh.load(ply_path)
    if isinstance(m, trimesh.PointCloud):
        if voxel_size is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(m.vertices))
            pcd.colors = o3d.utility.Vector3dVector(
                np.asarray(m.colors, dtype=np.float64)[:, :3] / 255
            )
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            mesh = o3d_to_pyrenderer(pcd)
        else:
            points = m.vertices.copy()
            colors = m.colors.copy()
            mesh = pyrender.Mesh.from_points(points, colors)
    elif isinstance(m, trimesh.Trimesh):
        if voxel_size is not None:
            m2 = m.as_open3d
            m2.vertex_colors = o3d.utility.Vector3dVector(
                np.asarray(m.visual.vertex_colors, dtype=np.float64)[:, :3] / 255
            )
            m2 = m2.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average,
            )
            mesh = o3d_to_pyrenderer(m2)
        else:
            mesh = pyrender.Mesh.from_trimesh(m)
    else:
        raise NotImplementedError(
            "Unsupported 3D object. Supported format is a `.ply` pointcloud or mesh."
        )
    return mesh


# In[ ]:


# /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/images/2019-09-28_08.31.29_00002_x0_z60_params.json
data = {
    "x_rot_mat": [[1,0,0],[0,6.123233995736766E-17,-1],[0,1,6.123233995736766E-17]],
    "z_rot_mat": [[0.50000000000000011,-0.8660254037844386,0],[0.8660254037844386,0.50000000000000011,0],[0,0,1]],
    "pano_rot_mat": [[-0.99218108866967136,-0.1246819992809647,-0.0055755126729838166],[0.12466066620694956,-0.99219127447899413,0.0040240711460282075],[-0.0060337042606172378,0.0032975601662910271,0.99997635997549683]],
    "pano_translation": [-0.237677,-5.618579,1.862724],
    "calibration_mat": [[1662.7687752661222,0,960],[0,1662.7687752661222,540],[0,0,1]],
    "source_pano_path": "/nfs/projects/artwin/SIE factory datasets/proc/2019-09-28_08.31.29/pano/00002-pano.jpg","source_ply_path":"/nfs/projects/artwin/SIE factory datasets/proc/2019-09-28_08.31.29/2019-09-28_08.31.29_txt.ply"
}


# In[ ]:


pcd_rot = np.array(data["x_rot_mat"]) @ np.array(data["z_rot_mat"]) @ np.array(data["pano_rot_mat"]).T
pcd_t = np.array(data["pano_translation"])
k = np.array(data["calibration_mat"])
f = 960/np.tan(30*np.pi/180)
t = -pcd_rot.T @pcd_t


# In[ ]:


mesh = load_ply('/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/2019-09-28_08.31.29.ply', None)


# In[ ]:


flags = RenderFlags.FLAT | RenderFlags.RGBA
scene = pyrender.Scene(bg_color=[0,0,0])
scene.add(mesh)
camera = pyrender.camera.IntrinsicsCamera(
    k[0, 0], k[1, 1], k[0, 2], k[1, 2]
)
camera_pose = np.eye(4)
camera_pose[:3, :3] = pcd_rot.T
camera_pose[:3, -1:] = pcd_t.reshape((3,1))
camera_pose[:, 1:3] *= -1
scene.add(camera, pose=camera_pose)


# In[ ]:


# Offscreen rendering
r = pyrender.OffscreenRenderer(1920, 1080,point_size=5)
rgb_rendering, depth_rendering = r.render(scene, flags=flags)
r.delete()


# In[ ]:


Image.fromarray(rgb_rendering)


# In[ ]:


x = projectMesh.projectMeshDebug(
    '/nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/2019-09-28_08.31.29_mesh.ply',
    f,
    pcd_rot,
    t,
    np.array([1920, 1080]),
    False, None, False)


# In[ ]:


np.unique(x[1].sum(axis=2))


# In[ ]:


def getCentered3Dindices(mode, sensorWidth, sensorHeight):
    if mode == 'x':
        length = sensorWidth
    elif mode == 'y':
        length = sensorHeight
    else:
        raise ValueError('Unknown mode!')

    halfFloat = length/2
    halfInt = int(halfFloat)
    lower = -halfInt
    upper = halfInt
    if not np.allclose(halfFloat, halfInt):
        upper += 1
    ind = np.arange(lower, upper)
    if mode == 'x':
        ind = np.broadcast_to(ind, (sensorHeight, sensorWidth)).T
    elif mode == 'y':
        ind = np.broadcast_to(ind, (sensorWidth, sensorHeight))
    ind = np.reshape(ind, (sensorHeight*sensorWidth, 1))
    ind = np.tile(ind, (1,3))
    return ind
    
def buildXYZcut(sensorWidth, sensorHeight, t, cameraDirection, scaling, sensorXAxis, sensorYAxis, depth):
    # TODO: compute xs, ys only once (may not actually matter)
    ts = np.broadcast_to(t, (sensorHeight*sensorWidth, 3))
    camDirs = np.broadcast_to(cameraDirection, (sensorHeight*sensorWidth, 3))
    xs = getCentered3Dindices('x', sensorWidth, sensorHeight)
    ys = getCentered3Dindices('y', sensorWidth, sensorHeight)
    sensorXAxes = np.broadcast_to(sensorXAxis, (sensorHeight*sensorWidth, 3))
    sensorYAxes = np.broadcast_to(sensorYAxis, (sensorHeight*sensorWidth, 3))
    sensorDirs = camDirs + scaling * np.multiply(xs, sensorXAxes) + scaling * np.multiply(ys, sensorYAxes)
    depths = np.reshape(depth.T, (sensorHeight*sensorWidth, 1))
    depths = np.tile(depths, (1,3))
    pts = ts + np.multiply(sensorDirs, depths)
    xyzCut = np.reshape(pts, (sensorHeight, sensorWidth, 3))
    return xyzCut, pts

#     pixel_centers_in_screen_space = np.transpose(np.mgrid[-hh:hh, -hw:hw], (1, 2, 0))
#     points_in_view_space = np.append(pixel_centers_in_screen_space / focal_length, np.ones(depth.shape + (1,)), axis=2) * np.repeat(depth.reshape(depth.shape + (1,)), 3, axis=2)
#     points_in_view_space = points_in_view_space.reshape((-1, 3))
#     points_in_world_space = (camera_pose @ np.concatenate([points_in_view_space, np.ones((points_in_view_space.shape[0], 1))], axis=1).T).T
#     points_in_world_space = points_in_world_space[:, :3]
#     points_in_world_space = points_in_world_space.reshape(pixel_centers_in_screen_space.shape[:2] + (3,))
#     points_in_world_space.shape

def project_mesh_core(depth, k, R, t, debug):
    # camera_pose_rendering_convention = np.eye(4)
    # camera_pose_rendering_convention[:3, :3] = R.T
    # camera_pose_rendering_convention[:3, -1:] = -R.T @ t
    # camera_pose_rendering_convention[:, 1:3] *= -1  # This is change of convention
    
    
    
    sensor_width = depth.shape[1]
    sensor_height = depth.shape[0]
    assert k[0, 0] == k[1, 1], "Camera pixel is not square."
    focal_length = k[0, 0]
    scaling = 1.0 / focal_length
    space_coord_system = np.eye(3)
    sensor_coord_system = np.matmul(R, space_coord_system)
    sensor_x_axis = sensor_coord_system[:, 0]
    sensor_y_axis = sensor_coord_system[:, 1]
    # make camera point toward -z by default, as in OpenGL
    camera_dir = sensor_coord_system[:, 2] # unit vector

    xyz_cut, pts = buildXYZcut(
        sensor_width, sensor_height,
        t, camera_dir, scaling,
        sensor_x_axis, sensor_y_axis, depth
    )

    xyz_pc = -1
    if debug:
        xyz_pc = o3d.geometry.PointCloud()
        xyz_pc.points = o3d.utility.Vector3dVector(pts)

    return xyz_cut, pts

def squarify(image: np.array, square_size: int) -> np.array:
    assert square_size >= image.shape[0] and square_size >= image.shape[1]
    shape = list(image.shape)
    shape[0] = shape[1] = square_size
    square = np.zeros(shape, dtype=image.dtype)
    h, w = image.shape[:2]
    offset_h = (square_size - h) // 2
    offset_w = (square_size - w) // 2
    square[offset_h : offset_h + h, offset_w : offset_w + w, ...] = image
    return square


# In[ ]:


sys.path.append('/home/kremeto1/inloc/inLocCIIRC_dataset/buildCutouts')
import read_model

def get_colmap_file(colmap_path, file_stem):
    colmap_path = Path(colmap_path)
    fp = colmap_path / f"{file_stem}.bin"
    if not fp.exists():
        fp = colmap_path / f"{file_stem}.txt"
    return str(fp)

# Load camera matrices and names of corresponding src images from
# colmap images.bin and cameras.bin files from colmap sparse reconstruction
def load_cameras_colmap(images_fp, cameras_fp):
    if images_fp.endswith(".bin"):
        images = read_model.read_images_binary(images_fp)
    else:  # .txt
        images = read_model.read_images_text(images_fp)

    if cameras_fp.endswith(".bin"):
        cameras = read_model.read_cameras_binary(cameras_fp)
    else:  # .txt
        cameras = read_model.read_cameras_text(cameras_fp)

    src_img_nms = []
    K = []
    T = []
    R = []
    w = []
    h = []

    for i in images.keys():
        R.append(read_model.qvec2rotmat(images[i].qvec))
        T.append((images[i].tvec)[..., None])
        k = np.eye(3)
        camera = cameras[images[i].camera_id]
        if camera.model in ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]:
            k[0, 0] = cameras[images[i].camera_id].params[0]
            k[1, 1] = cameras[images[i].camera_id].params[0]
            k[0, 2] = cameras[images[i].camera_id].params[1]
            k[1, 2] = cameras[images[i].camera_id].params[2]
        elif camera.model in ["RADIAL", "PINHOLE"]:
            k[0, 0] = cameras[images[i].camera_id].params[0]
            k[1, 1] = cameras[images[i].camera_id].params[1]
            k[0, 2] = cameras[images[i].camera_id].params[2]
            k[1, 2] = cameras[images[i].camera_id].params[3]
        # TODO : Take other camera models into account + factorize
        else:
            raise NotImplementedError("Camera models not supported yet!")

        K.append(k)
        w.append(cameras[images[i].camera_id].width)
        h.append(cameras[images[i].camera_id].height)
        src_img_nms.append(images[i].name)

    return K, R, T, h, w, src_img_nms


# In[ ]:


import argparse
args = {
  "input_root": Path("/home/kremeto1/neural_rendering/datasets/post_processed/imc/hagia_sophia_interior_thesis_test-100_src-fused"),
  "input_ply_path": Path("/home/kremeto1/neural_rendering/datasets/processed/imc/hagia_sophia_interior/dense/dense/0/fused.ply"),
  "input_root_colmap": Path("/home/kremeto1/neural_rendering/datasets/processed/imc/hagia_sophia_interior/dense/dense/0/sparse"),
  "output_root": Path("/home/kremeto1/neural_rendering/datasets/final/imc/hagia_sophia_interior_thesis_test_squarify-100_src-fused_inloc_format"),
  "test_size": 50,
  "squarify": True
}
# args = {
#   "input_root": Path("/nfs/projects/artwin/experiments/thesis/artwin_as_inloc/2019-09-28_08.31.29-splatting"),
#   "input_ply_path": Path("/nfs/projects/artwin/experiments/thesis/artwin_as_inloc/2019-09-28_08.31.29/2019-09-28_08.31.29.ply"),
#   "input_root_colmap": Path("/nfs/projects/artwin/experiments/thesis/artwin_as_inloc/2019-09-28_08.31.29/sparse"),
#   "output_root": Path("/nfs/projects/artwin/experiments/thesis/artwin_as_inloc/2019-09-28_08.31.29-splatting-inloc_format"),
#   "test_size": "0",
#   "val_ratio": "1.0",
#   "squarify": "False"
# }
args = argparse.Namespace(**args)


# In[ ]:


Ks, Rs, Ts, Hs, Ws, img_nms = load_cameras_colmap(
    get_colmap_file(args.input_root_colmap, "images"),
    get_colmap_file(args.input_root_colmap, "cameras")
)


# In[ ]:


import random

indices = list(range(len(Hs)))

test_idx = 584
test_idx = 50
photo_path = args.input_root / "train" / "{:04n}_reference.png".format(test_idx)
if not photo_path.exists():
    photo_path = args.input_root / "test" / "{:04n}_reference.png".format(test_idx)
rnd = random.Random(42)
rnd.shuffle(indices)
print(indices[test_idx])

calibration_mat = Ks[indices[test_idx]]
rotation_mat = Rs[indices[test_idx]]
translation = Ts[indices[test_idx]]
output_root = args.output_root
square = args.squarify

stem = photo_path.stem.strip("_reference")

depth_npy = photo_path.parent / (stem + "_depth.npy")
if not depth_npy.exists():
    depth_npy = photo_path.parent / (stem + "_depth.png.npy")
mesh_projection = photo_path.parent / (stem + "_color.png")
cutout_reference = photo_path


# In[ ]:


k = calibration_mat
hh = k[1, 2]
hw = k[0, 2]
assert k[0, 0] == k[1, 1], "Camera pixel is not square."
focal_length = k[0, 0]

dataset_depth = np.load(str(depth_npy))
# dataset_depth = dataset_depth * 100 / 255
# dataset_depth = np.divide(dataset_depth, np.sqrt(np.square(np.transpose(np.mgrid[-hh:hh, -hw:hw], (1, 2, 0)) / focal_length).sum(axis=2) + 1))


# In[ ]:


import cv2
prj = plt.imread(str(mesh_projection), cv2.IMREAD_UNCHANGED)[:, :, :3]
ref = plt.imread(str(cutout_reference), cv2.IMREAD_UNCHANGED)[:, :, :3]
# dataset_depth = np.load(str(depth_npy))
# XYZcut, pts = project_mesh_core(dataset_depth, calibration_mat, rotation_mat.T, (- rotation_mat.T @ translation).squeeze(), False)
XYZcut, pts = project_mesh_core(dataset_depth, calibration_mat, rotation_mat.T, (- rotation_mat.T @ translation).squeeze(), False)
# if square:
#     XYZcut = squarify(XYZcut, ref.shape[0])


# In[ ]:


ts = np.broadcast_to(t, (sensorHeight*sensorWidth, 3))
camDirs = np.broadcast_to(cameraDirection, (sensorHeight*sensorWidth, 3))
xs = getCentered3Dindices('x', sensorWidth, sensorHeight)
ys = getCentered3Dindices('y', sensorWidth, sensorHeight)
sensorXAxes = np.broadcast_to(sensorXAxis, (sensorHeight*sensorWidth, 3))
sensorYAxes = np.broadcast_to(sensorYAxis, (sensorHeight*sensorWidth, 3))
sensorDirs = camDirs + scaling * np.multiply(xs, sensorXAxes) + scaling * np.multiply(ys, sensorYAxes)
depths = np.reshape(depth.T, (sensorHeight*sensorWidth, 1))
depths = np.tile(depths, (1,3))
pts = ts + np.multiply(sensorDirs, depths)
xyzCut = np.reshape(pts, (sensorHeight, sensorWidth, 3))


# # THIS IS FUCKING IMPORTANT!

# In[ ]:


pixel_centers = np.append(np.transpose(np.mgrid[-hh:hh, -hw:hw], (1, 2, 0)), focal_length*np.ones((766, 1025) + (1,)), axis=2)
# pixel_centers*=-1
# mm = np.linalg.inv(calibration_mat)
# rays = np.matmul(np.tile(mm, (766, 1025, 1, 1)), pixel_centers[:, :, :, np.newaxis]).squeeze()
rays = pixel_centers / np.linalg.norm(pixel_centers, axis=2)[:,:,np.newaxis]

cam_pose = np.eye(4)
cam_pose[:3, :3] = rotation_mat.T
cam_pose[:3, -1:] = (- rotation_mat.T @ translation)
cam_pose[:3, :3] = cam_pose[:3, :3] @ R.from_euler('z', 90, degrees=True).as_matrix()
# cam_pose[:, 1:3] *= -1

points = np.multiply(rays, np.repeat(dataset_depth[:, ::-1, np.newaxis], 3, axis=2))
points = np.append(points, np.ones((766, 1025) + (1,)), axis=2)
points = np.matmul(np.tile(cam_pose, (766, 1025, 1, 1)), points[:, :, :, np.newaxis]).squeeze()[:,:,:3]


# In[ ]:


sensor_width = dataset_depth.shape[1]
sensor_height = dataset_depth.shape[0]
getCentered3Dindices('x', sensor_width, sensor_height).shape


# In[ ]:


np.transpose(np.mgrid[-hw:hw, -hh:hh], (2, 1, 0))[0,:,0]


# In[ ]:


def compute_xyz_cut(k, R, t, depth):
    assert k[0, 0] == k[1, 1]
    focal_length = k[0, 0]
    hh = k[1, 2]  # half height
    hw = k[0, 2]  # half width
    shape = (int(2 * hh), int(2 * hw))

    pixel_centers = np.append(np.transpose(np.mgrid[-hw:hw, -hh:hh], (2, 1, 0)) / focal_length, np.ones(shape + (1,)), axis=2)
    points = np.matmul(np.tile(R, shape + (1, 1)), pixel_centers[:, :, :, np.newaxis]).squeeze()[:, :, :3]
    points = np.multiply(points, np.repeat(depth[:, :, np.newaxis], 3, axis=2))
    points = points + t.reshape((1,1,3))
    return points


# In[ ]:


points = compute_xyz_cut(calibration_mat, rotation_mat.T, (- rotation_mat.T @ translation), dataset_depth)


# In[ ]:


x = np.load('/home/kremeto1/neural_rendering/datasets/post_processed/imc/pantheon_exterior_thesis_test-100_src-fused-splatting/test/0000_depth.png.npy')

# plt.hist(x)


# In[ ]:


from scipy.spatial.transform import Rotation as R
points = np.matmul(np.tile(R.from_euler('z', 90, degrees=True).as_matrix(), (766, 1025, 1, 1)), points[:, :, :, np.newaxis]).squeeze()[:,:,:3]


# # XYZ Cut vizualization

# In[ ]:


pose['calibration_mat'][1,2]


# In[ ]:


import numpy as np
import open3d as o3d
import scipy.io as sio
import cv2

def create_coordinate_frame(R, t, length=1):
    # Extract rotation matrix and translation vector from the camera matrix
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = t

    # Define the coordinate frame origin
    origin = np.zeros(3)

    # Create the coordinate frame using create_mesh_coordinate_frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=length, origin=origin)
    
    # Transform the coordinate frame to the desired pose
    coordinate_frame.transform(matrix)

    return coordinate_frame

def create_view_frustum_mesh(K, R, t):
    """
    Create a view frustum mesh based on camera intrinsic parameters.

    Args:
        intrinsics (open3d.camera.PinholeCameraIntrinsic): Camera intrinsic parameters.
        depth_range (tuple): Range of depth values to define the near and far clipping planes.

    Returns:
        open3d.geometry.TriangleMesh: The view frustum mesh.
    """

    # Define the corners of the view frustum in camera space

    width, height, fx, fy, cx, cy = int(K[0,2]*2), int(K[1,2]*2), K[0,0], K[1,1], K[0,2], K[1,2]

    def v(x, y, widht, height, f):
        return [
            10 * (x * 0.5 * width) / f,
            10 * (y * 0.5 * height) / f,
            10
        ]

    corners_camera = np.array([
        [0, 0, 0],
        v(-1, -1, width, height, fx),
        v(1, -1, width, height, fx),
        v(1, 1, width, height, fx),
        v(-1, 1, width, height, fx),
    ])

    # Transform the frustum corners to world coordinates
    corners_world = (R @ corners_camera.T + t.T).T

    # Define the frustum triangles (connecting the corners)
    triangles = [
        [0, 2, 1],
        [0, 3, 2],
        [0, 4, 3],
        [0, 1, 4]
    ]

    colors = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]])

    # Create the view frustum mesh
    view_frustum_mesh = o3d.geometry.TriangleMesh()
    view_frustum_mesh.vertices = o3d.utility.Vector3dVector(corners_world)
    view_frustum_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    view_frustum_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return view_frustum_mesh

matfile_path = '/home/kremeto1/neural_rendering/datasets/final/imc/hagia_sophia_interior_thesis_spheres_squarify_corrected_missing_points-100_src-fused_inloc_format2/matfiles/cutout_0438.png.mat'
# matfile_path = '/home/kremeto1/neural_rendering/datasets/final/imc/pantheon_exterior_thesis_spheres_squarify-100_src-fused_inloc_format/matfiles/cutout_0081.png.mat'
# matfile_path = '/home/kremeto1/neural_rendering/datasets/final/imc/grand_place_brussels_thesis_spheres_squarify-100_src-fused_inloc_format/matfiles/cutout_0076.png.mat'
# matfile_path = '/nfs/projects/artwin/experiments/thesis/artwin_as_inloc/joined-dataset-spheres-inloc_format/matfiles/cutout_0088.png.mat'
# matfile_path = '/home/kremeto1/neural_rendering/datasets/final/inloc/inloc_rendered_splatting-inloc_format/matfiles/cutout_DUC_cutout_037_150_0.png.mat'
# matfile_path = '/home/kremeto1/neural_rendering/datasets/raw/inloc/database/cutouts/DUC2/037/DUC_cutout_037_150_0.jpg.mat'
pose_path = matfile_path.replace('matfiles', 'poses')

pose = sio.loadmat(pose_path)
frame = create_coordinate_frame(pose['R'], pose['position'])

# intrinsics = o3d.camera.PinholeCameraIntrinsic(int(pose['calibration_mat'][0,2]*2), int(pose['calibration_mat'][1,2]*2), pose['calibration_mat'][0,0], pose['calibration_mat'][1,1], pose['calibration_mat'][0,2],pose['calibration_mat'][1,2])

# Create the camera view frustum
view_frustum = create_view_frustum_mesh(pose['calibration_mat'], pose['R'], pose['position'])


matfile = sio.loadmat(matfile_path)
print('__header__: ', matfile['__header__'])
print('__version__: ', matfile['__version__'])
print('__globals__: ', matfile['__globals__'])
print('Remaining keys: ', list(matfile.keys())[-2:])
points = matfile["XYZcut"]

row_linspace = np.linspace(0, 255, points.shape[1], dtype=np.uint8)
tiled_per_columns = np.tile(row_linspace, (points.shape[0], 1))
colored_columns = cv2.applyColorMap(tiled_per_columns, cv2.COLORMAP_JET) / 255

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.reshape((-1,3)))
pcd.colors = o3d.utility.Vector3dVector(colored_columns.reshape((-1,3)))

print('Saving pointcloud to PLY file')
o3d.io.write_point_cloud('test.ply', pcd)
print('Saving frame to PLY file')
o3d.io.write_triangle_mesh('test-frame.ply', frame)
print('Saving frustum to PLY file')
o3d.io.write_triangle_mesh('test-frustum.ply', view_frustum)

# Create a MeshLab server instance
# ms = ml.MeshSet()

# # Import the PointCloud
# ms.load_new_mesh("point_cloud", pcd)

# # Import the TriangleMesh
# ms.load_new_mesh("coord_frame", frame)

# # Merge the meshes
# ms.current_mesh().merge(ms.named_mesh("coord_frame"))

# # Export the merged mesh as a PLY file
# ms.save_current_mesh("test.ply")

# # Clean up the MeshLab server
# ms.clear()

# mmesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
# mm = np.eye(4)
# mm[:3,3] = (- rotation_mat.T @ translation).squeeze()
# mmesh.transform(mm)

# o3d.io.write_triangle_mesh('test.ply', mmesh)

# from open3d import JVisualizer
# visualizer = JVisualizer()
# visualizer.add_geometry(pcd)
# visualizer.show()


# In[ ]:


depth = np.load('/home/kremeto1/neural_rendering/datasets/final/inloc/inloc_rendered_spheres-inloc_format/depthmaps/depth_DUC_cutout_037_150_0.png.npy')
# depth[np.abs(depth - 1) < .0005] = 0.0
np.abs(depth - 1) < .0005
(np.abs(depth - 1) < .5).flatten().sum()


# In[ ]:


points[points.sum(axis=2)<0.005] = np.nan
points


# In[ ]:


from matplotlib import pyplot as plt
plt.hist(np.load('/home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_pyrender/CSE3/000/cse_cutout_000_0_-30_depth.npy'), range=(-1,6))


# In[ ]:


print(np.load('/home/kremeto1/neural_rendering/datasets/post_processed/imc/pantheon_exterior_thesis_test-100_src-fused-splatting/val/0050_depth.png.npy')[0,0])
print(np.load('/home/kremeto1/neural_rendering/datasets/post_processed/imc/pantheon_exterior_thesis_test-100_src-fused-spheres/val/0050_depth.png.npy')[0,0])
print(np.load('/home/kremeto1/neural_rendering/datasets/post_processed/imc/pantheon_exterior_thesis_test_squarify-100_src-fused/val/0050_depth.npy')[0,0])


# In[ ]:


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.reshape((-1,3)))
pcd.colors = o3d.utility.Vector3dVector(
    ((np.clip(cv2.applyColorMap((np.tile(np.arange(points.shape[1]),(points.shape[0],1))*255/points.shape[1]).astype(np.uint8), cv2.COLORMAP_JET),0,255))/255).reshape((-1,3))
)


# In[ ]:


# mm = rotation_mat.T.copy()
# mm[:, 1:3] *= -1
# cc = XYZcut.copy()
# tt = (- rotation_mat.T @ translation)
# cc[:,:,0] -= tt[0,0]
# cc[:,:,1] -= tt[1,0]
# cc[:,:,2] -= tt[2,0]
# cc = np.square(cc)
# cc = np.sum(cc, axis=2)
# cc = np.sqrt(cc)
dd = np.linalg.norm(points - (- rotation_mat.T @ translation).reshape((1,1,3)), axis=2)
# dd=cc


# In[ ]:


x = np.eye(4)
x[:3,:3] = rotation_mat.T
x[:, 1:3] *= -1
print(x.flatten())
print((rotation_mat.T @ translation).squeeze())


# In[ ]:


# k = calibration_mat
# hh = k[1, 2]
# hw = k[0, 2]
# assert k[0, 0] == k[1, 1], "Camera pixel is not square."
# focal_length = k[0, 0]

# camera_pose = np.eye(4)
# camera_pose[:3, :3] = rotation_mat.T
# camera_pose[:3, -1:] = (- rotation_mat.T @ translation)
# camera_pose[:, 1:3] *= -1

# pixel_centers_in_screen_space = np.transpose(np.mgrid[-hh:hh, -hw:hw], (1, 2, 0))
# points_in_view_space = np.append(pixel_centers_in_screen_space / focal_length, np.ones(depth.shape + (1,)), axis=2) * np.repeat(depth.reshape(depth.shape + (1,)), 3, axis=2)
# points_in_view_space = points_in_view_space.reshape((-1, 3))
# points_in_world_space = (camera_pose @ np.concatenate([points_in_view_space, np.ones((points_in_view_space.shape[0], 1))], axis=1).T).T
# points_in_world_space = points_in_world_space[:, :3]
# points_in_world_space = points_in_world_space.reshape(pixel_centers_in_screen_space.shape[:2] + (3,))
# points_in_world_space.shape


# In[ ]:


points = sio.loadmat('/home/kremeto1/neural_rendering/datasets/final/inloc/inloc_rendered_pyrender_inloc_format/matfiles/cutout_cse_cutout_000_0_-30.png.mat')["XYZcut"]
poses = sio.loadmat('/home/kremeto1/neural_rendering/datasets/final/inloc/inloc_rendered_pyrender_inloc_format/poses/cutout_cse_cutout_000_0_-30.png.mat')
prj = plt.imread('/home/kremeto1/neural_rendering/datasets/final/inloc/inloc_rendered_pyrender_inloc_format/cutouts/cutout_cse_cutout_000_0_-30.png', cv2.IMREAD_UNCHANGED)[:, :, :3]
rotation_mat = poses['R']
translation = poses['position'].T
calibration_mat = poses['calibration_mat']


# In[ ]:


k = calibration_mat
hh = k[1, 2]
hw = k[0, 2]
assert k[0, 0] == k[1, 1], "Camera pixel is not square."
focal_length = k[0, 0]

# pts = np.load('/nfs/projects/artwin/experiments/thesis/artwin_as_inloc/joined-dataset-pyrender-inloc_format/matfiles/cutout_0075.png.mat.npy').reshape((-1, 3))
# pts = points_in_world_space.copy().reshape((-1, 3))
# pts[:, 1] *= -1
scene = pyrender.Scene(bg_color=[0,0,0])
mesh = pyrender.Mesh.from_points(points.reshape(-1,3))
# mesh = load_ply('test.ply', None)

scene.add(mesh)
camera = pyrender.camera.IntrinsicsCamera(
    k[0, 0], k[1, 1], k[0, 2], k[1, 2]
)
cam_pose = np.eye(4)
cam_pose[:3, :3] = rotation_mat.T
cam_pose[:3, -1:] = (- rotation_mat.T @ translation)
# cam_pose[:, 1:3] *= -1
# cam_pose[:3, :3] = np.array([[0,-1,0],[1,0,0],[0,0,1]]) @ cam_pose[:3, :3]
# cam_pose[:3, -1:] = translation
# cam_pose[:3, :3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float) @ cam_pose[:3, :3]
scene.add(camera, pose=cam_pose)

# Offscreen rendering
r = pyrender.OffscreenRenderer(hw*2, hh*2, point_size=3)
rgb_rendering, depth_rendering = r.render(scene, flags=pyrender.RenderFlags.FLAT)
r.delete()


# In[ ]:


plt.figure(figsize=(12, 8))
# plt.imshow((np.clip(squarify(cv2.applyColorMap((255*depth_rendering/20).astype(np.uint8), cv2.COLORMAP_HSV), 1248),0,255) * 0.5 + 0.5 * prj).astype(np.uint8))
plt.imshow((np.clip(cv2.applyColorMap((255*depth_rendering/20).astype(np.uint8), cv2.COLORMAP_HSV),0,255) * 0.5 + 0.5 * prj).astype(np.uint8))


# In[ ]:


plt.figure(figsize=(12, 8))
# plt.imshow(np.divide(dd, np.sqrt(np.square(np.transpose(np.mgrid[-hh:hh, -hw:hw], (1, 2, 0)) / focal_length).sum(axis=2) + 1)))
plt.imshow((np.clip(cv2.applyColorMap((np.tile(np.arange(1025),(766,1))*255/1025).astype(np.uint8), cv2.COLORMAP_HSV),0,255)))


# In[ ]:


plt.figure(figsize=(12, 8))
plt.imshow(prj)


# In[ ]:


mat = sio.loadmat('/home/kremeto1/inloc/datasets/hagia-pyrender/densePE_top100_shortlist.mat')
index = 0
query_path = Path(str(mat['ImgList'][0][index][0][0]))
candidate_paths = [Path(str(i[0])) for i in mat['ImgList'][0][index][1][0]]
# candidate_scores = mat['ImgList'][0][index][2][0]
candidate_transforms = [i for i in mat['ImgList'][0][index][3][0]]
candidate_r = [i for i in mat['ImgList'][0][index][4][0]]
candidate_t = [i for i in mat['ImgList'][0][index][5][0]]


# In[ ]:


x = np.load('/home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_spheres/DUC1/088/DUC_cutout_088_210_30_depth.png.npy')
plt.figure(figsize=(12, 8))
plt.imshow(x)


# In[ ]:


len(x.flatten())


# In[ ]:


np.max(x)


# In[ ]:


import scipy.misc
plt.figure(figsize=(12, 8))
plt.imshow(cv2.resize(cv2.rotate(plt.imread('/home/kremeto1/neural_rendering/datasets/raw/inloc/query/iphone7/IMG_0731.JPG'), cv2.ROTATE_90_CLOCKWISE), (1600,1200)))


# In[ ]:


for i in Path('/home/kremeto1/neural_rendering/datasets/raw/inloc/query/iphone7/').glob('*.JPG'):
    plt.imsave(str(i).replace('iphone7', 'same_db_size'), cv2.resize(plt.imread(str(i)), (1600,1200)))


# In[ ]:


for i in Path('/home/kremeto1/neural_rendering/datasets/raw/inloc/query/iphone7/').glob('*.JPG'):
    new_path = str(i).replace('iphone7', 'same_db_size')
    image = plt.imread(str(i))
    if image.shape[:2] == (4032, 3024):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    new_img = cv2.resize(image, (1600, int(3024 * 1600 / 4032)))
    # img = np.zeros((1200, 1600, 3), dtype=image.dtype)
    # h, w = new_img.shape[:2]
    # # offset_h = (1600 - h) // 2
    # offset_w = (1600 - w) // 2
    # img[:, offset_w : offset_w + w, ...] = new_img
    plt.imsave(new_path, new_img)
    print(i.name)


# # Checking depths of datasets

# In[ ]:


import random
import skimage
random.seed(42)


# In[ ]:


def process_dataset(dataset, sample_size=None):
    depth_maps = []
    for part in ["train", "val"]:
        depths = list((Path(dataset) / part).glob("*_depth.png"))
        random.shuffle(depths)
        for depth in depths[:sample_size]:
            # read as uint16
            depth = skimage.io.imread(str(depth))
            depth_maps.append(depth.flatten())
    return np.concatenate(depth_maps)


# In[ ]:


def process_dataset_npy(dataset, sample_size=None):
    depth_maps = []
    depths = list(Path(dataset).glob("**/*.npy"))
    random.shuffle(depths)
    for depth in depths[:sample_size]:
        # read as uint16
        depth = np.load(str(depth))
        depth_maps.append(depth.flatten())
    return np.concatenate(depth_maps)


# In[ ]:


def my_hist(collection, data, labels, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(
        data,
        label=labels,
        log=True,
        **kwargs
    )
    ax.set_ylabel('Counts')
    ax.set_xlabel('Depths')
    ax.legend(loc='upper right')
    ax.set_xscale('log')
    ax.set_title(f'Histogram of depths of {collection}')
    #plt.savefig(f'{collection}_hist.png', dpi=300)
    plt.show()


# In[ ]:


pyrender = process_dataset('/home/kremeto1/neural_rendering/datasets/post_processed/inloc/inloc_rendered_pyrender', 250)
spheres = process_dataset('/home/kremeto1/neural_rendering/datasets/post_processed/inloc/inloc_rendered_spheres', 250)
splatting = process_dataset('/home/kremeto1/neural_rendering/datasets/post_processed/inloc/inloc_rendered_splatting', 250)


# In[ ]:


my_hist(collection="inloc", data=[pyrender]+[spheres]+[splatting], labels=['pyrender', 'spheres', 'splatting'])


# In[ ]:


pyrender_npy = process_dataset_npy('/home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_pyrender', 250)
spheres_npy = process_dataset_npy('/home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_spheres', 250)
splatting_npy = process_dataset_npy('/home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_splatting', 250)


# In[ ]:


my_hist(collection="inloc", data=[pyrender_npy]+[spheres_npy]+[splatting_npy], labels=['pyrender', 'spheres', 'splatting'])


# In[ ]:


with open('/home/kremeto1/neural_rendering/datasets/processed/imc/hagia_sophia_interior/dense/dense/0/fused_25.ply.kdtree.radii') as f:
    radii25 = [float(i) for i in f.readlines()[0].split(' ')[5:]]
with open('/home/kremeto1/neural_rendering/datasets/processed/imc/hagia_sophia_interior/dense/dense/0/fused_50.ply.kdtree.radii') as f:
    radii50 = [float(i) for i in f.readlines()[0].split(' ')[5:]]
with open('/home/kremeto1/neural_rendering/datasets/processed/imc/hagia_sophia_interior/dense/dense/0/fused.ply.kdtree.radii') as f:
    radii = [float(i) for i in f.readlines()[0].split(' ')[5:]]


# In[ ]:


plt.plot(np.linspace(0, 1, 100)[1:-1], np.quantile(radii25, np.linspace(0, 1, 100)[1:-1]))
plt.plot(np.linspace(0, 1, 100)[1:-1], np.quantile(radii50, np.linspace(0, 1, 100)[1:-1]))
plt.plot(np.linspace(0, 1, 100)[1:-1], np.quantile(radii, np.linspace(0, 1, 100)[1:-1]))


# In[ ]:


print(np.quantile(radii25, 0.9))
print(np.quantile(radii50, 0.9))
print(np.quantile(radii, 0.9))


# In[ ]:


# my_hist(collection="radii", data=[radii25]+[radii50]+[radii], labels=['hagia25', 'hagia50', 'hagia'], bins=100)

my_hist(collection="radii", data=[radii25], labels=['hagia25'])
my_hist(collection="radii", data=[radii50], labels=['hagia50'], bins=100)
my_hist(collection="radii", data=[radii], labels=['hagia'], bins=100)

