import numpy as np
import torch
import trimesh
import pyrender
import math
import imageio
import os
from scipy.spatial.transform import Rotation 

colors = {
        "cyan": [0, 255, 255, 1],
        "yellow": [255, 255, 0, 1],
        "blue": [26, 15, 162, 1],
        "grey": [77, 77, 77, 1],
        "grey_transparent": [77, 77, 77, 0.1],
        "ultra_bright_grey": [200, 200, 200, 1],
        "black": [0, 0, 0, 1],
        "white": [255, 255, 255, 1],
        "transparent": [255, 255, 255, 0],
        "magenta": [197, 27, 125, 1],
        'pink': [197, 140, 133, 1],
        'pink_transparent': [197, 140, 133, 0.1],
        "light_grey": [217, 217, 217, 255],
        "light_grey_transparent": [217, 217, 217, 0.1],
        'red': [162, 26, 15, 1],
        'green': [26, 162, 15, 1],
        'yellow_pale': [226, 215, 132, 1],
        'yellow_pale_transparent': [226, 215, 132, 0.1],
        }

marker2bodypart67 = {
    "head_ids": [12, 45, 9, 42, 6, 38],
    "mid_body_ids": [56, 35, 58, 24, 22, 0, 4, 36, 26, 1, 65, 33, 41, 8, 66, 35, 3, 4, 39],
    "left_hand_ids": [10, 11, 14, 31, 13, 17, 23, 28, 27],
    "right_hand_ids": [60, 43, 44, 47, 62, 46, 51, 57],
    "left_foot_ids": [29, 30, 18, 19, 7, 2, 15],
    "right_foot_ids": [61, 52, 53, 40, 34, 49, 40],
    "left_toe_ids": [32, 25, 20, 21, 16],
    "right_toe_ids": [54, 55, 59, 64, 50, 55]
}

marker2bodypart77 = {
    "head_ids": [12, 45, 9, 42, 6, 38],
    "mid_body_ids": [56, 35, 58, 24, 22, 0, 4, 36, 26, 1, 65, 33, 41, 8, 66, 35, 3, 4, 39],
    "left_hand_ids": [10, 11, 14, 31, 13, 17, 23, 28, 27],
    "right_hand_ids": [60, 43, 44, 47, 62, 46, 51, 57],
    "left_foot_ids": [29, 30, 18, 19, 7, 2, 15],
    "right_foot_ids": [61, 52, 53, 40, 34, 49, 40],
    "left_toe_ids": [32, 25, 20, 21, 16],
    "right_toe_ids": [54, 55, 59, 64, 50, 55],
    "left_finger_ids": [72, 73, 74, 75, 76],
    "right_finger_ids": [67, 68, 69, 70, 71]
}

bodypart2color = {
    "head_ids": 'cyan',
    "mid_body_ids": 'blue',
    "left_hand_ids": 'red',
    "right_hand_ids": 'green',
    "left_foot_ids": 'grey',
    "right_foot_ids": 'black',
    "left_toe_ids": 'yellow',
    "right_toe_ids": 'magenta',
    "left_finger_ids": 'red',
    "right_finger_ids": 'green',
    "special": 'light_grey'
}

def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:3]]

    return c

def tobodyparts(m_pcd, past=False):
    # m_pcd = np.array(m_pcd.vertices)
    # after trnaofrming poincloud visualize for each body part separately
    pcd_bodyparts = dict()
    for bp, ids in marker2bodypart77.items():
        points = m_pcd[ids]
        
        tfs = np.tile(np.eye(4), (points.shape[0], 1, 1))
        tfs[:, :3, 3] = points
        col_sp = trimesh.creation.uv_sphere(radius=0.01)

        # debug markers, maybe delete it
        # if bp == 'special':
        #     col_sp = trimesh.creation.uv_sphere(radius=0.03)

        if past:
            col_sp.visual.vertex_colors = c2rgba(colors["black"])
        else:
            col_sp.visual.vertex_colors = c2rgba(colors[bodypart2color[bp]])

        pcd_bodyparts[bp] = pyrender.Mesh.from_trimesh(col_sp, poses=tfs)
    return pcd_bodyparts

def toobj(obj_points):

        
    tfs = np.tile(np.eye(4), (obj_points.shape[0], 1, 1))
    tfs[:, :3, 3] = obj_points
    col_sp = trimesh.creation.uv_sphere(radius=0.01)

    # debug markers, maybe delete it
    # if bp == 'special':
    #     col_sp = trimesh.creation.uv_sphere(radius=0.03)
    col_sp.visual.vertex_colors = c2rgba(colors["pink"])
    return pyrender.Mesh.from_trimesh(col_sp, poses=tfs)


    


def plot_markers(save_path,m_pcd,obj_points):
    width = 640
    height = 480
    figsize = (width, height)

    seqlen = len(m_pcd)
    video_writer = imageio.get_writer(save_path, fps=30)
    scene = pyrender.Scene()
    pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
    camera_pose = np.eye(4)
    rotate=trimesh.transformations.rotation_matrix(np.radians(-30.0),
                                                    [1,0,0])
    camera_pose[:3, 3] = np.array([0, 2, 2.5])
    camera_node = scene.add(pc, pose=camera_pose, name='pc-camera')
    camera_pose = np.dot(camera_pose, rotate)
    viewer = pyrender.OffscreenRenderer(*figsize)
    for i in range(seqlen):
        pcd_bodyparts = tobodyparts(m_pcd[i])
        # visualize the body parts to video
        for node in scene.get_nodes():
            if node.name is not None and '-mesh' in node.name:
                scene.remove_node(node)
        for bp, mesh in pcd_bodyparts.items():
            scene.add(mesh,name='%s-mesh'%bp)
        obj_mesh = toobj(obj_points[i])
        scene.add(obj_mesh,name='obj-mesh')
        # show the scen


        color_img, depth_img = viewer.render(scene)
        color_img= color_img.astype(np.uint8)
        video_writer.append_data(color_img)
    video_writer.close()
    del viewer

if __name__ == '__main__':
    # name = 'Date01_Sub01_backpack_back_20'
    # m_pcd = np.load('./dataset/behave/markers/'+name+'/marker.npy')
    # motion_obj = np.load('./dataset/behave/markers/'+name+'/object.npy')
    # save_path = './dataset/behave/markers/'+name+'/'+name+'_marker.mp4'
    # obj_name = name.split('_')[2]
    # width = 640
    # height = 480
    # figsize = (width, height)

    # seqlen = len(m_pcd)
    # video_writer = imageio.get_writer(save_path, fps=30)
    # scene = pyrender.Scene()
    # pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=float(width) / height)
    # camera_pose = np.eye(4)
    # rotate=trimesh.transformations.rotation_matrix(np.radians(-30.0),
    #                                                 [1,0,0])
    # camera_pose[:3, 3] = np.array([0, 2, 2.5])
    # camera_node = scene.add(pc, pose=camera_pose, name='pc-camera')
    # camera_pose = np.dot(camera_pose, rotate)
    # viewer = pyrender.OffscreenRenderer(*figsize)

    # mesh_path = os.path.join("./dataset/behave/objects", obj_name, obj_name + ".obj")
    # obj_mesh = trimesh.load_mesh(mesh_path)
    # vertices, faces = obj_mesh.vertices, obj_mesh.faces
    # obj_angles = motion_obj[:,:3]
    # obj_trans = motion_obj[:,3:]
    # angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    # obj_verts = (vertices)[None, ...]
    # obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    # print(obj_verts[0][1])
    # print(m_pcd[0][1])

    # for i in range(seqlen):
    #     pcd_bodyparts = tobodyparts(m_pcd[i])
    #     # visualize the body parts to video
    #     for node in scene.get_nodes():
    #         if node.name is not None and '-mesh' in node.name:
    #             scene.remove_node(node)
    #     for bp, mesh in pcd_bodyparts.items():
    #         scene.add(mesh,name='%s-mesh'%bp)
    #     # show the scen
    #     obj_points = obj_verts[i]
    #     obj_mesh = toobj(obj_points)
    #     scene.add(obj_mesh,name='obj-mesh')

    #     color_img, depth_img = viewer.render(scene)
    #     color_img= color_img.astype(np.uint8)
    #     video_writer.append_data(color_img)
    # video_writer.close()
    # del viewer
    i = 0
    results = np.load('./save/behave_enc_512/samples_behave_enc_512_000001000_seed11/results.npy', allow_pickle=True)
    results = results.item()
    
    motion_obj = results['motion_obj'][i]
    motion = results['motion'][i].transpose(2, 0, 1)
    obj_name = results['obj_name'][i].split('_')[2]
    text = results['text'][i]
    print(text)
    mesh_path = os.path.join("./dataset/behave/objects", obj_name, obj_name + ".obj")
    obj_mesh = trimesh.load_mesh(mesh_path)
    vertices = obj_mesh.vertices
    motion_obj = motion_obj.squeeze()
    obj_angles = motion_obj[:3].transpose()
    obj_trans = motion_obj[3:].transpose()

    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts = (vertices)[None, ...]
    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    
    plot_markers('./save/behave_enc_512/samples_behave_enc_512_000001000_seed11/sample{}.mp4'.format(i), motion, obj_verts)


