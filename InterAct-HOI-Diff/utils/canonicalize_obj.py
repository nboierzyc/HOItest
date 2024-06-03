if __name__ == "__main__":
    datasets = ['behave', 'intercap', 'neuraldome', 'grab', 'chairs', 'omomo', 'imhd']
    data_root = '/media/volume/InterAct/data'
    for dataset in datasets:
        dataset_path = os.path.join(data_root, dataset)
        MOTION_PATH = os.path.join(dataset_path, 'sequences')
        OBJECT_PATH = os.path.join(dataset_path, 'objects')
        data_name = os.listdir(MOTION_PATH)
        for name in data_name:
            print(name)
            if dataset == 'grab':
                verts, faces = visualize_grab(name, MOTION_PATH)
                markers = verts[:,markerset_smplx]
            elif dataset == 'intercap' or dataset == 'behave':
                verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 10)
                markers = verts[:,markerset_smplh]
            elif dataset == 'neuraldome' or dataset == 'imhd':
                verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 16)
                markers = verts[:,markerset_smplh]
            elif dataset == 'chairs':
                verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 10)
                markers = verts[:,markerset_smplx]
            elif dataset == 'omomo':
                verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 16)
                markers = verts[:,markerset_smplx]
            np.save(os.path.join(MOTION_PATH, name, 'markers.npy'), markers)

            with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
                obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])

            mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
            
            obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

            centroid = obj_verts.mean(axis=0, keepdims=True).copy()
            
            new_obj_verts = obj_verts - centroid

            mesh_obj.export(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"))
            new_obj_trans = obj_trans - centroid
            print(centroid)
            
            
            angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
            obj_verts = (new_obj_verts)[None, ...]
            obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + new_obj_trans[:, None, :]
            rend_video_path = os.path.join('./marker_results', '{}_{}_{}.mp4'.format(dataset, name, obj_name))
            plot_markers(rend_video_path, markers ,obj_verts)
            break