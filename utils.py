import open3d as o3d
import numpy as np
import trimesh
from typing import Optional


def trimeshToO3D(mesh:trimesh, rgb:Optional[list]= None, points:Optional[bool]= False):
    mesh_o3d= o3d.geometry.TriangleMesh()
    mesh_o3d.vertices=o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles=o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.compute_vertex_normals()
    if rgb is None:
        # compute rgb colour based on normals
        vex_color_rgb = np.abs(mesh_o3d.vertex_normals)
    else:
        vex_color_rgb = np.ones([mesh.vertices.shape[0],3]) * rgb
    mesh_o3d.vertex_colors=o3d.utility.Vector3dVector(vex_color_rgb)

    if points == True:
        mesh_o3d= o3d.geometry.PointCloud()
        mesh_o3d.points = o3d.utility.Vector3dVector(mesh.vertices)
    return mesh_o3d

def visualize(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Visualizer')
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def turntable(mesh, save:Optional[bool]=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Visualizer')
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    
    # move the mesh back to fit everything in frame
    
    vis.add_geometry(mesh)
    mesh = mesh.translate([0, 0, -0.05])
    rot = 8
    for i in range(int(360/rot)):
        # rotate the mesh by 1 degree
        mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, rot*(np.pi/180), 0)), center=mesh.get_center())
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        if save:
            vis.capture_screen_image(f"turntable/{i}.png")

    vis.destroy_window()

def viz_3d(p_mesh: trimesh,
           q_mesh: trimesh,
           intermediate_points: list,
                  RES_PATH: str,
                  image_index: Optional[int]=0,
                  save_image:Optional[bool] = True):
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # background is black
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    
    initial = trimeshToO3D(p_mesh, [1,0,0])
    final = trimeshToO3D(q_mesh, [0,1,0])
    vis.add_geometry(initial)
    vis.add_geometry(final)
    save_image = True
    
    for i,update_point in enumerate(intermediate_points):
        final.vertices = o3d.utility.Vector3dVector(update_point)
        vis.update_geometry(final)
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            vis.capture_screen_image(f"{RES_PATH}/{i+image_index}.png")
    vis.destroy_window()

def trans_trimesh(tm:trimesh, 
                  angle:int, 
                  shift:Optional[int]=None):
    # rotation
    rad = angle / 180 * np.pi
    R = np.array([[np.cos(rad), 0, np.sin(rad)],
                  [0, 1, 0],
                  [-np.sin(rad), 0, np.cos(rad)]])
    homo_R = np.eye(4)
    homo_R[:3, :3] = R

    # shift
    if shift is None:
        homo_T = np.eye(4)
    else:
        centroid = np.mean(np.array(tm.vertices), axis=0)
        homo_T = np.eye(4)
        homo_T[:3, 3] = -centroid + np.array(shift)

    tm.apply_transform(homo_R @ homo_T)