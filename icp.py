import os
import trimesh
import numpy as np
from typing import Optional
from sklearn.neighbors import NearestNeighbors
import cv2

from utils import viz_3d, trans_trimesh, turntable, trimeshToO3D

import warnings 
warnings.filterwarnings("ignore")

RES_PATH = "images/"
MESH_PATH = "resources/bunny_v2"

def point_to_point(matched_p: list, 
                   matched_q: list
    ) -> tuple[np.ndarray, np.ndarray]:

    p_mean = np.mean(matched_p, axis=0)
    q_mean = np.mean(matched_q, axis=0)

    p_cen = matched_p - p_mean
    q_cen = matched_q - q_mean

    H = np.dot(p_cen.T, q_cen)
    U, S, VT = np.linalg.svd(H)

    I = np.eye(3)
    R = np.dot(VT.T, U.T)
    I[2,2] = np.linalg.det(R)
    R = np.dot(VT.T, np.dot(I,U.T))
    t = p_mean - np.dot(q_mean, R)
    return R, t

def create_euler(T):
    a = 1
    b = T[0]*T[1] - T[2]
    c = T[0]*T[2] + T[1]

    d = T[2]
    e = T[0]*T[1]*T[2] + 1
    f = T[1]*T[2] - T[0]

    g = -T[1]
    h = T[0]
    i = 1

    R = [[a, b, c], [d, e, f], [g, h, i]]
    t = [T[3], T[4], T[5]]

    return R, t

def point_to_plane(matched_p: list, 
                   matched_q: list, 
                   normals: list
                   ) -> tuple[np.ndarray, np.ndarray]:
    A = []
    b = []
    for i in range(len(normals)):
        a_row = np.cross(normals[i], matched_q[i])
        A.append(a_row)
        b_row = np.dot(normals[i], matched_p[i] - matched_q[i])
        b.append(b_row)
    A_fin = np.zeros((len(matched_p), 6))
    A_fin[:,0:3] = np.array(A)
    A_fin[:,3:6] = np.array(normals)

    A_fin = np.array(A_fin).reshape(len(normals), 6)
    b = np.array(b)

    tr = np.dot(np.linalg.pinv(A_fin), b)
    R, t = create_euler(tr)

    return R, t

def icp(p_mesh: trimesh, 
        q_mesh: trimesh, 
        max_iterations: Optional[int]=50, 
        tolerance: Optional[int]=0.001,
        sampling: Optional[bool]=True, 
        method: Optional[str]='point_to_point'
    ) -> tuple[list, np.ndarray, np.ndarray, list]:
    
    q_mesh_copy = q_mesh.copy()
    
    p = p_mesh.vertices
    p_norm = p_mesh.vertex_normals

    q = q_mesh_copy.vertices
    q_norm = q_mesh_copy.vertex_normals
    
    if sampling:
        # change q and q_norm to be only the sampled points!
        # uniform sampling 
        threshold = 0.7
        ids = np.random.uniform(0, 1, size=p.shape[0])
        p = p[ids < threshold]
        p_norm = p_mesh.vertex_normals[ids < threshold]

    p_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    p_tree.fit(p)

    intermediate_points = []
    intermediate_points.append(q_mesh_copy.vertices)
    
    errors = []
    for _ in range(max_iterations):
        distances, closest_indices = p_tree.kneighbors(q, return_distance=True)
        distances = distances.ravel()
        closest_indices = closest_indices.ravel()
        indexes  = [closest_indices[i] for i in range(closest_indices.shape[0])]

        matched_p = p[indexes].reshape(len(indexes),3)
        matched_p_norm = p_norm[indexes].reshape(len(indexes),3)
        matched_q = q.reshape([len(indexes),3])

        # reject matches that are too far (find cos(theta) between normals)
        matched_q_norm = q_norm.copy()

        # find angles between matched normals
        cos_thetas = np.sum(matched_p_norm * matched_q_norm, axis=1)/(np.linalg.norm(matched_p_norm, axis=1)*np.linalg.norm(matched_q_norm, axis=1))
        angles = np.arccos(cos_thetas) / np.pi * 180
        # generate cosine_threshold flags
        angle_threshold = 30
        cosine_threshold = angles < angle_threshold
        matched_p = matched_p[cosine_threshold, :]
        matched_q = matched_q[cosine_threshold, :]  
        matched_p_norm = matched_p_norm[cosine_threshold, :]

        assert matched_p.shape == matched_q.shape, "dimension mismatch in ICP"

        if method == "point_to_point":
            R, t = point_to_point(matched_p, matched_q)
            q = np.dot(q, R) + t
        elif method == "point_to_plane":
            R, t = point_to_plane(matched_p, matched_q, matched_p_norm)
            q = np.dot(q, R) + t
        else:
            raise ValueError(f"Unknown method: {method}")

        intermediate_points.append(q)

        # calculate mean error using new distances
        distances, _ = p_tree.kneighbors(q, return_distance=True)
        distances = distances.ravel()

        mean_error = np.mean(distances)
        print(f"mean error: {mean_error}")
        errors.append(mean_error)
        
        if abs(mean_error) < tolerance:
            print('\nbreak iteration, the distance between two adjacent iterations '
                      'is lower than tolerance (%0000f < %0000f)'
                      % (np.abs(mean_error), tolerance))
            break

    # get the final transformation matrices R and t from the most recent q and the init q 
    # although they must have the same shape, so we use matched_p_norm from last loop
    q_orig = q_mesh.vertices   
    q_orig = q_orig[:q.shape[0],:]
    print(q.shape, q_orig.shape)
    assert q_orig.shape == q.shape, "dimension mismatch in ICP"
    R_fin, t_fin = point_to_point(q, q_orig)

    return intermediate_points, R_fin, t_fin, errors

if __name__ == "__main__":
    # load all meshes from the directory in order
    mesh_paths = ["bun270_v2", "bun315_v2","bun000_v2", "bun045_v2", "bun090_v2", "bun180_v2"]
    angles = [270, 315, 0, 45, 90, 180]
    update_point_list  = []
    p_mesh = trimesh.load(os.path.join(MESH_PATH, mesh_paths[0] + ".ply"))
    niter=0
    trans_trimesh(p_mesh, angles[0])

    for i in range(1, len(mesh_paths)):
        q_mesh = trimesh.load(os.path.join(MESH_PATH, mesh_paths[i] + ".ply"))

        # transform the q_mesh to be orientied at 0 degrees
        trans_trimesh(q_mesh, angles[i])

        intermediate_points, R, t, errors = icp(p_mesh, q_mesh, method="point_to_plane", max_iterations=10)

        src_mesh = q_mesh.copy()
        src_mesh.vertices = np.dot(src_mesh.vertices, R) + t
        new_p = p_mesh + src_mesh
        new_mesh = trimeshToO3D(new_p, [1,0,0])
        # visualize(new_mesh)

        viz_3d(p_mesh=p_mesh, 
               q_mesh=q_mesh, 
               intermediate_points=intermediate_points, 
               RES_PATH=RES_PATH, 
               image_index=niter)
        
        # for next iteration
        niter += len(intermediate_points)
        p_mesh += src_mesh
        
    # save the final mesh 
    p_mesh.export(os.path.join(RES_PATH, "final_mesh.ply"))

    # # load mesh
    # p_mesh = trimesh.load(os.path.join(RES_PATH, "final_mesh.ply"))
    # convert to o3d mesh and visualize the turntable
    p_mesh = trimeshToO3D(mesh=p_mesh)
    turntable(p_mesh)

    # create a video from image sequence
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('images/icp.avi', fourcc, 12, (1920, 1080), isColor=True)
    dir1 = "images/"
    dir2 = "turntable/"
    images = []
    for i in range(niter):
        out.write(cv2.imread(f"{dir1}/{i}.png"))
    for i in range(45):
        out.write(cv2.imread(f"{dir2}/{i}.png"))
    out.release()

