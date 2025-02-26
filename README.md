# Surface Reconstruction using Iterative Closest Point (ICP)

![results](images/icp.gif)
In this project we reconstuct the surface of a bunny rabbit from meshes captured from 6 different angles: 0, 45, 90, 180, 270 and 315. Using the algorithm, the 6 meshes are merged into one. 

## Overview
A few points are sampled from each mesh and correspondences between are established using 1-nearest neighbour. The rotation `R` and translation `t` matrices are calculated by an optimization algorithm—
1. Point-to-Point ICP: This variant of the algorithm minimizes the sum of squared distances between corresponding points in the two point clouds `A` and `B`.
1. Point-to-Plane ICP: This variant minimizes the distances between points and the corresponding  (found using the normals of point cloud `A`) fitted to the other point cloud `B`.
reference: Low, Kok-Lim. (2004). Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration. @see [Low, Kok-Lim. (2004)](https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf)


The manually implemented code helps understand the inner workings of ICP.  

## Running the code
1. Open the project directory
1. run `poetry install` and `poetry shell` to set up the environment and install dependencies. 
1. run `python3 icp.py` to run the code
1. feel free to change `method` to either `point-to-point` or `point_to_plane` and play with `max_iterations` and `threshold`. 
