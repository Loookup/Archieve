#!/usr/bin/env python3

# import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2 as pc2
import open3d as o3d
import rospy
from open3d_ros_helper import open3d_ros_helper as orh
import numpy as np
import os
import matplotlib as plt
from scipy.spatial import Delaunay
from collections import defaultdict
import pcl
import time
import cv2


def get_normal_plane(normal_vector, Point):

    d = 0

    for idx in range(0, 3):
        d -= normal_vector[idx]*Point[idx]

    plane_coeffi = np.hstack((normal_vector, d))

    return plane_coeffi


def filter_point_cloud(Cloud, plane_coeffi, dist_offset):

    max = len(Cloud)

    denominator = np.sqrt(plane_coeffi[0]**2 + plane_coeffi[1]**2 + plane_coeffi[2]**2)

    filtered_points = []

    for idx in range(0, max):

        temp_dist = abs(plane_coeffi[0]*Cloud[idx][0] + plane_coeffi[1]*Cloud[idx][1] + plane_coeffi[2]*Cloud[idx][2] + plane_coeffi[3]) / denominator

        if temp_dist <= dist_offset:

            filtered_points.append(Cloud[idx])
        else: pass

    return filtered_points


def RANSAC(pcd):

    t1 = time.time()

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=100)

    inlier_cloud = pcd.select_by_index(inliers)

    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    inlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])

    outlier_cloud.paint_uniform_color([1, 0, 0])

    t2 = time.time()

    print(f"Time to segment points using RANSAC {t2 - t1}")

    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return plane_model



# def RANSAC(coeffi):



#     pc = pcl.load("/home/chulyonglee/unload_ws/pointcloud_archieve/seg_0_inner.pcd")

#     model_p = pcl.SampleConsensusModelPlane(cloud)
#     if argc > 1:
#         if argvs == "-f":
#             ransac = pcl.RandomSampleConsensus (model_p)
#             ransac.set_DistanceThreshold (.01)
#             ransac.computeModel()
#             inliers = ransac.get_Inliers()
#         elif argvs == "-sf":
#             ransac = pcl.RandomSampleConsensus (model_s)
#             ransac.set_DistanceThreshold (.01)
#             ransac.computeModel()
#             inliers = ransac.get_Inliers()
#         else:
#             inliers = []
#     else:
#         inliers = []




def cal_inner_product_abs(depth_dir, n_vector):

    dot_result = np.dot(depth_dir, n_vector)

    return np.abs(dot_result)


def get_normal_vector(triangles):

    x0, y0, z0 = triangles[0]
    x1, y1, z1 = triangles[1]
    x2, y2, z2 = triangles[2]

    vector_1 = np.array([x1-x0, y1-y0, z1-z0])
    vector_2 = np.array([x2-x0, y2-y0, z2-z0])

    n_vector = np.array([vector_1[1]*vector_2[2]-vector_1[2]*vector_2[1], -vector_1[0]*vector_2[2]+vector_1[2]*vector_2[0],
                        vector_1[0]*vector_2[1]-vector_1[1]*vector_2[0]])
    
    length = np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)

    unit_n_vec = n_vector / length

    return unit_n_vec



def alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs 
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra.vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

    # Find tetrahedrals
    tetras = tetra.vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)

    return Vertices, Edges, Triangles





def make_triangles_and_draw(Points, Points_bound):

    Vertices,Edges,Triangles = alpha_shape_3D(Points, 1)

    num = len(Triangles)

    triangles_list = []
    unit_normal_vector_list = []

    abs_dot_result_list = []

    depth_dir = np.array([1, 0, 0])
    
    for idx in range(0, num):

        triangle_mesh = draw_triangle(Points[Triangles[idx]])

        triangles_list.append(triangle_mesh)

        unit_normal_vector = get_normal_vector(Points[Triangles[idx]])

        unit_normal_vector_list.append(unit_normal_vector)

        abs_dot_result = cal_inner_product_abs(depth_dir, unit_normal_vector)

        abs_dot_result_list.append(abs_dot_result)

        # unit_normal_vector_list.append(unit_normal_vector)

    front_estimated_idx = np.argmax(abs_dot_result_list) # front_estimated_idx = 147

    ref_unit_normal_vector = unit_normal_vector_list[front_estimated_idx]

    # print(ref_unit_normal_vector)

    ref_plane_coeffi = get_normal_plane(ref_unit_normal_vector, Points[Triangles[front_estimated_idx]][0])

    dist_offset = 0.1

    filtered_pcd = filter_point_cloud(Points, ref_plane_coeffi, dist_offset)

    # print(len(filtered_pcd))

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(filtered_pcd)

    plane_model = RANSAC(pcd)

    # print(ref_plane_coeffi)

    # print(plane_model)

    triangles_list.pop(front_estimated_idx)

    front_estimated_triangle_mesh = draw_triangle(Points[Triangles[front_estimated_idx]])

    front_estimated_triangle_mesh.paint_uniform_color([1, 0, 0])

    triangles_list.append(front_estimated_triangle_mesh)

    # o3d.visualization.draw_geometries(triangles_list)

    proj_points = project_to_plane(Points_bound, plane_model)

    pcd_proj = o3d.geometry.PointCloud() 

    pcd_proj.points = o3d.utility.Vector3dVector(proj_points)

    pcd_proj.paint_uniform_color([1.0, 0, 0])

    pcd_un_proj = o3d.geometry.PointCloud()

    pcd_un_proj.points = o3d.utility.Vector3dVector(Points_bound)

    pcd_un_proj.paint_uniform_color([0, 1.0, 0])

    # hull, _ = pcd_proj.compute_convex_hull()

    pcd_proj.get_oriented_bounding_box()
    # print(hull.get_volume()) #It should be 1 but it gives me 0
    # print(hull.get_oriented_bounding_box().volume()) 

    # o3d.visualization.draw_geometries([pcd_proj, pcd_un_proj])

    # bot = pro[:,:2]

    # key = cv2.waitKey(0)

    # while key is not 27:

    #     cv2.imshow("img", bot)


    # hull = o3d.geometry.compute_point_cloud_convex_hull(pcd_proj)

    # hull = pcd_proj.compute_convex_hull()

    # o3d.visualization.draw_geometries([hull[0]])



def project_to_plane(points, plane_coeffi):

    max = len(points)

    a, b, c, d = plane_coeffi[0], plane_coeffi[1], plane_coeffi[2], plane_coeffi[3]
    
    proj_points_list = []

    # print(max)
    
    for idx in range(0, max):

        x1, y1, z1 = points[idx][0], points[idx][1], points[idx][2]

        t = -(a*x1 + b*y1 + c*z1 + d) / np.sqrt(a**2 + b**2 + c**2)

        proj_point = [a*t+x1, b*t+y1, c*t+z1]

        proj_points_list.append(proj_point)

    return proj_points_list







def find_boundary(pcd):

    Points = np.asarray(pcd.points)

    # o3d.visualization.draw_geometries([pcd])

    Colors = np.asarray(pcd.colors)

    th = 165 # 0, 33, 102, 165, 234

    indicator = Colors[th]

    index = 0

    for idx in range(th+1, len(Colors)):

        if np.all(Colors[idx] == indicator): pass
        else: 
            index = idx -1

            break

    pcd_bound = o3d.geometry.PointCloud() 

    pcd_bound.points = o3d.utility.Vector3dVector(Points[th:index])

    pcd_bound.colors = o3d.utility.Vector3dVector(Colors[th:index])

    o3d.visualization.draw_geometries([pcd_bound])

    return index 





def draw_triangle(three_vertices):

    mesh = o3d.geometry.TriangleMesh()

    np_triangles = np.array([[0, 1, 2]]).astype(np.int32)

    mesh.vertices = o3d.utility.Vector3dVector(three_vertices)

    mesh.triangles = o3d.utility.Vector3iVector(np_triangles)

    mesh.paint_uniform_color([0, 1, 0])

    return mesh


def pcd_save(pcd):

    try:  
        o3d.io.write_point_cloud("cam0_seg_0_boundary.pcd", pcd)
        print("Succeed Saving")
    except Exception as e:
        print(e)

    

def read_file():

    pcd_0_bound = o3d.io.read_point_cloud("/home/chulyonglee/unload_ws/pointcloud_archieve/cam0_seg_0_boundary.pcd")
    pcd_0_inner = o3d.io.read_point_cloud("/home/chulyonglee/unload_ws/pointcloud_archieve/cam0_seg_0_inner.pcd")


    Points_inner = np.asarray(pcd_0_inner.points)

    Points_bound = np.asarray(pcd_0_bound.points)

    # index = find_boundary(pcd, th)

    # o3d.visualization.draw_geometries([pcd])

    make_triangles_and_draw(Points_inner, Points_bound)
    
    # if isinstance(Triangles, type(pcd2)):
    #     pass
    # elif isinstance(Triangles, np.ndarray):
    #     pcd2.points = o3d.utility.Vector3dVector(Triangles)

    # print(Vertices,Edges,Triangles)

    # o3d.visualization.draw_geometries([pcd])

    print("---end---")
    os._exit(1)




def subscribe_pcd():

    rospy.Subscriber('/obb_calculator/cam0/boundary', pc2, callback)
    # rospy.Subscriber('/cam0/k4a/points2/voxel', pc2, callback)

    rospy.spin() 

def callback(pc):

        #pc = pcl.VoxelGridFilter(.pc)

    rospy.loginfo(("received"))

    o3dpc = orh.rospc_to_o3dpc(pc)

    o3d.visualization.draw_geometries([o3dpc])

    pcd = o3d.geometry.PointCloud()

    pcd = np.asarray(o3dpc.points)

    print(pcd)

    # o3dpc.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    print("---end---")
    os._exit(1)


# def RANSAC_Plane(pcd):

#     plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                             ransac_n=3,
#                                             num_iterations=1000)
#     [a, b, c, d] = plane_model
#     print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

#     inlier_cloud = pcd.select_by_index(inliers)
#     inlier_cloud.paint_uniform_color([1.0, 0, 0])
#     outlier_cloud = pcd.select_by_index(inliers, invert=True)
#     o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
   



if __name__ == "__main__":

    # rospy.init_node('node', anonymous=True)
    read_file()
    # subscribe_pcd()



