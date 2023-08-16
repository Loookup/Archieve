#!/usr/bin/env python3
"""
uint8 TYPE_UNKNOWN = 0
uint8 TYPE_BOX     = 1
uint8 TYPE_ICEBOX  = 2
uint8 TYPE_POUCH   = 3
uint8 TYPE_SACK    = 4
uint8 TYPE_BOTTLE  = 5
"""

from unld_msgs.msg import ObjectArray
from unld_msgs.msg import Object
from sensor_msgs.msg import PointCloud2 as pc2
import rospy
import numpy as np
import open3d as o3d
import os
from open3d_ros_helper import open3d_ros_helper as orh
from geometry_msgs.msg import Pose, Vector3


def make_bbox_only(object_info, mesh):
    box_posi = object_info.pose.position

    box_ori = object_info.pose.orientation

    box_size = object_info.size

    translation = [box_posi.x, box_posi.y, box_posi.z]

    quaternion = [box_ori.w, box_ori.x, box_ori.y, box_ori.z]

    mesh_box = o3d.geometry.TriangleMesh.create_box(
        width=box_size.x, height=box_size.y, depth=box_size.z
    )
    mesh_box.compute_vertex_normals()  # create shadows
    mesh_box.paint_uniform_color([1.0, 0, 0])
    mesh_box.translate(translation, relative=False)
    mesh_box.rotate(
        R=o3d.geometry.get_rotation_matrix_from_quaternion(quaternion),
        center=translation,
    )

    if mesh == True:
        pass
    else:
        mesh_box = mesh_box.sample_points_uniformly(number_of_points=1000)

    return mesh_box


def ros_2_pcd(object_info):
    pcd = pc2()

    pcd = object_info.points

    o3dpc = orh.rospc_to_o3dpc(pcd)

    o3dpc.paint_uniform_color([0, 1.0, 0])

    return o3dpc


def Render_Whole_Boxes(whole_info, mesh):

    pcd = o3d.geometry.PointCloud()

    bbox_list = []

    if mesh == True:

        for idx in range(0, len(whole_info)):

            bbox = make_bbox_only(whole_info[idx], mesh)

            bbox_list.append(bbox)

        o3d.visualization.draw_geometries(bbox_list)

    else:
        bbox = make_bbox_only(whole_info[0], mesh)

        o3dpc = ros_2_pcd(whole_info[0])

        pre_stage = Unify_two_Geo(bbox, o3dpc)

        for idx in range(1, len(whole_info)):

            bbox = make_bbox_only(whole_info[idx], mesh)

            o3dpc = ros_2_pcd(whole_info[idx])

            cur_stage = Unify_two_Geo(bbox, o3dpc)

            pre_stage = Unify_two_Geo(pre_stage, cur_stage)

        o3d.visualization.draw_geometries([pre_stage])

        pcd = pre_stage

    return pcd

    # pcd = o3d.geometry.PointCloud()

    # pcd.points = o3d.utility.Vector3dVector(uni_load)

    # o3d.visualization.draw_geometries([pcd])


def Unify_two_Geo(Geo_1, Geo_2):
    
    p1_load = np.asarray(Geo_1.points)

    p2_load = np.asarray(Geo_2.points)

    p3_load = np.concatenate((p1_load, p2_load), axis=0)

    p1_color = np.asarray(Geo_1.colors)

    p2_color = np.asarray(Geo_2.colors)

    p3_color = np.concatenate((p1_color, p2_color), axis=0)

    pcdd = o3d.geometry.PointCloud()

    pcdd.points = o3d.utility.Vector3dVector(p3_load)

    pcdd.colors = o3d.utility.Vector3dVector(p3_color)

    return pcdd


def make_bbox_and_pcd(object_info):
    box_posi = object_info.pose.position

    box_ori = object_info.pose.orientation

    box_size = object_info.size

    # print(box_posi)

    # print(box_posi.x)

    # print(box_ori)
    # print(box_ori.x)

    # print(box_size)
    # print(box_size.x)

    pcd = pc2()

    pcd = object_info.points

    o3dpc = orh.rospc_to_o3dpc(pcd)

    o3dpc.paint_uniform_color([0, 1.0, 0])

    translation = [box_posi.x, box_posi.y, box_posi.z]

    quaternion = [box_ori.x, box_ori.y, box_ori.z, box_ori.w]

    if box_size.x <= box_size.y:
        width = box_size.y
        depth = box_size.x
        height = box_size.x
    else:
        depth = box_size.y
        width = box_size.x
        height = box_size.y

    mesh_box = o3d.geometry.TriangleMesh.create_box(
        width=width, height=height, depth=depth
    )
    # mesh_box.compute_vertex_normals() # create shadows
    mesh_box.paint_uniform_color([1.0, 0, 0])
    mesh_box.translate(translation, relative=True)
    mesh_box.rotate(
        R=o3d.geometry.get_rotation_matrix_from_quaternion(quaternion), center=[0, 0, 0]
    )

    pcd_box = mesh_box.sample_points_uniformly(number_of_points=1000)

    pcdd = Unify_two_Geo(o3dpc, pcd_box)

    o3d.visualization.draw_geometries([pcdd])

    # vis = o3d.visualization.VisualizerWithVertexSelection()
    # vis.create_window()
    # vis.add_geometry(o3dpc)
    # vis.add_geometry(pcd_box)
    # vis.run()
    # vis.destroy_window()

    # try:
    #     o3d.io.write_point_cloud("copy_seg_box_0.pcd", mesh_box)
    #     print("Succeed Saving")
    # except Exception as e:
    #     print(e)


# o3dpc.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])


def get_info(info):
    print(info[0])


def save_pointcloud(pcd):
    try:
        o3d.io.write_point_cloud("copy_2.pcd", pcd)
        print("Succeed Saving")
    except Exception as e:
        print(e)


def callback(msg):
    rospy.loginfo("Received Data")

    Whole_info = msg.data

    # get_info(Whole_info)

    pcd = Render_Whole_Boxes(whole_info=Whole_info, mesh=True)

    save_pointcloud(pcd)

    # make_bbox_and_pcd(Whole_info[1])

    # make_bbox_and_pcd(Whole_info[2])

    rospy.loginfo("----end----")
    os._exit(1)


if __name__ == "__main__":
    rospy.init_node("node", anonymous=True)
    rospy.loginfo("Initiate Node")
    rospy.Subscriber("/unld/detection/objects_array", ObjectArray, callback)
    rospy.spin()
