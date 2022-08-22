
from multiprocessing.spawn import import_main_path
import numpy as np
import cv2
import json
import os
import pickle
import glob
import open3d as o3d
import utils.mkv2mp4 as mkv2mp4
import utils.task_compiler as task_compiler
import utils.armarker_localizer as armarker_localizer

import open3d
import numpy as np

def create_pointcloud(img, depth, intrinsic_kinect=None):
    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d.geometry.Image(img), open3d.geometry.Image(depth),
        depth_trunc=1.5,
        # depth_scale=1.0,
        convert_rgb_to_intensity=False)
    intrinsic = open3d.camera.PinholeCameraIntrinsic()
    intrinsic_matrix = intrinsic_kinect['intrinsic_matrix']

    intrinsic.set_intrinsics(
        intrinsic_kinect['width'],
        intrinsic_kinect['height'],
        intrinsic_matrix[0],
        intrinsic_matrix[4],
        intrinsic_matrix[6],
        intrinsic_matrix[7])
    #intrinsic = open3d.camera.PinholeCameraIntrinsic(
    #        open3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic,
        project_valid_depth_only=False)
    return pcd

def transform(self, xyz_array):
    # transform the coordinates to the original image
    if self.rot_mat_4x4_marker_to_camera is not None and xyz_array is not None:
        # copy the array
        tmp_xyz_array = copy.deepcopy(xyz_array)
        tmp_xyz_array.append(1)
        tmp_xyz_array_ret = np.dot(
            self.rot_mat_4x4_marker_to_camera, tmp_xyz_array)
        return tmp_xyz_array_ret[0:3].tolist()
    else:
        return xyz_array


if __name__ == '__main__':
    dev_list = [1,2,3,5,7,10]
    offset_z = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    height_list = [100, 80, 60, 40]
    color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    color_gray = (10, 10, 10)
    pcd_list = []
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i, dev in enumerate(dev_list):
    #for i, height in enumerate(height_list):
        fp_mkv = '../data/dev{}_h40.mkv'.format(dev)
        #fp_mkv = '../data/dev1_h{}.mkv'.format(height)
        output_dir = os.path.basename(fp_mkv).replace('.mkv', '')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #fp_mp4, fp_depth_mp4, fp_depth_npy = mkv2mp4.run(
        #    str(fp_mkv), output_dir, scale=1.0)
        #print(fp_mp4, fp_depth_mp4, fp_depth_npy)
        fp_depth_npy = '{}/{}_depth.npy'.format(output_dir, output_dir)
        fp_depth_mp4 = '{}/{}_depth.mp4'.format(output_dir, output_dir)
        fp_mp4 = '{}/{}.mp4'.format(output_dir, output_dir)
        fp_calib = os.path.join(output_dir, 'intrinsic.json')
        depth_img = np.load(str(fp_depth_npy))
        depth_img = depth_img.astype(np.float32)
        # extract the first frame of the video
        cap = cv2.VideoCapture(str(fp_mp4))
        ret, frame = cap.read()

        # load the camera model
        with open(str(fp_calib), 'r') as f:
            intrinsic = json.load(f)
        if ret:
            rgb_img = frame
        else:
            print('failed to read the first frame')
        open3d_cloud = create_pointcloud(
            rgb_img,
            depth_img[0],
            intrinsic_kinect=intrinsic)

        indices_image = np.arange(
            rgb_img.shape[0] * rgb_img.shape[1]).reshape(
                rgb_img.shape[0], rgb_img.shape[1])
        indices_image = np.array(indices_image, dtype=np.int32)

        # center area

        y_min = rgb_img.shape[0] // 2 - rgb_img.shape[0] // 4
        y_max = rgb_img.shape[0] // 2 + rgb_img.shape[0] // 4
        x_min = rgb_img.shape[1] // 2 - rgb_img.shape[1] // 8
        x_max = rgb_img.shape[1] // 2 + rgb_img.shape[1] // 8
        indices = indices_image[y_min:y_max, x_min:x_max]
        indices = indices.reshape(-1)
        whole_points = np.array(open3d_cloud.points)
        whole_colors = np.array(open3d_cloud.colors)
        center_points = whole_points[indices]
        center_colors = whole_colors[indices]
        # import pdb; pdb.set_trace()
        # change the color of the center area
        # center_colors[:, 0] = color_map[i][0]
        # center_colors[:, 1] = color_map[i][1]
        # center_colors[:, 2] = color_map[i][2]
        #center_colors[:, 0] = color_gray[0]
        #center_colors[:, 1] = color_gray[1]
        #center_colors[:, 2] = color_gray[2]
        # SET POINTS GRAY
        center_cloud = open3d.geometry.PointCloud()
        center_cloud.points = open3d.utility.Vector3dVector(center_points)
        center_cloud.colors = open3d.utility.Vector3dVector(center_colors)

        center_cloud.remove_non_finite_points()
        center_cloud = center_cloud.uniform_down_sample(10)
        print(center_cloud)
        print(np.asarray(center_cloud.points))
        #o3d.visualization.draw_geometries([open3d_cloud])
        # show rgb_img
        #cv2.imshow('rgb_img', rgb_img)
        #cv2.waitKey(0)
        rot_mat_4x4_marker_to_camera, frame = armarker_localizer.estimate_homo_transform_matrix(rgb_img, intrinsic)
        center_cloud.transform(rot_mat_4x4_marker_to_camera)
        open3d_cloud.transform(rot_mat_4x4_marker_to_camera)
        # offset the points
        points = np.array(center_cloud.points)
        # points[:, 2] = points[:, 2] + offset_z[i]
        center_cloud.points = open3d.utility.Vector3dVector(points)
        # pcd_list.append(center_cloud)
        print(center_cloud)
        vis.add_geometry(center_cloud)
        ctr = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera.json")
        ctr.convert_from_pinhole_camera_parameters(parameters)
        vis.run()
        vis.capture_screen_image('{}/{}.png'.format(output_dir, output_dir))
        vis.clear_geometries()
        # save image
        cv2.imwrite('{}/{}.jpg'.format(output_dir, output_dir), rgb_img)
        # save the point cloud
        o3d.io.write_point_cloud('{}/{}.pcd'.format(output_dir, output_dir), open3d_cloud)
        # plot 3d scatter using matplotlib
    #    import matplotlib
    #    import matplotlib.pyplot as plt
    #    fig = plt.figure()
    #    # center_cloud = center_cloud.uniform_down_sample(30)
    #    print(center_cloud)
    #    points = np.array(center_cloud.points)
    #    colors = np.array(center_cloud.colors)
    #    #ax = fig.add_subplot(111, projection='3d')
    #    #ax = fig.add_subplot(111, projection='3d')
    #    #ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.')
    #    ax = fig.add_subplot(111)
    #    ax.scatter(points[:, 1], points[:, 2], marker='.')
    #    #for point, c in zip(points, colors):
    #    #    ax.scatter(point[0], point[1], point[2], color=[c])
    #    # equal aspect ratio
    #    ax.set_aspect('equal')
    #    plt.show()