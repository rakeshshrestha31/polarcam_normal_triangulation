#!/usr/bin/env python

import struct
import numpy as np
import cv2
import math
import functools
import json

import gtk, gtk.gdk as gdk, gtk.gtkgl as gtkgl, gtk.gdkgl as gdkgl
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from GLZPR import GLZPR, _demo_draw

def read_depth_img(path):
    with open(path, 'rb') as f:
        f.readline()
        max_depth = float(str(f.readline()).split()[1])
        [width, height] = [int(x) for x in str(f.readline()).split()]
        img = np.zeros(shape=(height, width))
        f.readline()
        for x in range(height):
            for y in range(width):
                n = f.read(2)
                num = struct.unpack('H', n)[0] #int.from_bytes(n, byteorder='little')
                num /= 65535 / max_depth
                img[x, y] = num
        return img, max_depth

def read_azimuth_img(path):
	array = []
	with open(path, 'rb') as f:
		while True:
			float_packed = f.read(4)
			if not float_packed:
				break
			array.append(struct.unpack('f', float_packed))
	array = np.asarray(array)
	array = np.resize(array, (600, 772))
	print(array.shape)
	print(np.amax(array))
	return array

def list_to_transformation_pose(list_pose):
	# R, t = (np.resize(list_pose[:9], (3,3)), list_pose[-4:-1])
	# T = np.eye(4)
	# T[:3, :3] = R
	# T[:3, 3] = t
	T = np.resize(list_pose, (4, 4))

	# T[:3, :3] = R.transpose()
	# T[:3, 3] = -R.transpose().dot(t)

	return T

def transform(T, p):
	'''
	T: 4x4 array
	p: 3 tuple
	'''
	p_homogeneous = T.dot(np.resize(p+(1,), (4, 1)))
	return tuple(p_homogeneous[:3].transpose()[0])

def read_trajectory(path):
	with open(path, 'r') as f:
		data = json.load(f)
	data = dict([
		(int(i['file_name']), list_to_transformation_pose(i['world_pos'])) for i in data
	])
	return data

def image_to_3d(u, v, z, intrinsics):
	x = (float(u) - intrinsics[2])/intrinsics[0] * z
	y = (float(v) - intrinsics[3])/intrinsics[1] * z
	return x,y,z

def threeD_to_image(xyz, intrinsics):
	x, y, z = xyz
	u = (intrinsics[0] * x / z) + intrinsics[2]
	v = (intrinsics[1] * y / z) + intrinsics[3]
	return int(round(u)), int(round(v))

def depth_to_pointcloud(depth_img, intrinsics):
	points = [(0,0,0) for i in range(depth_img.size)]
	i = 0
	max_depth = float(np.amax(depth_img))
	for row in range(0, depth_img.shape[0]):
		for col in range(0, depth_img.shape[1]):
			d = float(depth_img[row, col])/max_depth
			if d > 0.1:
				points.append(image_to_3d(col, row, d, intrinsics))
	return points

def compute_point_cloud_with_normal(depth_img1, azimuth_img1, pose1, depth_img2, azimuth_img2, pose2, intrinsics):
	'''
	Calculates normal for points based on depth image and azimuth image
	'''
	max_depth = float(np.amax(depth_img1))
	T_21 = np.linalg.inv(pose2).dot(pose1)
	# T_21 = pose2.dot(np.linalg.inv(pose1))

	T_12 = np.linalg.inv(T_21)
	width = depth_img1.shape[1]
	height = depth_img1.shape[0]

	num_overlap = 0
	pointcloud = []

	for row in range(depth_img1.shape[0]):
		for col in range(depth_img1.shape[1]):
			d1 = float(depth_img1[row, col])
			
			if (d1 < (max_depth * 0.1)):
				continue
			xyz1 = image_to_3d(col, row, d1, intrinsics)
			
			# debug
			pointcloud.append(xyz1)
			continue
			# project and find out if there is overlap
			xyz2 = transform(T_21, xyz1)
			u, v = threeD_to_image(xyz2, intrinsics)
			

			# print xyz2, xyz2_gt			
			if u < 0 or u >= width or v < 0 or v >= height:
				continue

			d2 = float(depth_img2[v, u])
			xyz2_gt = image_to_3d(u, v, d2, intrinsics)
			# in the first camera frame
			# xyzw2_gt = T_12.dot(xyz2_gt + (1,))
			# xyz2_gt = xyzw2[:3]

			# if math.fabs(d2 - xyz2[2]) > max_depth * 0.2:
			# 	continue

			# print xyz2, xyz2_gt

			# pointcloud.append(xyz2_gt)
			
			pointcloud.append(xyz2)		
			pointcloud.append(xyz2_gt)
			num_overlap += 1

	for row in range(depth_img2.shape[0]):
		for col in range(depth_img2.shape[1]):
			d2 = float(depth_img2[row, col])
			
			if (d2 < (max_depth * 0.1)):
				continue	

			xyz2 = image_to_3d(col, row, d2, intrinsics)
			
			#debug
			pointcloud.append(transform(T_12, xyz2))
			continue

			# project and find out if there is overlap
			xyzw2_world = pose2.dot(np.resize(xyzw2, (4, 1)))
			pointcloud.append(xyzw2_world[:3].transpose()[0])	

	print num_overlap
	return pointcloud

def triangulate_azimuth(azimuth1, pose1, azimuth2, pose2):
	'''
	Calculates normal given azimuth and pose of two frames
	'''
	projection1 = np.array([math.cos(azimuth1), math.sin(azimuth1)]).transpose()
	projection2 = np.array([math.cos(azimuth2), math.sin(azimuth2)]).transpose()

	# the z axis of image plane
	image_plane_normal1 = pose1[:3, 2]
	image_plane_normal2 = pose2[:3, 2]

	cross1 = np.cross(projection1, image_plane_normal1)
	cross2 = np.cross(projection2, image_plane_normal2)

	point_normal = np.cross(cross1, cross2)

	return point_normal
		
def draw_point_cloud(pointcloud, event):
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
	
	glBegin(GL_POINTS);
	for point in pointcloud:
		glVertex3f(point[0], point[1], point[2])
	glEnd()

def save_point_cloud(pointcloud, filename):
	with open(filename, 'w') as f:
		f.write('OFF\n')
		f.write('%d 0 0\n' % len(pointcloud))
		for point in pointcloud:
			f.write('%f %f %f\n' % (point[0], point[1], point[2]))


if __name__ == '__main__':
	base_path = '/home/rakesh/workspace/polar_cam/fish_vase_zp'
	import sys, os
	if len(sys.argv) < 3:
		print('Usage: python normal_reconstruction.py <img1_num> <img2_num>')
		exit(0)
	img1_num = int(sys.argv[1])
	img2_num = int(sys.argv[2])

	img1_str = "%.4d" % img1_num
	img2_str = "%.4d" % img2_num

	print img1_str, img2_str

	azimuth_img1 = read_azimuth_img(os.path.join(base_path, 'az', img1_str + '.bin'))
	depth_img1, max_depth1 = read_depth_img(os.path.join(base_path, 'depth', img1_str + '.pgm'))
	
	azimuth_img2 = read_azimuth_img(os.path.join(base_path, 'az', img2_str + '.bin'))
	depth_img2, max_depth2 = read_depth_img(os.path.join(base_path, 'depth', img2_str + '.pgm'))

	trajectory = read_trajectory('/home/rakesh/workspace/polar_cam/fish_vase_zp/keyframes.json')
	pose1 = trajectory[img1_num]
	pose2 = trajectory[img2_num]

	print pose1
	print pose2

	with open('/home/rakesh/workspace/polar_cam/fish_vase_zp/parameter.json') as f:
		config = json.load(f)

	intrinsics = config['cam_intrinsic_param']

	pointcloud = compute_point_cloud_with_normal(
		depth_img1, azimuth_img1, pose1,
		depth_img2, azimuth_img2, pose2,
		intrinsics
	)
	save_point_cloud(pointcloud, 'pointcloud.off')

	# pointcloud1 = depth_to_pointcloud(depth_img1, intrinsics)
	# save_point_cloud(pointcloud1, 'pointcloud1.off')

	# pointcloud2 = depth_to_pointcloud(depth_img2, intrinsics)
	# save_point_cloud(pointcloud2, 'pointcloud2.off')

	# save_point_cloud(pointcloud1 + pointcloud2, 'pointcloud_combined.off')

	# im_color = cv2.applyColorMap(np.uint8(np.float32(depth_img1) / max_depth1*255), cv2.COLORMAP_JET)
	# cv2.imshow('depth_img1', im_color)

	# im_color = cv2.applyColorMap(np.uint8(np.float32(depth_img2) / max_depth2*255), cv2.COLORMAP_JET)
	# cv2.imshow('depth_img2', im_color)

	# cv.waitKey()

	exit(0)

	# im_color = cv2.applyColorMap(np.uint8(azimuth_img/math.pi*255), cv2.COLORMAP_JET)
	# cv2.imshow('azimuth_img', im_color)
		
	
	import sys
	glutInit(sys.argv)
	gtk.gdk.threads_init()
	window = gtk.Window(gtk.WINDOW_TOPLEVEL)
	window.set_title("Zoom Pan Rotate")
	window.set_size_request(640,480)
	window.connect("destroy",lambda event: gtk.main_quit())
	vbox = gtk.VBox(False, 0)
	window.add(vbox)
	zpr = GLZPR()
	zpr.draw = functools.partial(draw_point_cloud, pointcloud)
	vbox.pack_start(zpr,True,True)
	window.show_all()
	gtk.main()
