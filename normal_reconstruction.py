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
from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays, \
                                                  glBindVertexArray
from OpenGL.arrays import vbo

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
	T = np.resize(list_pose, (4, 4))
	return T

def transform(T, p):
	'''
	T: 4x4 array
	p: 3 tuple
	'''
	p_homogeneous = T.dot(np.resize(p+(1,), (4, 1)))
	return tuple(p_homogeneous[:3].transpose()[0])

def project_vector_to_plane(vector, plane_normal):
	vector = np.resize(vector, (3,))
	plane_normal = np.resize(plane_normal, (3,))
	return vector - np.dot(vector.transpose(), plane_normal) * plane_normal

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
	normals = []

	azimuth_img1_reprojected = np.zeros((height, width), dtype='f')
	azimuth_img2_reprojected = np.zeros((height, width), dtype='f')

	azimuth_img1_original = np.zeros((height, width), dtype='f')
	azimuth_img2_original = np.zeros((height, width), dtype='f')

	for row in range(0, depth_img1.shape[0]):
		for col in range(0, depth_img1.shape[1]):
			d1 = float(depth_img1[row, col])
			
			if (d1 < (max_depth * 0.1)):
				continue
			xyz1 = image_to_3d(col, row, d1, intrinsics)
			
			# project and find out if there is overlap
			xyz2 = transform(T_21, xyz1)
			u, v = threeD_to_image(xyz2, intrinsics)
			
			# print xyz2, xyz2_gt			
			if u < 0 or u >= width or v < 0 or v >= height:
				continue

			d2 = float(depth_img2[v, u])
			xyz2_gt = image_to_3d(u, v, d2, intrinsics)
			# in the first camera frame
			xyz2_gt = transform(T_12, xyz2_gt)
			
			if math.fabs(d2 - xyz2[2]) > max_depth * 0.1:
				continue

			pointcloud.append(xyz1)

			if row % 15 or col % 15:
				continue

			azimuth1 = azimuth_img1[row, col]
			azimuth2 = azimuth_img2[v, u]
			normal = triangulate_azimuth(azimuth1, np.eye(4), azimuth2, T_12)
			normal = np.resize(normal, (3,));

			normal_start = xyz1
			normal_end = np.array(xyz1) + normal*max_depth*0.05

			normals.append(normal_start)
			normals.append(normal_end)
			
			projection1 = (normal[0], normal[1], 0)
			azimuth_img1_reprojected[row, col] = math.fabs(math.atan2(projection1[1], projection1[0])-math.pi)
			azimuth_img1_original[row, col] = math.fabs(azimuth_img1[row, col]-math.pi)

			projection2 = project_vector_to_plane(normal, T_12[:3, 2:3])
			projection2 = T_21[:3, :3].dot(projection2)
			azimuth_img2_reprojected[v, u] = math.fabs(math.atan2(projection2[1], projection2[0])-math.pi)
			azimuth_img2_original[row, col] = math.fabs(azimuth_img2[row, col]-math.pi)
			
			num_overlap += 1

	# cv2.imwrite(
	# 	'original1.png', 
	# 	cv2.applyColorMap(np.uint8(azimuth_img1_original/math.pi*255), cv2.COLORMAP_JET)
	# )
	# cv2.imwrite(
	# 	'original2.png', 
	# 	cv2.applyColorMap(np.uint8(azimuth_img2_original/math.pi*255), cv2.COLORMAP_JET)
	# )
	# cv2.imwrite(
	# 	'reprojected1.png', 
	# 	cv2.applyColorMap(np.uint8(azimuth_img1_reprojected/math.pi*255), cv2.COLORMAP_JET)
	# )
	# cv2.imwrite(
	# 	'reprojected2.png', 
	# 	cv2.applyColorMap(np.uint8(azimuth_img2_reprojected/math.pi*255), cv2.COLORMAP_JET)
	# )

	print num_overlap
	return pointcloud, normals

def triangulate_azimuth(azimuth1, pose1, azimuth2, pose2):
	'''
	Calculates normal given azimuth and pose of two frames
	'''
	projection1 = np.resize([math.cos(azimuth1), math.sin(azimuth1), 0], (3,))
	projection2 = np.resize([math.cos(azimuth2), math.sin(azimuth2), 0], (3,))

	# transform to world frame
	projection1 = pose1[:3, :3].dot(projection1)
	projection2 = pose2[:3, :3].dot(projection2)

	# the z axis of image plane
	image_plane_normal1 = pose1[:3, 2]
	image_plane_normal2 = pose2[:3, 2]

	cross1 = np.cross(projection1, image_plane_normal1)
	cross2 = np.cross(projection2, image_plane_normal2)

	point_normal = np.cross(cross1, cross2)

	# make sure that the normal is in the right side
	if np.dot(point_normal, pose1[:3, 2]) > 0:
		point_normal *= -1

	return point_normal/np.linalg.norm(point_normal)
		
vertex_indices = None
vertex_positions = None

normal_indices = None
normal_positions = None

def draw_point_cloud(event):
	global vertex_indices, vertex_positions

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

	glPushMatrix() 
	glMultMatrixf(np.array([
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	], dtype='f'))

	vertex_indices.bind()
  	vertex_positions.bind()
  	glEnableVertexAttribArray(0) # from 'location = 0' in shader
  	glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)
  	glDrawElements(GL_POINTS, len(pointcloud), GL_UNSIGNED_INT, None)
  	glBindVertexArray(0)
  	vertex_positions.unbind()

  	normal_indices.bind()
  	normal_positions.bind()
  	glEnableVertexAttribArray(0)
	glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)
	glDrawElements(GL_LINES, len(normals), GL_UNSIGNED_INT, None)

	glPopMatrix()

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

	pointcloud, normals = compute_point_cloud_with_normal(
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

	# exit(0)

	# im_color = cv2.applyColorMap(np.uint8(azimuth_img/math.pi*255), cv2.COLORMAP_JET)
	# cv2.imshow('azimuth_img', im_color)

	vertex_positions = vbo.VBO(np.array(pointcloud, dtype='f'))
	vertex_indices = vbo.VBO(np.array([i for i in range(len(pointcloud))], dtype=np.int32), target=GL_ELEMENT_ARRAY_BUFFER)

	normal_positions = vbo.VBO(np.array(normals, dtype='f'))
	normal_indices = vbo.VBO(np.array([(i, i+1) for i in range(0, len(normals), 2)], dtype=np.int32), target=GL_ELEMENT_ARRAY_BUFFER)

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
	zpr.draw = draw_point_cloud
	vbox.pack_start(zpr,True,True)

	light_ambient = np.array([0.2, 0.2, 0.2, 1.0], dtype='f')
	glEnable(GL_LIGHTING);
	glEnable(GL_NORMALIZE)
	
	# glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, light_ambient);
	

	
	window.show_all()
	gtk.main()
