import unittest
import numpy as np
import random
import math

from normal_reconstruction import triangulate_azimuth

class TestNormalReconstruction(unittest.TestCase):

    def test_triangulation(self):
        pose1 = np.asarray([[ 0.901875,  -0.132353,   0.411223,   0.0471894],
                            [-0.484847,   0.14004,    0.990079,   0.911462 ],
                            [ 0.0115298, -0.0530567, -0.408669,   0.150097 ],
                            [ 0.,         0.,         0.,         1.       ]])

        pose2 = np.asarray([[ 0.98471,   -0.0541512,  0.165573,   0.0382079],
                            [-0.0158226,  0.0597072,  0.997801,   0.985778 ],
                            [-0.0287617, -0.0830465, -0.163652,   0.241677 ],
                            [ 0.,         0.,         0.,         1.       ]])

        # the normal should be around the camera's z-axis
        normal = np.asarray([
            np.asarray([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]) +
            pose1[:3, 2]
        ]).reshape((3, 1))
        normal /= np.linalg.norm(normal)
        
        projection1 = normal - np.dot(normal.transpose(), pose1[:3, 2:3])*pose1[:3, 2:3]
        projection2 = normal - np.dot(normal.transpose(), pose2[:3, 2:3])*pose2[:3, 2:3]

        # the vector should be in their respective frame
        projection1 = pose1[:3, :3].transpose().dot(projection1)
        projection2 = pose1[:3, :3].transpose().dot(projection2)

        print projection1, projection2
        
        

if __name__ == '__main__':
    unittest.main()