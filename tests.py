import unittest
import numpy as np
import random
import math

from normal_reconstruction import triangulate_azimuth, transform
    
def project_vector_to_plane(vector, plane_normal):
    return vector - np.dot(vector.transpose(), plane_normal) * plane_normal

class TestNormalReconstruction(unittest.TestCase):

    def test_triangulation(self):
        pose1 = np.asarray([[ 0.9018748,   0.14004038, -0.40866922,  0.5060414],
                            [-0.13235333,  0.99007866,  0.04718941, -0.01872385],
                            [ 0.4112231,   0.01152979,  0.91146181,  0.06318428],
                            [ 0.,          0.,          0.,          1.        ]])

        pose2 = np.asarray([[ 0.98470966,  0.05970723, -0.16365187,  0.06009007],
                            [-0.05415116,  0.99780149,  0.03820789,  0.0727731 ],
                            [ 0.16557337, -0.02876174,  0.98577798, -0.23800894],
                            [ 0.,          0.,          0.,          1.        ]])

        pose2 = np.linalg.inv(pose1).dot(pose2)
        pose1 = np.eye(4)

        # the normal should be around the camera's z-axis
        normal = np.asarray([
            np.asarray([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]) +
            pose1[:3, 2]
        ]).reshape((3, 1))
        normal /= np.linalg.norm(normal)
        
        projection1 = project_vector_to_plane(normal, pose1[:3, 2:3])
        projection2 = project_vector_to_plane(normal, pose2[:3, 2:3])

        print 'world projection1: ', projection1.transpose()
        print 'world projection2: ', projection2.transpose()

        # the vector should be in their respective frame
        projection1 = pose1[:3, :3].transpose().dot(projection1)
        projection2 = pose2[:3, :3].transpose().dot(projection2)

        self.assertLess(math.fabs(projection1[2, 0]), 1e-6)
        self.assertLess(math.fabs(projection2[2, 0]), 1e-6)

        azimuth1 = math.atan2(projection1[1, 0], projection1[0, 0])
        azimuth2 = math.atan2(projection2[1, 0], projection2[0, 0])

        normal_eval = triangulate_azimuth(azimuth1, pose1, azimuth2, pose2)
        normal_eval = np.resize(normal_eval, (3, 1))

        print 'normal: ', normal.transpose()
        print 'normal eval: ', normal_eval.transpose()

        self.assertLess( np.linalg.norm(normal - normal_eval), 1e-6 )
        

if __name__ == '__main__':
    unittest.main()