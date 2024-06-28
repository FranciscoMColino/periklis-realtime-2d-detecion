import numpy as np

import sensor_msgs_py.point_cloud2 as pc2

from ember_detection_interfaces.msg import EmberBoundingBox3D, EmberBoundingBox3DArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, String, UInt32

def compute_points_from_bbox(bbox, resize_factor, fx, fy, cx, cy, depth_image):
        u1, v1, u2, v2 = bbox // resize_factor

        # check v1, v2, u1, u2 bounds
        v1 = max(0, v1)
        v1 = min(depth_image.shape[0] - 1, v1)
        v2 = max(0, v2)
        v2 = min(depth_image.shape[0] - 1, v2)
        u1 = max(0, u1)
        u1 = min(depth_image.shape[1] - 1, u1)
        u2 = max(0, u2)
        u2 = min(depth_image.shape[1] - 1, u2)

        depth_values_bb = depth_image[v1:v2, u1:u2]
        depth_values_bb = depth_values_bb[np.isfinite(depth_values_bb)]
        if len(depth_values_bb) == 0:
            print('No finite depth values in the bounding box')
            return None
        
        center_bbox_x = (u1 + u2) // 2
        center_bbox_y = (v1 + v2) // 2

        # find value closest to the center of the bounding box that is finite
        # ugly code, but works
        center_depth = None
        for i in range(1, 10):
            center_depth = depth_image[center_bbox_y - i, center_bbox_x]
            if np.isfinite(center_depth):
                break
            center_depth = depth_image[center_bbox_y + i, center_bbox_x]
            if np.isfinite(center_depth):
                break
            center_depth = depth_image[center_bbox_y, center_bbox_x - i]
            if np.isfinite(center_depth):
                break
            center_depth = depth_image[center_bbox_y, center_bbox_x + i]
            if np.isfinite(center_depth):
                break

        if center_depth is None:
            center_depth = np.min(depth_values_bb)

        main_depth = center_depth

        z1 = main_depth
        x1 = ((u1 - cx) * z1) / fx
        y1 = ((v1 - cy) * z1) / fy

        z_h = main_depth
        x_h = ((u2 - cx) * z_h) / fx
        #y_h = ((v2 - cy) * z_h) / fy

        z2 = main_depth + np.abs(x_h - x1)
        x2 = ((u2 - cx) * z2) / fx
        y2 = ((v2 - cy) * z2) / fy

        x1, y1, z1 = z1, -x1, -y1
        x2, y2, z2 = z2, -x2, -y2

        points = np.array([
                    [x1, y1, z1],
                    [x1, y1, z2],
                    [x1, y2, z1],
                    [x1, y2, z2],
                    [x2, y1, z1],
                    [x2, y1, z2],
                    [x2, y2, z1],
                    [x2, y2, z2]
                ])

        return points

def compute_points_from_bbox_2(bbox, resize_factor, fx, fy, cx, cy, depth_image):
    u1, v1, u2, v2 = bbox // resize_factor

    # check v1, v2, u1, u2 bounds
    v1 = max(0, v1)
    v1 = min(depth_image.shape[0] - 1, v1)
    v2 = max(0, v2)
    v2 = min(depth_image.shape[0] - 1, v2)
    u1 = max(0, u1)
    u1 = min(depth_image.shape[1] - 1, u1)
    u2 = max(0, u2)
    u2 = min(depth_image.shape[1] - 1, u2)

    depth_values_bb = depth_image[v1:v2, u1:u2]
    depth_values_bb = depth_values_bb[np.isfinite(depth_values_bb)]
    if len(depth_values_bb) == 0:
        print('No finite depth values in the bounding box')
        return None
    
    center_bbox_x = (u1 + u2) // 2
    center_bbox_y = (v1 + v2) // 2

    # find value closest to the center of the bounding box that is finite
    # ugly code, but works
    center_depth = None
    for i in range(1, 10):
        center_depth = depth_image[center_bbox_y - i, center_bbox_x]
        if np.isfinite(center_depth):
            break
        center_depth = depth_image[center_bbox_y + i, center_bbox_x]
        if np.isfinite(center_depth):
            break
        center_depth = depth_image[center_bbox_y, center_bbox_x - i]
        if np.isfinite(center_depth):
            break
        center_depth = depth_image[center_bbox_y, center_bbox_x + i]
        if np.isfinite(center_depth):
            break

    if center_depth is None:
        center_depth = np.min(depth_values_bb)

    main_depth = center_depth

    z1 = main_depth
    x1 = ((u1 - cx) * z1) / fx
    y1 = ((v1 - cy) * z1) / fy

    z2 = main_depth
    x2 = ((u2 - cx) * z2) / fx
    y2 = ((v2 - cy) * z2) / fy

    x1, y1, z1 = z1, -x1, -y1
    x2, y2, z2 = z2, -x2, -y2

    parent_point_1 = np.array([x1, y1, z1])
    parent_point_2 = np.array([x2, y2, z1])

    center = (parent_point_1 + parent_point_2) / 2

    translated_parent_point_1 = parent_point_1 - center
    translated_parent_point_2 = parent_point_2 - center
    
    # calculate the rotation angle to x axis
    theta = np.arctan2(translated_parent_point_1[1] - translated_parent_point_2[1], translated_parent_point_1[0] - translated_parent_point_2[0])
    phi = np.arctan2(translated_parent_point_1[2] - translated_parent_point_2[2], translated_parent_point_1[0] - translated_parent_point_2[0])

    rotated_parent_point_1 = rotate_point_3d(translated_parent_point_1, theta, phi)
    rotated_parent_point_2 = rotate_point_3d(translated_parent_point_2, theta, phi)

    translated_parent_point_1 = rotated_parent_point_1 + center
    translated_parent_point_2 = rotated_parent_point_2 + center

    rpp1_x, rpp1_y, rpp1_z = translated_parent_point_1
    rpp2_x, rpp2_y, rpp2_z = translated_parent_point_2

    dist_parents = np.linalg.norm(rotated_parent_point_1 - rotated_parent_point_2)

    points = np.array([
        [rpp1_x + dist_parents/2, rpp1_y, z1,],
        [rpp1_x - dist_parents/2, rpp1_y, z1,],
        [rpp2_x + dist_parents/2, rpp2_y, z1,],
        [rpp2_x - dist_parents/2, rpp2_y, z1,],
        [rpp1_x + dist_parents/2, rpp1_y, z2,],
        [rpp1_x - dist_parents/2, rpp1_y, z2,],
        [rpp2_x + dist_parents/2, rpp2_y, z2,],
        [rpp2_x - dist_parents/2, rpp2_y, z2]
    ])

    return points

"""
Compute the parent points from a bounding box
Parent points are the 2 points that define the 2D bounding box in 3D space where the line between them intersects the object
These parent points should be used to calculate the 3D bounding box
"""
def compute_parents_from_bbox(bbox, resize_factor, fx, fy, cx, cy, depth_image):

    u1, v1, u2, v2 = bbox // resize_factor

    # check v1, v2, u1, u2 bounds
    v1 = max(0, v1)
    v1 = min(depth_image.shape[0] - 1, v1)
    v2 = max(0, v2)
    v2 = min(depth_image.shape[0] - 1, v2)
    u1 = max(0, u1)
    u1 = min(depth_image.shape[1] - 1, u1)
    u2 = max(0, u2)
    u2 = min(depth_image.shape[1] - 1, u2)

    depth_values_bb = depth_image[v1:v2, u1:u2]
    depth_values_bb = depth_values_bb[np.isfinite(depth_values_bb)]
    if len(depth_values_bb) == 0:
        print('No finite depth values in the bounding box')
        return None
    
    center_bbox_x = (u1 + u2) // 2
    center_bbox_y = (v1 + v2) // 2

    # find value closest to the center of the bounding box that is finite
    # ugly code, but works
    center_depth = None
    for i in range(1, 10):
        center_depth = depth_image[center_bbox_y - i, center_bbox_x]
        if np.isfinite(center_depth):
            break
        center_depth = depth_image[center_bbox_y + i, center_bbox_x]
        if np.isfinite(center_depth):
            break
        center_depth = depth_image[center_bbox_y, center_bbox_x - i]
        if np.isfinite(center_depth):
            break
        center_depth = depth_image[center_bbox_y, center_bbox_x + i]
        if np.isfinite(center_depth):
            break

    if center_depth is None:
        center_depth = np.min(depth_values_bb)

    main_depth = center_depth

    z1 = main_depth
    x1 = ((u1 - cx) * z1) / fx
    y1 = ((v1 - cy) * z1) / fy

    z2 = main_depth
    x2 = ((u2 - cx) * z2) / fx
    y2 = ((v2 - cy) * z2) / fy

    x1, y1, z1 = z1, -x1, -y1
    x2, y2, z2 = z2, -x2, -y2

    parent_point_1 = np.array([x1, y1, z1])
    parent_point_2 = np.array([x2, y2, z2])

    return parent_point_1, parent_point_2

def compute_3d_bbox_from_parents(parent_point_1, parent_point_2):

    z1, z2 = parent_point_1[2], parent_point_2[2]

    parent_point_1[2] = 0 # set z to 0
    parent_point_2[2] = 0 # set z to 0

    center = (parent_point_1 + parent_point_2) / 2
    
    translated_parent_point_1 = parent_point_1 - center
    translated_parent_point_2 = parent_point_2 - center

    theta = np.arctan2(translated_parent_point_1[1] - translated_parent_point_2[1], translated_parent_point_1[0] - translated_parent_point_2[0])
    phi = np.arctan2(translated_parent_point_1[2] - translated_parent_point_2[2], translated_parent_point_1[0] - translated_parent_point_2[0])

    rotated_parent_point_1 = rotate_point_3d(translated_parent_point_1, theta, phi)
    rotated_parent_point_2 = rotate_point_3d(translated_parent_point_2, theta, phi)

    translated_parent_point_1 = rotated_parent_point_1 + center
    translated_parent_point_2 = rotated_parent_point_2 + center

    rpp1_x, rpp1_y, rpp1_z = translated_parent_point_1
    rpp2_x, rpp2_y, rpp2_z = translated_parent_point_2

    dist_parents = np.linalg.norm(rotated_parent_point_1 - rotated_parent_point_2)

    points = np.array([
        [rpp1_x + dist_parents/2, rpp1_y, z1,],
        [rpp1_x - dist_parents/2, rpp1_y, z1,],
        [rpp2_x + dist_parents/2, rpp2_y, z1,],
        [rpp2_x - dist_parents/2, rpp2_y, z1,],
        [rpp1_x + dist_parents/2, rpp1_y, z2,],
        [rpp1_x - dist_parents/2, rpp1_y, z2,],
        [rpp2_x + dist_parents/2, rpp2_y, z2,],
        [rpp2_x - dist_parents/2, rpp2_y, z2]
    ])

    return points

def compute_3d_bbox_from_parents_2(parent_point_1, parent_point_2):

    z1, z2 = parent_point_1[2], parent_point_2[2]

    parent_point_1[2] = 0 # set z to 0
    parent_point_2[2] = 0 # set z to 0

    center = (parent_point_1 + parent_point_2) / 2
    
    dist_parents = np.linalg.norm(parent_point_1 - parent_point_2)

    points = np.array([
        [center[0] + dist_parents/2, center[1], z1,],
        [center[0] - dist_parents/2, center[1], z1,],
        [center[0] + dist_parents/2, center[1], z2,],
        [center[0] - dist_parents/2, center[1], z2,],
        [center[0], center[1] + dist_parents/2, z1,],
        [center[0], center[1] - dist_parents/2, z1,],
        [center[0], center[1] + dist_parents/2, z2,],
        [center[0], center[1] - dist_parents/2, z2]
    ])

    return points

def rotate_point_2d(point, theta):
    x, y = point

    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    point = np.array([x, y])
    rotated_point = R.dot(point)
    return rotated_point

def rotate_point_3d(point, theta, phi):
    x, y, z = point
    
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    point_rotated = np.dot(R_y, np.dot(R_x, np.array([x, y, z])))
    return point_rotated

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    """
    x, y, z, w = q
    R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                  [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]])
    return R

def pose_msg_to_transform_matrix(msg):
    pose = msg.pose
    translation = np.array([pose.position.x,
                            pose.position.y,
                            pose.position.z])
    quaternion = [pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w]
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix

def apply_transformation(point, transformation_matrix):
    """
    Applies a transformation matrix to a 3D point.
    Args:
    - point: array or list with 3 elements (x, y, z)
    - transformation_matrix: 4x4 transformation matrix
    Returns:
    - Transformed 3D point as an array with 3 elements
    """
    point_h = np.append(point, 1)  # Convert to homogeneous coordinates
    transformed_point_h = transformation_matrix @ point_h  # Matrix multiplication
    return transformed_point_h[:3]  # Convert back to 3D coordinates

"""
Convert a PointCloud2 message to a numpy array, compatible with Open3D.
"""
def pc2_msg_to_numpy(msg):
    pc2_points = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
    pc2_points_64 = pc2_points.astype(np.float64)
    valid_idx = np.isfinite(pc2_points_64[:, 0]) & np.isfinite(pc2_points_64[:, 1]) & np.isfinite(pc2_points_64[:, 2])
    pc2_points_64 = pc2_points_64[valid_idx]
    return pc2_points_64

def build_ember_bbox(points):
        ember_bbox = EmberBoundingBox3D()
        for point in points:
            ember_bbox.points.append(Point(x=point[0], y=point[1], z=point[2]))
        ember_bbox.det_label = String(data='person')
        ember_bbox.points_count = UInt32(data=len(points))
        return ember_bbox