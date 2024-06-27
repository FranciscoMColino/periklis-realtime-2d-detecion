import numpy as np

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