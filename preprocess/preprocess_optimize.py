import numpy as np
import random
import math


def rotate_scene_smplx_predefine(scene_verts, rot_angle):  # rotate around z axis before cropping scene cube
    scene_verts_aug = np.zeros(scene_verts.shape)
    scene_verts_aug[:, 0] = scene_verts[:, 0] * math.cos(rot_angle) - scene_verts[:, 1] * math.sin(rot_angle)
    scene_verts_aug[:, 1] = scene_verts[:, 0] * math.sin(rot_angle) + scene_verts[:, 1] * math.cos(rot_angle)
    scene_verts_aug[:, 2] = scene_verts[:, 2]
    return scene_verts_aug


def crop_scene_cube_smplx_predifine(scene_verts, r=2.0, with_wall_ceilling=True, random_seed=None,
                                    scene_min_x=None, scene_max_x=None, scene_min_y=None, scene_max_y=None, rotate=False):
    scene_center = np.array([0.0, 0.0, 0.0])
    random.seed(random_seed)
    scene_center[0] = random.uniform(scene_min_x + r/2, scene_max_x - r/2)
    scene_center[1] = random.uniform(scene_min_y + r/2, scene_max_y - r/2)

    min_x = scene_center[0] - r/2
    max_x = scene_center[0] + r/2
    min_y = scene_center[1] - r/2
    max_y = scene_center[1] + r/2

    # cropped scene verts point cloud
    if not rotate:
        scene_verts_crop = scene_verts[np.where((scene_verts[:,0] >= min_x) & (scene_verts[:,0] <= max_x) &
                                                (scene_verts[:,1] >= min_y) & (scene_verts[:,1] <= max_y))]
    else:
        rot_angle = random.uniform(0, 2*(math.pi))
        # P(x1,y1), rotate theta around Q(x0,y0) --> (x,y)
        # x = (x1 - x0) * cos(theta) - (y1 - y0) * sin(theta) + x0
        # y = (x1 - x0) * sin(theta) + (y1 - y0) * cos(theta) + y0
        x = (scene_verts[:, 0] - scene_center[0]) * math.cos(-rot_angle) - (scene_verts[:, 1] - scene_center[1]) * math.sin(-rot_angle) + scene_center[0]
        y = (scene_verts[:, 0] - scene_center[0]) * math.sin(-rot_angle) + (scene_verts[:, 1] - scene_center[1]) * math.cos(-rot_angle) + scene_center[1]
        scene_verts_crop = scene_verts[np.where((x >= min_x) & (x <= max_x) &
                                                (y >= min_y) & (y <= max_y))]

    scene_center[2] = np.min(scene_verts[:, 2]) + 1.0  # fix dist from origin to floor

    scene_verts_crop = scene_verts_crop - scene_center
    scene_verts_crop = scene_verts_crop[np.where(scene_verts_crop[:, 2] <= 1.0)] # remove points higher than virtual veiling
    scene_verts = scene_verts - scene_center

    if with_wall_ceilling:
        # add ceiling/walls to scene_verts_crop
        n_pts_edge = 70
        grid = (max_x - min_x) / n_pts_edge
        ceiling_points, wall1_points, wall2_points, wall3_points, wall4_points = [], [], [], [], []
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                x = min_x + (i + 1) * grid - scene_center[0]
                y = min_y + (j + 1) * grid - scene_center[1]
                ceiling_points.append(np.array([x, y, 1.0]))  # ceiling hight: 1m from scene_center(origin)
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                x = min_x + (i + 1) * grid - scene_center[0]
                z = -1.0 + (j + 1) * grid
                wall1_points.append(np.array([x, min_y - scene_center[1], z]))
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                x = min_x + (i + 1) * grid - scene_center[0]
                z = -1.0 + (j + 1) * grid
                wall2_points.append(np.array([x, max_y - scene_center[1], z]))
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                y = min_y + (i + 1) * grid - scene_center[1]
                z = -1.0 + (j + 1) * grid
                wall3_points.append(np.array([min_x - scene_center[0], y, z]))
        for i in range(n_pts_edge):
            for j in range(n_pts_edge):
                y = min_y + (i + 1) * grid - scene_center[1]
                z = -1.0 + (j + 1) * grid
                wall4_points.append(np.array([max_x - scene_center[0], y, z]))
        ceiling_points = np.asarray(ceiling_points)  # [n_ceiling_pts, 3]
        wall1_points = np.asarray(wall1_points)
        wall2_points = np.asarray(wall2_points)
        wall3_points = np.asarray(wall3_points)
        wall4_points = np.asarray(wall4_points)

        if not rotate:
            scene_verts_crop = np.concatenate((scene_verts_crop, ceiling_points,
                                               wall1_points, wall2_points, wall3_points, wall4_points), axis=0)
        if rotate:
            ceiling_points_rotate = np.zeros(ceiling_points.shape)
            ceiling_points_rotate[:, 0] = ceiling_points[:, 0] * math.cos(rot_angle) - ceiling_points[:, 1] * math.sin(rot_angle)
            ceiling_points_rotate[:, 1] = ceiling_points[:, 0] * math.sin(rot_angle) + ceiling_points[:, 1] * math.cos(rot_angle)
            ceiling_points_rotate[:, 2] = ceiling_points[:, 2]

            wall1_points_rotate = np.zeros(wall1_points.shape)
            wall1_points_rotate[:, 0] = wall1_points[:, 0] * math.cos(rot_angle) - wall1_points[:, 1] * math.sin(rot_angle)
            wall1_points_rotate[:, 1] = wall1_points[:, 0] * math.sin(rot_angle) + wall1_points[:, 1] * math.cos(rot_angle)
            wall1_points_rotate[:, 2] = wall1_points[:, 2]

            wall2_points_rotate = np.zeros(wall2_points.shape)
            wall2_points_rotate[:, 0] = wall2_points[:, 0] * math.cos(rot_angle) - wall2_points[:, 1] * math.sin(rot_angle)
            wall2_points_rotate[:, 1] = wall2_points[:, 0] * math.sin(rot_angle) + wall2_points[:, 1] * math.cos(rot_angle)
            wall2_points_rotate[:, 2] = wall2_points[:, 2]

            wall3_points_rotate = np.zeros(wall3_points.shape)
            wall3_points_rotate[:, 0] = wall3_points[:, 0] * math.cos(rot_angle) - wall3_points[:, 1] * math.sin(rot_angle)
            wall3_points_rotate[:, 1] = wall3_points[:, 0] * math.sin(rot_angle) + wall3_points[:, 1] * math.cos(rot_angle)
            wall3_points_rotate[:, 2] = wall3_points[:, 2]

            wall4_points_rotate = np.zeros(wall4_points.shape)
            wall4_points_rotate[:, 0] = wall4_points[:, 0] * math.cos(rot_angle) - wall4_points[:, 1] * math.sin(rot_angle)
            wall4_points_rotate[:, 1] = wall4_points[:, 0] * math.sin(rot_angle) + wall4_points[:, 1] * math.cos(rot_angle)
            wall4_points_rotate[:, 2] = wall4_points[:, 2]

            scene_verts_crop = np.concatenate((scene_verts_crop, ceiling_points_rotate,
                                               wall1_points_rotate, wall2_points_rotate, wall3_points_rotate, wall4_points_rotate), axis=0)

    shift = -scene_center
    return scene_verts, scene_verts_crop, shift   # shifted by scene_center



def augmentation_crop_scene_smplx(scene_verts, scene_verts_crop, random_seed):
    # random rotate around z-axis
    random.seed(random_seed)
    rot_angle = random.uniform(0, 2*(math.pi))
    scene_verts_global = np.zeros(scene_verts.shape)
    scene_verts_crop_global = np.zeros(scene_verts_crop.shape)

    scene_verts_global[:, 0] = scene_verts[:, 0] * math.cos(rot_angle) - scene_verts[:, 1] * math.sin(rot_angle)
    scene_verts_global[:, 1] = scene_verts[:, 0] * math.sin(rot_angle) + scene_verts[:, 1] * math.cos(rot_angle)
    scene_verts_global[:, 2] = scene_verts[:, 2]
    scene_verts_crop_global[:, 0] = scene_verts_crop[:, 0] * math.cos(rot_angle) - scene_verts_crop[:, 1] * math.sin(rot_angle)
    scene_verts_crop_global[:, 1] = scene_verts_crop[:, 0] * math.sin(rot_angle) + scene_verts_crop[:, 1] * math.cos(rot_angle)
    scene_verts_crop_global[:, 2] = scene_verts_crop[:, 2]

    return scene_verts_global, scene_verts_crop_global, rot_angle