""" Module used to generate geometrical synthetic shapes """

import os
import cv2 as cv
import numpy as np
import math
import copy
# import time


from tqdm import tqdm
from pathlib import Path
from shapely.geometry import Polygon
from argparse import ArgumentParser

import event_simulator as es
# from datasets import event_simulator as es


random_state = np.random.RandomState(None)


def set_random_state(state):
    global random_state
    random_state = state


def get_random_color(num=1, background_color=0):
    """ Output a random scalar in grayscale with a least a small
        contrast with the background color """

    color = random_state.randint(256, size=num)
    mask = abs(color - background_color) < 50  # not enough contrast
    color[mask] = (color[mask] + 128) % 256

    return color


def get_random_speed(min_speed=0, max_speed=4, num=1):
    """ Output a random speed_x speed_y"""

    speed = random_state.randint(min_speed, max_speed, size=(num,2)) * \
            random_state.choice([1, -1], size=(num,2))

    speed += 1
    return speed.squeeze()


def get_random_position(size, num=1):
    """ Output random xy position """

    pos = random_state.uniform(0, 1, (num, 2))
    pos[:,0] *= size[0]
    pos[:,1] *= size[1]

    return pos.astype(int)


def get_random_thickness(min_thickness=10, max_thickness=15, num=1):
    """ Output random thickness """
    return random_state.randint(min_thickness, max_thickness, size=(num))


def get_random_radius(min_rad=30, max_rad=80, num=1):
    """ Output random radius """
    return random_state.randint(min_rad, max_rad, size=(num))


def get_random_length(min_len=100, max_len=300, num=1):
    """ Output random length """
    return random_state.randint(min_len, max_len, size=(num, 2)) * \
            random_state.choice([1, -1], size=(num, 2))


def get_random_rotation(min_rotation=0, max_rotation=0.5, num=1):
    """ Output a Rotation matrix between -5 deg and 5 deg"""

    uni = random_state.uniform(min_rotation, max_rotation, num) * \
            random_state.choice([1, -1], num)

    theta = np.radians(uni)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    R = np.transpose(R, (2, 0, 1))

    return R.squeeze()


def get_R(uni):
    theta = np.radians(uni)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    R = np.transpose(R, (2, 0, 1))
    R = R.squeeze()
    return R


def get_different_color(previous_colors, min_dist=50, max_count=20):
    """ Output a color that contrasts with the previous colors
    Parameters:
      previous_colors: np.array of the previous colors
      min_dist: the difference between the new color and
                the previous colors must be at least min_dist
      max_count: maximal number of iterations
    """
    color = random_state.randint(256)
    count = 0
    while np.any(np.abs(previous_colors - color) < min_dist) and count < max_count:
        count += 1
        color = random_state.randint(256)
    return color


def calculate_rigid_transformation_batch_lines(pts, center, speed, rotation):
    """ Output new Position of transformation """


    if len(pts) == 1:
        pts_rotation = np.matmul(pts - center, rotation) + center
    else:
        pts_rotation = [np.matmul(pts[i] - center[i], rotation[i]) for i in range(len(pts))] + center
    pts_translation = pts_rotation + speed

    return np.array(pts_translation)


def calculate_rigid_transformation_batch(pts, center, speed, rotation):
    """ Output new Position of transformation """
    pts_rotation = [np.matmul(pts[i] - center, rotation) for i in range(len(pts))] + center
    pts_translation = pts_rotation + speed
    pts = np.array(pts_translation)

    return pts

def calculate_rigid_transformation(pts, center, speed, rotation):
    """ Output new Position of transformation """
    return np.matmul(pts - center, rotation) + center + speed


def add_salt_and_pepper(img):
    """ Add salt and pepper noise to an image """
    noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv.randu(noise, 0, 255)
    black = noise < 30
    white = noise > 225
    img[white > 0] = 255
    img[black > 0] = 0
    cv.blur(img, (5, 5), img)
    return np.empty((0, 2), dtype=np.int)



def generate_random_shape(img_size, num_frames, bg_config):
    """ Generate one random shape 
    Parameters:
    img_size: size of the image
    num_frames: number of frames
    bg_config: background generation configuration
    """

    draw_func = [
        draw_lines,
        draw_polygon,
        draw_multiple_polygons,
        draw_ellipses,
        draw_star,
        draw_checkerboard,
        draw_stripes,
        draw_cube
    ]

    return random_state.choice(draw_func)(img_size, num_frames, bg_config)



def generate_background(size=(960, 1280), num_frames=11, nb_blobs=100, min_rad_ratio=0.01,
                        max_rad_ratio=0.04, min_kernel_size=200, max_kernel_size=300):
    """ Generate a customized background image
    Parameters:
      size: size of the image
      nb_blobs: number of circles to draw
      min_rad_ratio: the radius of blobs is at least min_rad_size * max(size)
      max_rad_ratio: the radius of blobs is at most max_rad_size * max(size)
      min_kernel_size: minimal size of the kernel
      max_kernel_size: maximal size of the kernel
    """
    # return [np.zeros(size, dtype=np.uint8) for i in range(num_frames)]

    dim = max(size)
    canvas_size = size

    blob_size_ratio = random_state.uniform(0, 1) + 0.1
    min_rad_ratio *= blob_size_ratio
    max_rad_ratio *= blob_size_ratio

    img = np.zeros(canvas_size, dtype=np.uint8)
    cv.randu(img, 0, 255)
    # cv.threshold(img, random_state.randint(256), 255, cv.THRESH_BINARY, img)
    background_color = int(np.mean(img))

    speed = get_random_speed(num=nb_blobs, min_speed=1, max_speed=2)
    blobs = get_random_position(canvas_size, nb_blobs)
    colors = get_random_color(nb_blobs, background_color)
    radius = get_random_radius(int(dim*min_rad_ratio), int(dim*max_rad_ratio), nb_blobs)
    kernel_size = random_state.randint(min_kernel_size, max_kernel_size)


    frames = []
    for j in range(num_frames):
        tmp = img.copy()
        for i in range(nb_blobs):
            cv.circle(tmp, (blobs[i,1]+j*speed[i,1], blobs[i,0]+j*speed[i,0]), radius[i], int(colors[i]), -1)
            
        cv.blur(tmp, (kernel_size, kernel_size), tmp)
        # # cv.imwrite(f"bg{j}.png", tmp*255)
        frames.append(tmp)


    # [cv.circle(img, (blobs[i,1], blobs[i,0]), radius[i], int(colors[i]), -1) for i in range(nb_blobs)]
    # cv.blur(img, (kernel_size, kernel_size), img)

    # frames = np.array([img[start_y[i]:end_y[i], start_x[i]:end_x[i]] for i in range(num_frames)])
    return frames


def final_blur(img, kernel_size=(5, 5)):
    """ Apply a final Gaussian blur to the image
    Parameters:
      kernel_size: size of the kernel
    """
    cv.GaussianBlur(img, kernel_size, 0, img)


def ccw(A, B, C):
    """ Check if the points are listed in counter-clockwise order """
    return((C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0])
            > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))


def intersect(A, B, C, D):
    """ Return true if line segments AB and CD intersect """
    return np.any((ccw(A, C, D) != ccw(B, C, D)) &
                  (ccw(A, B, C) != ccw(A, B, D)))


# def ccw(A, B, C, dim):
#     """ Check if the points are listed in counter-clockwise order """
#     if dim == 2:  # only 2 dimensions
#         return((C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0])
#                > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))
#     else:  # dim should be equal to 3
#         return((C[:, 1, :] - A[:, 1, :])
#                * (B[:, 0, :] - A[:, 0, :])
#                > (B[:, 1, :] - A[:, 1, :])
#                * (C[:, 0, :] - A[:, 0, :]))


# def intersect(A, B, C, D, dim):
#     """ Return true if line segments AB and CD intersect """
#     return np.any((ccw(A, C, D, dim) != ccw(B, C, D, dim)) &
#                   (ccw(A, B, C, dim) != ccw(A, B, D, dim)))


def keep_points_inside(points, size):
    """ Keep only the points whose coordinates are inside the dimensions of
    the image of size 'size' """

    mask = (points[:,0] >= 0) & (points[:,0] < size[1]) &\
           (points[:,1] >= 0) & (points[:,1] < size[0])

    return points[mask]


def  get_next_position(points, center, speed, rotation):
    new_p = [np.matmul(points[i] - center[i], rotation[i]) for i in range(num_lines)]
    new_p = np.array(new_p + center + speed, dtype=int)


def draw_lines(img_size, num_frames, bg_config, nb_lines=10):
    """ Draw random lines and output the positions of the endpoints
    Parameters:
      nb_lines: maximal number of lines
    """ 


    images_ = generate_background(img_size, num_frames=num_frames)
    background_color = int(np.mean(images_))

    num_lines = random_state.randint(1, nb_lines)
    p1 = get_random_position(img_size, num_lines)
    p2 = p1 + get_random_length(num=num_lines)


    out_p1 = p1[np.newaxis, :, :]
    out_p2 = p2[np.newaxis, :, :]


    # Generate Speed and Rotation for animation
    thickness = get_random_thickness(num=num_lines)
    rotation = get_random_rotation(max_rotation=0.2, num=num_lines)
    speed = get_random_speed(min_speed=10, max_speed=20, num=num_lines) / 10.0
    colors = get_random_color(num=num_lines, background_color=background_color)


    # Generate Frames and Remove collided lines
    for i in range(num_frames):

        center = (out_p1[i] + out_p2[i]) / 2

        # print(out_p1[i], out_p2[i])
        p1 = calculate_rigid_transformation_batch_lines(out_p1[i], center, speed, rotation)
        p2 = calculate_rigid_transformation_batch_lines(out_p2[i], center, speed, rotation)

        valid_idx = [0]
        for j in range(1, num_lines):
            if not intersect(p1[valid_idx], p2[valid_idx], np.array([p1[j]]), np.array([p2[j]])):
                valid_idx.append(j)

        # print(out_p1.shape, p1.shape)
        out_p1 = np.vstack((out_p1[:,valid_idx,:], p1[np.newaxis,valid_idx,:]))
        out_p2 = np.vstack((out_p2[:,valid_idx,:], p2[np.newaxis,valid_idx,:]))


        if len(valid_idx) == 1:
            return draw_lines(img_size, num_frames, bg_config, nb_lines)

        speed = speed[valid_idx]
        rotation = rotation[valid_idx]
        num_lines = len(valid_idx)


        

    images = []
    # Draw lines on background images
    for i in range(num_frames):

        # image = np.zeros(img_size, dtype=np.uint8)
        image = images_[i]

        for j in range(num_lines):
            cv.line(image, tuple(out_p1[i,j].astype(int)), tuple(out_p2[i,j].astype(int)), int(colors[j]), thickness[j])
        
        # image = cv.resize(image, (160, 120), interpolation=cv.INTER_LINEAR,)
        images.append(image)

    points = np.hstack((out_p1, out_p2))  
    # event_sim = es.Event_simulator(images[0], 0)
    # events = np.array([event_sim.simulate(img, 0) for img in images[1:]]) * 255

    return images, points

    


def generate_polygons(img_size, max_rad_ratio=0.25, max_sides=8):
    
    min_dim = min(img_size[0], img_size[1])
    num_corners = random_state.randint(3, max_sides)
    rad = np.maximum(random_state.rand()*min_dim/2, min_dim*max_rad_ratio)

    cx = random_state.randint(rad, img_size[1] - rad) # Center of a circle
    cy = random_state.randint(rad, img_size[0] - rad)


    # Sample num_corners pooints inside the circle
    slices = np.linspace(0, 2 * np.pi, num_corners + 1)
    angles = slices[:-1] + random_state.rand(num_corners) * (slices[1:] - slices[:-1])

    x = cx + 0.8 * rad * np.cos(angles)
    y = cy + 0.8 * rad * np.sin(angles)
    points = np.dstack([x, y]).squeeze()


    # Filter the points that are too close or that have an angle too flat
    p_minus = np.roll(points, -1, axis=0)
    norms = np.linalg.norm(p_minus - points, axis=1) # short distance
    points = points[norms > 10]

    p_minus = np.roll(points, -1, axis=0)
    p_plus = np.roll(points, 1, axis=0)
    corner_angles = angle_between_vectors(p_minus - points,  p_plus-points) # acute angles
    points = (points[corner_angles < (2 * math.pi / 3)])
    points = points.squeeze()

    if len(points) < 3:
        return generate_polygons(img_size, max_rad_ratio, max_sides)

    return points

def draw_polygon(img_size, num_frames, bg_config, max_sides=8):
    """ Draw a polygon with a random number of corners
    and return the corner points
    Parameters:
      max_sides: maximal number of sides + 1
    """

    images_ = generate_background(img_size, num_frames=num_frames)
    background_color = int(np.mean(images_))
    points = generate_polygons((img_size[0]+30, img_size[1]+40)) + [15, 20]
    num_lines = len(points)


    # Generate Speed and Rotation for animation
    # rotation = get_random_rotation()
    # speed = get_random_speed()
    speed = get_random_speed(min_speed=10, max_speed=20) / 10.0
    uni = random_state.uniform(0.03, 0.08, 1) * random_state.choice([1, -1], 1)
    rotation = get_R(uni)
    decrease = True
    color = get_random_color(background_color=background_color)


    # Generate animated scences 
    frame_points = []
    images = []
    for i in range(num_frames):
        # image = np.zeros(img_size, dtype=np.uint8)
        image = images_[i]

        center = np.average(points, axis=0)
        points = calculate_rigid_transformation(points, center, speed, rotation)

        if decrease:
            speed /= 1.05
            uni /= 1.05
            rotation = get_R(uni)

            if (speed < 0.5).all():
                speed = get_random_speed(max_speed=20) / 50.0
                uni = random_state.uniform(0.01, 0.05, 1) * \
                            random_state.choice([1, -1], 1)
                decrease = False
        else:
            speed *= 1.1
            uni *= 1.1
            speed = np.clip(speed, -2, 2)
            uni = np.clip(uni, -0.05, 0.05)
            rotation = get_R(uni)

        corners = points.reshape((-1, 1, 2))
        cv.fillPoly(image, np.array([corners], dtype=np.int32), int(color))
        frame_points.append(points)
        images.append(image)


    points = np.array(frame_points)
    # event_sim = es.Event_simulator(images[0], 0)
    # events = np.array([event_sim.simulate(img, 0) for img in images[1:]])

    return images, points


def overlap(center, rad, centers, rads):
    """ Check that the circle with (center, rad)
    doesn't overlap with the other circles """
    flag = False
    for i in range(len(rads)):
        if np.linalg.norm(center - centers[i]) < rad + rads[i]:
            flag = True
            break
    return flag


def angle_between_vectors(v1, v2):
    """ Compute the angle (in rad) between the two vectors v1 and v2. """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)


    v_dot = [np.dot(v, u) for v, u in zip(v1_u, v2_u)]
    return np.arccos(np.clip(v_dot, -1.0, 1.0))


def draw_multiple_polygons(img_size, num_frames, bg_config, max_sides=8, nb_polygons=15, **extra):
    """ Draw multiple polygons with a random number of corners
    and return the corner points
    Parameters:
      max_sides: maximal number of sides + 1
      nb_polygons: maximal number of polygons
    """


    images_ = generate_background(img_size, num_frames=num_frames)
    background_color = int(np.mean(images_))
    polygons = np.array([generate_polygons(img_size, max_rad_ratio=0.1) for _ in range(nb_polygons)])


    # Generate Speed and Rotation for animation
    rotation = get_random_rotation(num=nb_polygons)
    speed = get_random_speed(min_speed=10, max_speed=20, num=nb_polygons) / 10.0
    color = get_random_color(num=nb_polygons, background_color=background_color)


    
    valid_idx = np.arange(nb_polygons)

    # Generate Frames and Remove collided lines
    polygon_shape = np.array([Polygon(poly) for poly in polygons])
    polygons_stack = np.array([polygon_shape    ])
    for i in range(num_frames):


        for j, poly in enumerate(polygon_shape):
            points = poly.exterior.coords
            center = np.average(points, axis=0)
            polygon_shape[j] = Polygon(calculate_rigid_transformation_batch(points, center, speed[j], rotation[j]))


        valid_idx = [0]
        for j, poly in enumerate(polygon_shape[1:]):


            is_collision = False
            for idx in valid_idx:

                if poly.intersects(polygon_shape[idx]):
                    is_collision = True
                    break

            if not is_collision:
                valid_idx.append(j+1)


        # print(polygon_shape.shape)
        polygon_shape = polygon_shape[valid_idx]
        polygons_stack = polygons_stack[:, valid_idx]

        # print(polygons.shape, polygons_stack.shape)
        polygons_stack = np.vstack((polygons_stack, polygon_shape))
        rotation = rotation[valid_idx]
        speed = speed[valid_idx]




    # Get Points, Images, Events
    frame_points_stack = []
    images = []

    # print()
    # print(polygons_stack)
    for i in range(num_frames):

        # img = np.zeros(img_size, dtype=np.uint8)
        img = images_[i]

        polygons = polygons_stack[i]
        # print(polygons.shape)
        frame_points = np.empty((0, 2), dtype=np.int)
        for j, poly in enumerate(polygons):
            x, y = poly.exterior.coords.xy

            points = np.dstack((x, y)).squeeze()
            corners = points.reshape((-1, 1, 2)).astype(int)
            cv.fillPoly(img, [corners], int(color[j]))
            frame_points = np.concatenate((frame_points, corners.squeeze()))

        frame_points_stack.append(frame_points)
        images.append(img)


    points = np.array(frame_points_stack)
    # event_sim = es.Event_simulator(images[0], 0)
    # events = np.array([event_sim.simulate(img, 0) for img in images[1:]])

    return images, points




def draw_ellipses(img_size, num_frames, bg_config, nb_ellipses=20):
    """ Draw several ellipses
    Parameters:
      nb_ellipses: maximal number of ellipses
    """



    images_ = generate_background(img_size, num_frames=num_frames)
    background_color = int(np.mean(images_))
    

    centers = np.empty((0, 2), dtype=np.int)
    max_rads = np.empty((0, 1), dtype=np.int)
    rads = np.empty((0, 2), dtype=np.int)
    min_dim = min(img_size[0], img_size[1]) / 4
    
    for i in range(nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img_size[1] - max_rad)  # center
        y = random_state.randint(max_rad, img_size[0] - max_rad)
        new_center = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        diff = centers - new_center
        if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - max_rads)):
            continue
        centers = np.concatenate([centers, new_center], axis=0)
        max_rads = np.concatenate([max_rads, np.array([[max_rad]])], axis=0)
        rads = np.concatenate([rads, np.array([[ax, ay]])], axis=0)


    colors = [get_random_color(background_color=background_color) for _ in range(len(centers))]
    angles = [random_state.rand()*90 for _ in range(len(centers))]
    rotation = [random_state.rand()*10-5 for _ in range(len(centers))]
    speed = np.array([get_random_speed() for _ in range(len(centers))])


    img_list = []
    for i in range(num_frames):

        # img = np.zeros(img_size, dtype=np.uint8)
        img = images_[i]

        for j, (center, rad, angle, R, S, color) in enumerate(zip(centers, rads, angles, rotation, speed, colors)):

            new_center = center + S
            angles[j] = angle + R

            # Check that the ellipsis will not overlap with pre-existing 
            temp_max_rads = max_rads[np.arange(len(max_rads))!=j]
            temp_centers = centers[np.arange(len(centers))!=j]

            diff = temp_centers - new_center
            if not np.any(max(rad) > (np.sqrt(np.sum(diff * diff, axis=1)) - temp_max_rads)):
                centers[j] = new_center


            center = (center[0], center[1])
            axes = (rad[0], rad[1])


            cv.ellipse(img, center, axes, angle + R, 0 , 360, int(color), -1)

        img_list.append(img)


        points = np.array([np.empty((0, 2), dtype=np.int) for _ in range(num_frames)])
        images = np.array(img_list)

        # event_sim = es.Event_simulator(images[0], 0)
        # events = np.array([event_sim.simulate(img, 0) for img in images[1:]])


    return images, points


def draw_star(img_size, num_frames, bg_config, nb_branches=6):
    """ Draw a star and output the interest points
    Parameters:
      nb_branches: number of branches of the star
    """

    images_ = generate_background(img_size, num_frames=num_frames)
    background_color = int(np.mean(images_))

    num_branches = random_state.randint(3, nb_branches)
    min_dim = min(img_size[0], img_size[1])
    thickness = random_state.randint(min_dim * 0.01, min_dim * 0.02)
    rad = max(random_state.rand() * min_dim / 2, min_dim / 5)
    x = random_state.randint(rad, img_size[1] - rad)  # select the center of a circle
    y = random_state.randint(rad, img_size[0] - rad)
    # Sample num_branches points inside the circle
    slices = np.linspace(0, 2 * math.pi, num_branches + 1)
    angles = [slices[j] + random_state.rand() * (slices[j+1] - slices[j])
              for j in range(num_branches)]
    points = np.array([[int(x + max(random_state.rand(), 0.3) * rad * math.cos(a)),
                        int(y + max(random_state.rand(), 0.3) * rad * math.sin(a))]
                       for a in angles])
    points = np.concatenate(([[x, y]], points), axis=0)
    color = get_random_color(num_branches, background_color=background_color)


    speed = get_random_speed(min_speed=10, max_speed=20) / 10.0
    uni = random_state.uniform(0.01, 0.05, 1) * random_state.choice([1, -1], 1)
    rotation = get_R(uni)
    decrease = True


    pts_list = []
    img_list = []
    for i in range(num_frames):

        # img = np.zeros(img_size, dtype=np.uint8)
        img = images_[i]
        pts = np.empty((0, 2), dtype=np.int)

        center = (points[0][0], points[0][1])
        points = (np.matmul(points - center, rotation) + center + speed).astype(int)

        if decrease:
            speed /= 1.05
            uni /= 1.05
            rotation = get_R(uni)

            if (speed < 0.8).all():
                speed = get_random_speed(max_speed=20) / 50.0
                uni = random_state.uniform(0.01, 0.05, 1) * \
                            random_state.choice([1, -1], 1)
                decrease = False
        else:
            speed *= 1.1
            uni *= 1.1
            speed = np.clip(speed, -2, 2)
            uni = np.clip(uni, -0.05, 0.05)
            rotation = get_R(uni)



        for j in range(1, num_branches + 1):
            cv.line(img, (points[0][0], points[0][1]),
                    (points[j][0], points[j][1]),
                    int(color[j-1]), thickness)

        # Keep only the points inside the image
        pts = keep_points_inside(points, img_size)

        if len(pts) == 0:
            draw_star(img_size, num_frames, bg_config, nb_branches)

        pts_list.append(pts)
        img_list.append(img)


    images = np.array(img_list)
    points = np.array(pts_list)

    # event_sim = es.Event_simulator(images[0], 0)
    # events = np.array([event_sim.simulate(img, 0) for img in images[1:]])
    return images, points



def draw_checkerboard(img_size, num_frames, bg_config, max_rows=7, max_cols=7, transform_params=(0.05, 0.15)):
    """ Draw a checkerboard and output the interest points
    Parameters:
      max_rows: maximal number of rows + 1
      max_cols: maximal number of cols + 1
      transform_params: set the range of the parameters of the transformations"""

    images_ = generate_background(img_size, num_frames=num_frames)
    background_color = int(np.mean(images_))

    # Create the grid
    rows = random_state.randint(3, max_rows)  # number of rows
    cols = random_state.randint(3, max_cols)  # number of cols
    s = min((img_size[1] - 1) // cols, (img_size[0] - 1) // rows)  # size of a cell
    x_coord = np.tile(range(cols + 1),
                      rows + 1).reshape(((rows + 1) * (cols + 1), 1))
    y_coord = np.repeat(range(rows + 1),
                        cols + 1).reshape(((rows + 1) * (cols + 1), 1))
    points = s * np.concatenate([x_coord, y_coord], axis=1)

    # Warp the grid using an affine transformation and an homography
    # The parameters of the transformations are constrained
    # to get transformations not too far-fetched
    alpha_affine = np.max(img_size) * (transform_params[0]
                                        + random_state.rand() * transform_params[1])
    center_square = np.float32(img_size) // 2
    min_dim = min(img_size)
    square_size = min_dim // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size,
                       [center_square[0]-square_size, center_square[1]+square_size]])
    pts2 = pts1 + random_state.uniform(-alpha_affine,
                                       alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    affine_transform = cv.getAffineTransform(pts1[:3], pts2[:3])
    pts2 = pts1 + random_state.uniform(-alpha_affine / 2,
                                       alpha_affine / 2,
                                       size=pts1.shape).astype(np.float32)
    perspective_transform = cv.getPerspectiveTransform(pts1, pts2)

    # Apply the affine transformation
    points = np.transpose(np.concatenate((points,
                                          np.ones(((rows + 1) * (cols + 1), 1))),
                                         axis=1))
    warped_points = np.transpose(np.dot(affine_transform, points))

    # Apply the homography
    warped_col0 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[0, :2]), axis=1),
                         perspective_transform[0, 2])
    warped_col1 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[1, :2]), axis=1),
                         perspective_transform[1, 2])
    warped_col2 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[2, :2]), axis=1),
                         perspective_transform[2, 2])
    warped_col0 = np.divide(warped_col0, warped_col2)
    warped_col1 = np.divide(warped_col1, warped_col2)
    warped_points = np.concatenate([warped_col0[:, None], warped_col1[:, None]], axis=1)
    warped_points = warped_points.astype(float)
    warped_points = (warped_points / 1.5) + random_state.randint(0, img_size[0]//2)



    # Fill the rectangles with colors
    colors = np.zeros((rows * cols,), np.int32)
    for i in range(rows):
        for j in range(cols):
            # Get a color that contrast with the neighboring cells743
            if i == 0 and j == 0:
                col = get_random_color()
            else:
                neighboring_colors = []
                if i != 0:
                    neighboring_colors.append(colors[(i-1) * cols + j])
                if j != 0:
                    neighboring_colors.append(colors[i * cols + j - 1])
                col = get_different_color(np.array(neighboring_colors))
            colors[i * cols + j] = col
            # Fill the cell

    # Random lines on the boundaries of the board
    nb_rows = random_state.randint(2, rows + 2)
    nb_cols = random_state.randint(2, cols + 2)
    thickness = random_state.randint(min_dim * 0.01, min_dim * 0.02)
    
    row_idx  = [random_state.randint(rows + 1) for _ in range(nb_rows)]
    col_idx1 = [random_state.randint(cols + 1) for _ in range(nb_rows)]
    col_idx2 = [random_state.randint(cols + 1) for _ in range(nb_rows)]
    rows_colors = [get_random_color() for _ in range(nb_rows)]

    col_idx  = [random_state.randint(cols + 1) for _ in range(nb_cols)]
    row_idx1 = [random_state.randint(rows + 1) for _ in range(nb_cols)]
    row_idx2 = [random_state.randint(rows + 1) for _ in range(nb_cols)]
    cols_colors = [get_random_color() for _ in range(nb_cols)]


    # Speed and Rotation
    speed = get_random_speed(min_speed=10, max_speed=20) / 10.0
    uni = random_state.uniform(0.01, 0.05, 1) * random_state.choice([1, -1], 1)
    rotation = get_R(uni)
    decrease = True

    img_list = []
    pts_list = []
    for t in range(num_frames):

        # img = np.zeros(img_size, dtype=np.uint8)
        img = images_[t]
        center = np.average(warped_points, axis=0)
        warped_points = np.matmul(warped_points - center, rotation) + center + speed
        warped_points1 = warped_points.astype(int)

        if decrease:
            speed /= 1.05
            uni /= 1.05
            rotation = get_R(uni)

            if (speed < 0.5).all():
                speed = get_random_speed(max_speed=20) / 50.0
                uni = random_state.uniform(0.01, 0.05, 1) * \
                            random_state.choice([1, -1], 1)
                decrease = False
        else:
            speed *= 1.1
            uni *= 1.1
            speed = np.clip(speed, -2, 2)
            uni = np.clip(uni, -0.05, 0.05)
            rotation = get_R(uni)


        # Draw Checkerboard
        for i in range(rows):
            for j in range(cols):
                cv.fillConvexPoly(img, np.array([(warped_points1[i * (cols + 1) + j, 0],
                                                    warped_points1[i * (cols + 1) + j, 1]),
                                                    (warped_points1[i * (cols + 1) + j + 1, 0],
                                                    warped_points1[i * (cols + 1) + j + 1, 1]),
                                                    (warped_points1[(i + 1)
                                                                * (cols + 1) + j + 1, 0],
                                                    warped_points1[(i + 1)
                                                                * (cols + 1) + j + 1, 1]),
                                                    (warped_points1[(i + 1)
                                                                * (cols + 1) + j, 0],
                                                    warped_points1[(i + 1)
                                                                * (cols + 1) + j, 1])]),
                                    int(colors[i * cols + j]))

        # Draw lines on the boundaries of the board at random
        for i in range(nb_rows):
            cv.line(img, (warped_points1[row_idx[i] * (cols + 1) + col_idx1[i], 0],
                        warped_points1[row_idx[i] * (cols + 1) + col_idx1[i], 1]),
                    (warped_points1[row_idx[i] * (cols + 1) + col_idx2[i], 0],
                    warped_points1[row_idx[i] * (cols + 1) + col_idx2[i], 1]),
                    int(rows_colors[i]), thickness)
        for i in range(nb_cols):
            cv.line(img, (warped_points1[row_idx1[i] * (cols + 1) + col_idx[i], 0],
                        warped_points1[row_idx1[i] * (cols + 1) + col_idx[i], 1]),
                    (warped_points1[row_idx2[i] * (cols + 1) + col_idx[i], 0],
                    warped_points1[row_idx2[i] * (cols + 1) + col_idx[i], 1]),
                    int(cols_colors[i]), thickness)

        # Keep only the points inside the image
        points = keep_points_inside(warped_points1, img.shape[:2])

        if (len(points)) == 0:
            draw_checkerboard(img_size, num_frames, bg_config, max_rows, max_cols, transform_params)

        pts_list.append(points)
        img_list.append(img)


    points = np.array(pts_list)
    images = np.array(img_list)

    # event_sim = es.Event_simulator(images[0], 0)
    # events = np.array([event_sim.simulate(img, 0) for img in images[1:]])

    return images, points


def draw_stripes(img_size, num_frames, bg_config, max_nb_cols=13, min_width_ratio=0.04,
                 transform_params=(0.05, 0.15)):
    """ Draw stripes in a distorted rectangle and output the interest points
    Parameters:
      max_nb_cols: maximal number of stripes to be drawn
      min_width_ratio: the minimal width of a stripe is
                       min_width_ratio * smallest dimension of the image
      transform_params: set the range of the parameters of the transformations
    """

    images_ = generate_background(img_size, num_frames=num_frames)
    background_color = int(np.mean(images_))
    # Create the grid
    board_size = (int(img_size[0] * (0.5 + random_state.rand()/2)),
                  int(img_size[1] * (0.5 + random_state.rand()/2)))
    col = random_state.randint(5, max_nb_cols)  # number of cols
    cols = np.concatenate([board_size[1] * random_state.rand(col - 1),
                           np.array([0, board_size[1] - 1])], axis=0)
    cols = np.unique(cols.astype(int))
    # Remove the indices that are too close
    min_dim = min(img_size)
    min_width = min_dim * min_width_ratio
    cols = cols[(np.concatenate([cols[1:],
                                 np.array([board_size[1] + min_width])],
                                axis=0) - cols) >= min_width]
    col = cols.shape[0] - 1  # update the number of cols
    cols = np.reshape(cols, (col + 1, 1))
    cols1 = np.concatenate([cols, np.zeros((col + 1, 1), np.int32)], axis=1)
    cols2 = np.concatenate([cols,
                            (board_size[0] - 1) * np.ones((col + 1, 1), np.int32)],
                           axis=1)
    points = np.concatenate([cols1, cols2], axis=0)

    # Warp the grid using an affine transformation and an homography
    # The parameters of the transformations are constrained
    # to get transformations not too far-fetched
    # Prepare the matrices
    alpha_affine = np.max(img_size) * (transform_params[0]
                                        + random_state.rand() * transform_params[1])
    center_square = np.float32(img_size) // 2
    square_size = min(img_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size,
                       [center_square[0]-square_size, center_square[1]+square_size]])
    pts2 = pts1 + random_state.uniform(-alpha_affine,
                                       alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    affine_transform = cv.getAffineTransform(pts1[:3], pts2[:3])
    pts2 = pts1 + random_state.uniform(-alpha_affine / 2,
                                       alpha_affine / 2,
                                       size=pts1.shape).astype(np.float32)
    perspective_transform = cv.getPerspectiveTransform(pts1, pts2)

    # Apply the affine transformation
    points = np.transpose(np.concatenate((points,
                                          np.ones((2 * (col + 1), 1))),
                                         axis=1))
    warped_points = np.transpose(np.dot(affine_transform, points))

    # Apply the homography
    warped_col0 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[0, :2]), axis=1),
                         perspective_transform[0, 2])
    warped_col1 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[1, :2]), axis=1),
                         perspective_transform[1, 2])
    warped_col2 = np.add(np.sum(np.multiply(warped_points,
                                            perspective_transform[2, :2]), axis=1),
                         perspective_transform[2, 2])
    warped_col0 = np.divide(warped_col0, warped_col2)
    warped_col1 = np.divide(warped_col1, warped_col2)
    warped_points = np.concatenate([warped_col0[:, None], warped_col1[:, None]], axis=1)
    warped_points = warped_points.astype(int)

    # Fill the rectangles
    colors = np.zeros(col+1)
    colors[-1] = get_random_color(background_color=background_color)


    # Draw lines on the boundaries of the stripes at random
    nb_rows = random_state.randint(2, 5)
    nb_cols = random_state.randint(2, col + 2)

    thickness = random_state.randint(min_dim * 0.01, min_dim * 0.02)


    row_idx  = [random_state.choice([0, col + 1]) for _ in range(nb_rows)]
    col_idx1 = [random_state.randint(col + 1) for _ in range(nb_rows)]
    col_idx2 = [random_state.randint(col + 1) for _ in range(nb_rows)]
    row_color = [get_random_color() for _ in range(nb_rows)]


    col_idx  = [random_state.randint(col + 1) for _ in range(nb_cols)]
    col_color = [get_random_color() for _ in range(nb_cols)]

    for i in range(col):
            colors[i] = (colors[i-1] + 128 + random_state.randint(-30, 30)) % 256


    # Translation and Rotation
    speed = get_random_speed(min_speed=10, max_speed=20) / 10.0
    uni = random_state.uniform(0.01, 0.05, 1) * random_state.choice([1, -1], 1)
    rotation = get_R(uni)
    decrease = True


    img_list = []
    pts_list = []
    for t in range(num_frames):
        
        # img = np.zeros(img_size, dtype=np.uint8)
        img = images_[t]
        center = np.average(warped_points, axis=0)
        warped_points = (np.matmul(warped_points - center, rotation) + center + speed).astype(int)

        if decrease:
            speed /= 1.05
            uni /= 1.05
            rotation = get_R(uni)

            if (speed < 0.5).all():
                speed = get_random_speed(max_speed=20) / 50.0
                uni = random_state.uniform(0.01, 0.05, 1) * \
                            random_state.choice([1, -1], 1)
                decrease = False
        else:
            speed *= 1.1
            uni *= 1.1
            speed = np.clip(speed, -2, 2)
            uni = np.clip(uni, -0.05, 0.05)
            rotation = get_R(uni)
        
            
        for i in range(col):
            cv.fillConvexPoly(img, np.array([(warped_points[i, 0],
                                            warped_points[i, 1]),
                                            (warped_points[i+1, 0],
                                            warped_points[i+1, 1]),
                                            (warped_points[i+col+2, 0],
                                            warped_points[i+col+2, 1]),
                                            (warped_points[i+col+1, 0],
                                            warped_points[i+col+1, 1])]),
                            colors[i])
                            
        for i in range(nb_rows):
            cv.line(img, (warped_points[row_idx[i] + col_idx1[i], 0],
                        warped_points[row_idx[i] + col_idx1[i], 1]),
                    (warped_points[row_idx[i] + col_idx2[i], 0],
                    warped_points[row_idx[i] + col_idx2[i], 1]),
                    int(row_color[i]), thickness)


        for i in range(nb_cols):
            cv.line(img, (warped_points[col_idx[i], 0],
                        warped_points[col_idx[i], 1]),
                    (warped_points[col_idx[i] + col + 1, 0],
                    warped_points[col_idx[i] + col + 1, 1]),
                    int(col_color[i]), thickness)


        # Keep only the points inside the image
        points = keep_points_inside(warped_points, img.shape[:2])

        if len(points) == 0:
            return draw_stripes(img_size, 
                        num_frames, 
                        bg_config, 
                        max_nb_cols, 
                        min_width_ratio,
                        transform_params)

        pts_list.append(points)
        img_list.append(img)

    points = np.array(pts_list)
    images = np.array(img_list)

    # event_sim = es.Event_simulator(images[0], 0)
    # events = np.array([event_sim.simulate(img, 0) for img in images[1:]])

    return images, points


def draw_cube(img_size, num_frames, bg_config, min_size_ratio=0.2, min_angle_rot=math.pi / 10,
              scale_interval=(0.4, 0.6), trans_interval=(0.5, 0.2)):
    """ Draw a 2D projection of a cube and output the corners that are visible
    Parameters:
      min_size_ratio: min(img.shape) * min_size_ratio is the smallest achievable
                      cube side size
      min_angle_rot: minimal angle of rotation
      scale_interval: the scale is between scale_interval[0] and
                      scale_interval[0]+scale_interval[1]
      trans_interval: the translation is between img.shape*trans_interval[0] and
                      img.shape*(trans_interval[0] + trans_interval[1])
    """
    # Generate a cube and apply to it an affine transformation
    # The order matters!
    # The indices of two adjacent vertices differ only of one bit (as in Gray codes)

    images_ = generate_background(img_size, num_frames=num_frames)
    background_color = int(np.mean(images_))
    min_dim = min(img_size[:2])
    min_side = min_dim * min_size_ratio
    lx = min_side + random_state.rand() * 2 * min_dim / 3  # dimensions of the cube
    ly = min_side + random_state.rand() * 2 * min_dim / 3
    lz = min_side + random_state.rand() * 2 * min_dim / 3
    cube = np.array([[0, 0, 0],
                     [lx, 0, 0],
                     [0, ly, 0],
                     [lx, ly, 0],
                     [0, 0, lz],
                     [lx, 0, lz],
                     [0, ly, lz],
                     [lx, ly, lz]])
    rot_angles = random_state.rand(3) * 3 * math.pi / 10. + math.pi / 10.
    rotation = np.array([random_state.uniform(0.008, 0.0012) * random_state.choice([1, -1]) for _ in range(3)])


    scaling = np.array([[scale_interval[0] +
                    random_state.rand() * scale_interval[1], 0, 0],
                    [0, scale_interval[0] +
                    random_state.rand() * scale_interval[1], 0],
                    [0, 0, scale_interval[0] +
                    random_state.rand() * scale_interval[1]]])
    trans = np.array([img_size[1] * trans_interval[0] +
                    random_state.randint(-img_size[1] * trans_interval[1],
                                        img_size[1] * trans_interval[1]),
                    img_size[0] * trans_interval[0] +
                    random_state.randint(-img_size[0] * trans_interval[1],
                                        img_size[0] * trans_interval[1]),
                    0])

    col_face = get_random_color(background_color=background_color)
    thickness = (random_state.randint(1, 10))
    speed = get_random_speed(min_speed=20, max_speed=30) / 10.0
    decrease = True

    for i in [0, 1, 2]:
            for j in [0, 1, 2, 3]:
                col_edge = (col_face + 128
                            + random_state.randint(-64, 64))\
                            % 256

    img_list = []
    pts_list = []
    for t in range(num_frames):

        # img = np.zeros(img_size, dtype=np.uint8)
        img = images_[t]
        cube = np.array([[0, 0, 0],
                     [lx, 0, 0],
                     [0, ly, 0],
                     [lx, ly, 0],
                     [0, 0, lz],
                     [lx, 0, lz],
                     [0, ly, lz],
                     [lx, ly, lz]])

        rot_angles += rotation

        rotation_1 = np.array([[math.cos(rot_angles[0]), -math.sin(rot_angles[0]), 0],
                            [math.sin(rot_angles[0]), math.cos(rot_angles[0]), 0],
                            [0, 0, 1]])
        rotation_2 = np.array([[1, 0, 0],
                            [0, math.cos(rot_angles[1]), -math.sin(rot_angles[1])],
                            [0, math.sin(rot_angles[1]), math.cos(rot_angles[1])]])
        rotation_3 = np.array([[math.cos(rot_angles[2]), 0, -math.sin(rot_angles[2])],
                            [0, 1, 0],
                            [math.sin(rot_angles[2]), 0, math.cos(rot_angles[2])]])

                        
        cube = trans + np.transpose(np.dot(scaling,
                                        np.dot(rotation_1,
                                                np.dot(rotation_2,
                                                        np.dot(rotation_3,
                                                                np.transpose(cube))))))


        # The hidden corner is 0 by construction
        # The front one is 7
        cube = cube[:, :2]  # project on the plane z=0
        cube = cube.astype(int)
        points = cube[1:, :]  # get rid of the hidden corner
        points += speed.astype(int)

        if decrease:
            speed /= 1.1
            rotation /= 1.1

            if (speed < 0.8).all():
                speed = get_random_speed(min_speed=10, max_speed=30) / 10.0
                rotation = np.array([random_state.uniform(0.002, 0.005) / 10 * random_state.choice([1, -1]) for _ in range(3)])
                decrease = False
        else:
            speed *= 1.1
            rotation *= 1.1
            speed = np.clip(speed, -2, 2)
            rotation = np.clip(rotation, -0.002, 0.002)


        # Get the three visible faces
        faces = np.array([[7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]])

        # Fill the faces and draw the contours
        for i in [0, 1, 2]: 
            cv.fillPoly(img, [cube[faces[i]].reshape((-1, 1, 2))], int(col_face))
        
        for i in [0, 1, 2]:
            for j in [0, 1, 2, 3]:
                cv.line(img, (cube[faces[i][j], 0], cube[faces[i][j], 1]),
                        (cube[faces[i][(j + 1) % 4], 0], cube[faces[i][(j + 1) % 4], 1]),
                        int(col_edge), thickness)



        # Keep only the points inside the image
        points = keep_points_inside(points, img_size[:2])

        if len(points) == 0:
            return draw_cube(img_size, 
                            num_frames, 
                            bg_config, 
                            min_size_ratio, 
                            min_angle_rot, 
                            scale_interval, 
                            trans_interval)

        pts_list.append(points)
        img_list.append(img)

    points = np.array(pts_list)
    images = np.array(img_list)

    # event_sim = es.Event_simulator(images[0], 0)
    # events = np.array([event_sim.simulate(img, 0) for img in images[1:]])

    return images, points


def gaussian_noise(img):
    """ Apply random noise to the image """
    cv.randu(img, 0, 255)
    return np.empty((0, 2), dtype=np.int)


def draw_interest_points(img, points):
    """ Convert img in RGB and draw in green the interest points """
    img_rgb = np.stack([img, img, img], axis=2)
    for i in range(points.shape[0]):
        cv.circle(img_rgb, (points[i][0], points[i][1]), 5, (0, 255, 0), -1)
    return img_rgb




if __name__ == "__main__":

    bg_config = {"min_kernel_size": 150,
                "max_kernel_size": 500,
                "min_rad_ratio": 0.02,
                "max_rad_ratio": 0.031,}


    parser = ArgumentParser()
    parser.add_argument("-d", "--draw_function", dest="draw_function",
                        help="Select a draw function")
    parser.add_argument("-H", "--height", dest="height", default=130, type=int,
                        help="Original Height of a generated image")
    parser.add_argument("-W", "--width", dest="width", default=170, type=int,
                        help="Original Width of a generated image")
    parser.add_argument("-i", "--iteration", dest="iteration", default=1000, type=int,
                        help="How many iterations for each shape batch")
    parser.add_argument("-f", "--frames", dest="frames", default=6, type=int,
                        help="Frames per batch")
    parser.add_argument("-v", "--validation_size", dest="val_size", type=int,
                        help="validation size")
    parser.add_argument("-r", "--resize", 
                        action="store_true", dest="is_resize", default=False,
                        help="if resize the generated image after generation step or not")
    parser.add_argument("-x", "--resize_height", dest="resize_height", default=160, type=int,
                        help="resize height of a generated image")
    parser.add_argument("-y", "--resize_width", dest="resize_width", default=120, type=int,
                        help="resize width of a generated image")

    args = parser.parse_args()

    H, W = args.height, args.width
    y, x = args.resize_height, args.resize_width

    iteration = args.iteration
    frames = args.frames
    resize = args.is_resize


    # print("======================================")
    # print("Generation Synthetic dataset")
    # print("======================================")
    # print("Original image size: ", H, W)
    # print("Resize image size: ", y, x)
    # print("Iteration: ", iteration)
    # print("Frames: ", frames)
    # print("Total images: ", frames*iteration)
    # print("Resize: ", resize)
    # print("======================================")
    


    if args.draw_function:
        draw_list = [args.draw_function]
    else:
        # draw_list = ["draw_checkerboard", "draw_cube", "draw_ellipses", "draw_lines", "draw_polygon", "draw_star", "draw_stripes", "draw_multiple_polygons"]
        draw_list = ["draw_checkerboard"]

    
    for k in range(len(draw_list)):

        draw_foo = draw_list[k]

        # img_path = Path(f"/media/gen/data/{draw_foo.__name__}/images")
        # pts_path = Path(f"/media/gen/data/{draw_foo.__name__}/raw_points")
        # check_path = Path(f"/media/gen/data/{draw_foo.__name__}/images_check")
        img_path = Path(f"/media/gen/data/{draw_foo}/images")
        pts_path = Path(f"/media/gen/data/{draw_foo}/raw_points")
        check_path = Path(f"/media/gen/data/{draw_foo}/images_check")

        img_path.mkdir(parents=True, exist_ok=True)
        pts_path.mkdir(parents=True, exist_ok=True)
        check_path.mkdir(parents=True, exist_ok=True)
        
        draw_function = locals()[draw_foo]
        for i in tqdm(range(iteration), desc=draw_foo):

            imgs, pnts = draw_function((H, W), frames, bg_config)

            for j, (img, pts) in enumerate(zip(imgs, pnts)):

                idx = i*frames+j

                if resize:
                    img = cv.resize(img, (y, x))
                    pts = pts / 8

                img[random_state.randint(0,x-1),random_state.randint(0,y-1)] = 128
                if j%2:
                    img[-1,-1] = 256

                if i % 199 == 0:
                    check_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                    for pt in pts:
                        cv.circle(check_img, (int(pt[0]), int(pt[1])), 3, [0, 255, 0], -1)
                    chk_ = str(Path(check_path, "chk_{:05d}.png".format(idx)))
                    cv.imwrite(chk_, check_img)

                img_ = str(Path(img_path, "img_{:05d}.png".format(idx)))
                pnt_ = str(Path(pts_path, "pnt_{:05d}.npy".format(idx)))
                cv.imwrite(img_, img)
                np.save(pnt_, pts)




