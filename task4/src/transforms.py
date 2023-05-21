import numpy as np
from skimage.transform import warp, AffineTransform, ProjectiveTransform
from numpy.linalg import inv

import cv2

from src.global_params import *


def adjust_image(im, win_h = STANDARD_WIH_H, win_w = STANDARD_WIH_W, ignore_small=True):
    H, W = im.shape[:2]
    k = max(H / win_h, W / win_w)
    if ignore_small and k < 1:
        return im, 1
    H, W = int(H / k), int(W / k)
    return cv2.resize(im, (W, H)), k


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """

    C = points.mean(axis=0)
    new_points = points - C
    N = np.sqrt(2) / np.linalg.norm(new_points, axis=1).mean()
    new_points = N * new_points
    M = np.array([
        [N, 0, -N * C[0]],
        [0, N, -N * C[1]],
        [0, 0,         1],
    ])

    return M, new_points


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """
    assert src_keypoints.shape == dest_keypoints.shape

    src_matrix, src_normalized = center_and_normalize_points(src_keypoints)
    dest_matrix, dest_normalized = center_and_normalize_points(dest_keypoints)

    # x_1 and y_1
    x1 = src_normalized[:, 0]
    y1 = src_normalized[:, 1]

    # x_2' and y_2'
    x2 = dest_normalized[:, 0]
    y2 = dest_normalized[:, 1]

    N = src_keypoints.shape[0]
    A = np.zeros(shape=(2 * N, 9))

    # a_x
    A[::2, 0] = -x1
    A[::2, 1] = -y1
    A[::2, 2] = -1
    A[::2, 3:6] = 0
    A[::2, 6:9] = -x2[:, np.newaxis] * A[::2, 0:3]

    # a_y
    A[1::2, 0:3] = 0
    A[1::2, 3] = -x1
    A[1::2, 4] = -y1
    A[1::2, 5] = -1
    A[1::2, 6:9] = -y2[:, np.newaxis] * A[1::2, 3:6]

    H = np.linalg.svd(A)[2][-1].reshape(3, 3)

    return inv(dest_matrix) @ H @ src_matrix


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape, interpolation='nn', mask=None):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    if mask is None:
        mask = np.ones(shape=image.shape, dtype=np.bool8)
    else:
        assert mask.shape == image.shape
        assert mask.dtype == np.bool8
    
    interpolation = {
        "nn": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "biqubic": 3,
        "biquartic": 4,
        "biquintic": 5,
    }[interpolation]

    transform = rotate_transform_matrix(transform)
    return (
        warp(image, transform, output_shape=output_shape, order=interpolation).astype(np.uint8),
        warp(mask, transform, output_shape=output_shape),
    )


def get_rotation_transform(angle):
    return AffineTransform(rotation=angle)


def get_flatten_transform(pts_src, pts_dst):
    M = find_homography(pts_src, pts_dst)
    return rotate_transform_matrix(ProjectiveTransform(M))


def combine_transforms(transform1, transform2):
    return transform1 + transform2


def scan_transform(angle, position, scale, paper_size=(297, 210)):
    return AffineTransform(translation=[-p for p in position]) +\
           AffineTransform(rotation=-angle) +\
           AffineTransform(translation=[p / 2 for p in paper_size]) +\
           AffineTransform(scale=scale)


def inverse_transform(transform):
    return ProjectiveTransform(matrix=inv(transform.params))


def project_image(scan, background, transform):
    scan_out, mask_out = warp_image(scan, transform, background.shape[:2])
    if len(scan_out.shape) == 2:
        scan_out = scan_out.reshape(*scan_out.shape, 1)

    if len(mask_out.shape) == 2:
        mask_out = mask_out.reshape(*mask_out.shape, 1)
    
    if len(background.shape) == 2:
        background = background.reshape(*background.shape, 1)
    im_out = np.where(mask_out, scan_out, background)

    return im_out.squeeze()