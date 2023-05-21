import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform, AffineTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv


DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=200):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    descriptor_extractor.detect_and_extract(rgb2gray(img))

    return descriptor_extractor.keypoints, descriptor_extractor.descriptors


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


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=100, residual_threshold=1, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """
    matches = match_descriptors(src_descriptors, dest_descriptors, cross_check=True)
    src_match = src_keypoints[matches[:, 0]]
    dest_match = dest_keypoints[matches[:, 1]]
    N = matches.shape[0]

    H = ProjectiveTransform(find_homography(src_match, dest_match))
    inliers = np.linalg.norm(dest_match - H(src_match), axis=-1) < residual_threshold
    inliers_cnt = inliers.sum()

    for _ in range(max_trials):
        indices = np.random.choice(N, 4, replace=False)
        sample_src_match = src_match[indices]
        sample_dest_match = dest_match[indices]

        new_H = ProjectiveTransform(find_homography(sample_src_match, sample_dest_match))
        new_inliers = np.linalg.norm(dest_match - new_H(src_match), axis=-1) < residual_threshold
        new_inliers_cnt = new_inliers.sum()

        if new_inliers_cnt > inliers_cnt:
            H, inliers, inliers_cnt = new_H, new_inliers, new_inliers_cnt

    H = ProjectiveTransform(find_homography(src_match[inliers, :], dest_match[inliers, :]))

    if return_matches:
        return H, matches[inliers]
    return H


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()

    for i in range(center_index - 1, -1, -1):
        result[i] = result[i + 1] + forward_transforms[i]

    for i in range(center_index + 1, image_count):
        result[i] = result[i - 1] + ProjectiveTransform(inv(forward_transforms[i - 1].params))

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_centers(image_collection, center_warps):
    """Get centers' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        yield transform([[height / 2, width / 2]])


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
        """
    min_coords, max_coords = get_min_max_coords(tuple(get_corners(image_collection, simple_center_warps)))
    shift = AffineTransform(translation=-min_coords[::-1])

    return (transform + shift for transform in simple_center_warps), tuple(np.ceil(max_coords - min_coords).astype(int))[::-1]


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    mask = np.ones(shape=image.shape[:2], dtype=np.bool8)
    transform = rotate_transform_matrix(transform)

    return warp(image, transform, output_shape=output_shape), warp(mask, transform, output_shape=output_shape)


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    for image, transform in zip(image_collection, final_center_warps):
        image, mask = warp_image(image, transform, output_shape)

        to_update = mask & ~result_mask
        result_mask = result_mask | mask

        result[to_update] = image[to_update]

    return result


def get_gaussian_pyramid(image, n_layers=4, sigma=3):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    result = [image]
    for layer in range(n_layers - 1):
        result.append(
            gaussian(
                result[layer],
                sigma=sigma,
                # channel_axis=-1,
            )
        )


def get_laplacian_pyramid(image, n_layers=4, sigma=3):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    gaussian_pyramid = get_gaussian_pyramid(image, n_layers, sigma)
    return [prv - nxt for prv, nxt in zip(gaussian_pyramid[:-1], gaussian_pyramid[1:])] + [gaussian_pyramid[-1]]


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=4, image_sigma=2, merge_sigma=15):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    centers = list(get_centers(image_collection, final_center_warps))
    X, Y = np.meshgrid(np.arange(output_shape[1]), np.arange(output_shape[0]))

    images = []
    masks = []

    for n_image, (image, transform) in enumerate(zip(image_collection, final_center_warps)):
        image, mask = warp_image(image, transform, output_shape)
        images.append(image)

        for k in range(n_image):
            n_image_dist = (Y - centers[n_image][0]) ** 2 + (X - centers[n_image][1]) ** 2
            k_image_dist = (Y - centers[k][0]) ** 2 + (X - centers[k][1]) ** 2
            merge_mask = n_image_dist < k_image_dist

            masks[k] = masks[k] & ~merge_mask
            mask = mask & merge_mask

        masks.append(mask)

    masks_pyramid = np.array([
        get_gaussian_pyramid(mask[..., np.newaxis], n_layers, merge_sigma) for mask in masks
    ])
    masks_pyramid /= np.maximum(masks_pyramid.sum(axis=0), 1e-9)

    images_pyramid = np.array([
        get_laplacian_pyramid(image, n_layers, image_sigma) for image in images
    ])
    return np.clip((images_pyramid * masks_pyramid).sum(axis=(0, 1)), 0, 1)


def cylindrical_inverse_map(coords, h, w, scale):
    """Function that transform coordinates in the output image
    to their corresponding coordinates in the input image
    according to cylindrical transform.

    Use it in skimage.transform.warp as `inverse_map` argument

    coords ((M, 2) np.ndarray) : coordinates of output image (M == col * row)
    h (int) : height (number of rows) of input image
    w (int) : width (number of cols) of input image
    scale (int or float) : scaling parameter

    Returns:
        (M, 2) np.ndarray : corresponding coordinates of input image (M == col * row) according to cylindrical transform
    """
    C = np.array([coords[:, 0], coords[:, 1], np.ones(coords.shape[0])])
    K = np.array([[scale, 0, w / 2], [0, scale, h / 2], [0, 0, 1]])

    C_hat = inv(K) @ C
    B_hat_x = np.tan(C_hat[0])
    B_hat_y = C_hat[1] / np.cos(C_hat[0])
    B_hat = np.array([B_hat_x, B_hat_y, np.ones(B_hat_x.shape[1])])

    return (K @ B_hat)[:2].T


def warp_cylindrical(img, scale=None, crop=True):
    """Warp image to cylindrical coordinates

    img ((H, W, 3)  np.ndarray) : image for transformation
    scale (int or None) : scaling parameter. If None, defaults to W * 0.5
    crop (bool) : crop image to fit (remove unnecessary zero-padding of image)

    Returns:
        (H, W, 3)  np.ndarray : warped image (H and W may differ from original)
    """
    h, w = img.shape[:2]
    return warp(
        img,
        cylindrical_inverse_map,
        map_args={
            'h': h,
            'w': w,
            'scale': scale if scale is not None else w * 0.5
        },
        output_shape=(h, w)
    )


# Pick a good scale value for the 5 test image sets
cylindrical_scales = {
    0: 1000,
    1: 750,
    2: 2500,
    3: 10000,
    4: 10000,
}
