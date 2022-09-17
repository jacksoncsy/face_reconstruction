import numpy as np

from skimage.transform import estimate_transform, warp


def parse_bbox_from_landmarks(landmarks):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])
    bbox = [left, top, right, bottom]
    return bbox


def bbox2point(bbox):
    ''' take only the bbox from landmarks
    '''
    left, top, right, bottom = bbox
    size = 1.1 * (right - left + bottom - top) / 2.
    center = np.array([
        right - (right - left) / 2.,
        bottom - (bottom - top) / 2.,
    ])
    return size, center


def compute_similarity_transform(src_size, src_center, crop_size, scale=1.25):
    size = int(src_size * scale)
    src_pts = np.array([
        [src_center[0] - size / 2, src_center[1] - size / 2],
        [src_center[0] - size / 2, src_center[1] + size / 2],
        [src_center[0] + size / 2, src_center[1] - size / 2],
    ])

    dst_pts = np.array([
        [0, 0], 
        [0, crop_size - 1],
        [crop_size - 1, 0],
    ])

    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform


def transform_image(image, tform, crop_size):
    return warp(image, tform.inverse, output_shape=(crop_size, crop_size))