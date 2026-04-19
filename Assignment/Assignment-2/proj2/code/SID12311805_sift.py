import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    image = image.astype(np.float32)
    num_points = len(x)

    if num_points == 0:
        return np.zeros((0, 128), dtype=np.float32)

    # Precompute gradients once for all points.
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ori = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)

    half = feature_width // 2
    cell_size = feature_width // 4
    num_bins = 8

    # Pad so boundary keypoints still produce a valid descriptor.
    mag_pad = np.pad(mag, ((half, half), (half, half)), mode='constant')
    ori_pad = np.pad(ori, ((half, half), (half, half)), mode='constant')

    # Gaussian window to emphasize gradients near keypoint center.
    g1d = cv2.getGaussianKernel(feature_width, feature_width / 2.0)
    gauss_w = (g1d @ g1d.T).astype(np.float32)

    fv = np.zeros((num_points, 128), dtype=np.float32)

    for i in range(num_points):
        cx = int(np.round(x[i])) + half
        cy = int(np.round(y[i])) + half

        y0 = cy - half
        y1 = cy + half
        x0 = cx - half
        x1 = cx + half

        patch_mag = mag_pad[y0:y1, x0:x1]
        patch_ori = ori_pad[y0:y1, x0:x1]

        if patch_mag.shape != (feature_width, feature_width):
            continue

        patch_mag = patch_mag * gauss_w

        desc = []
        for ry in range(4):
            for rx in range(4):
                cy0 = ry * cell_size
                cy1 = (ry + 1) * cell_size
                cx0 = rx * cell_size
                cx1 = (rx + 1) * cell_size

                cell_mag = patch_mag[cy0:cy1, cx0:cx1].reshape(-1)
                cell_ori = patch_ori[cy0:cy1, cx0:cx1].reshape(-1)

                hist, _ = np.histogram(
                    cell_ori,
                    bins=num_bins,
                    range=(0, 2 * np.pi),
                    weights=cell_mag
                )
                desc.extend(hist.tolist())

        desc = np.array(desc, dtype=np.float32)

        # SIFT-style normalization and clipping for robustness.
        norm = np.linalg.norm(desc) + 1e-12
        desc = desc / norm
        desc = np.clip(desc, 0, 0.2)
        desc = desc / (np.linalg.norm(desc) + 1e-12)
        desc = np.sqrt(desc)

        fv[i, :] = desc

    return fv

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    raise NotImplementedError('`get_features` function in ' +
        '`student_sift.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
