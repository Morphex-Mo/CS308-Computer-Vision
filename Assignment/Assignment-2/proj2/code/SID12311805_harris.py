import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    # ---------------------------
    # Harris corner detection
    # ---------------------------
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32)

    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy

    sigma = 1.5
    sxx = cv2.GaussianBlur(ixx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    syy = cv2.GaussianBlur(iyy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sxy = cv2.GaussianBlur(ixy, (0, 0), sigmaX=sigma, sigmaY=sigma)

    k = 0.04
    det_m = sxx * syy - sxy * sxy
    trace_m = sxx + syy
    r = det_m - k * (trace_m ** 2)

    # Suppress points near border where descriptor window cannot fit.
    border = feature_width // 2
    if border > 0:
        r[:border, :] = 0
        r[-border:, :] = 0
        r[:, :border] = 0
        r[:, -border:] = 0

    r_max = float(np.max(r)) if r.size > 0 else 0.0
    if r_max <= 0:
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)
        confidences = np.array([], dtype=np.float32)
        return x, y, confidences, scales, orientations

    thresh = 0.01 * r_max
    local_max = (r == cv2.dilate(r, np.ones((3, 3), dtype=np.uint8)))
    candidate_mask = local_max & (r > thresh)
    ys, xs = np.where(candidate_mask)

    if xs.size == 0:
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)
        confidences = np.array([], dtype=np.float32)
        return x, y, confidences, scales, orientations

    candidate_conf = r[ys, xs]
    sort_idx = np.argsort(-candidate_conf)
    xs = xs[sort_idx]
    ys = ys[sort_idx]
    candidate_conf = candidate_conf[sort_idx]

    # Limit ANMS candidate count for speed while keeping strong responses.
    max_candidates = 6000
    if xs.size > max_candidates:
        xs = xs[:max_candidates]
        ys = ys[:max_candidates]
        candidate_conf = candidate_conf[:max_candidates]

    # ---------------------------
    # Adaptive Non-Maximal Suppression (ANMS)
    # ---------------------------
    num_candidates = xs.size
    radii = np.full(num_candidates, np.inf, dtype=np.float32)
    robust_factor = 1.1

    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    for i in range(1, num_candidates):
        stronger = candidate_conf[:i] > (candidate_conf[i] * robust_factor)
        if np.any(stronger):
            stronger_pts = pts[:i][stronger]
            dx = stronger_pts[:, 0] - pts[i, 0]
            dy = stronger_pts[:, 1] - pts[i, 1]
            d2 = dx * dx + dy * dy
            radii[i] = np.sqrt(np.min(d2))

    n = 1500
    keep = min(n, num_candidates)
    keep_idx = np.argsort(-radii)[:keep]

    x = xs[keep_idx].astype(np.float32)
    y = ys[keep_idx].astype(np.float32)
    confidences = candidate_conf[keep_idx].astype(np.float32)

    # Keep outputs sorted by suppression radius (largest first).
    order = np.argsort(-radii[keep_idx])
    x = x[order]
    y = y[order]
    confidences = confidences[order]
    return x, y, confidences, scales, orientations

    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################

    raise NotImplementedError('`get_interest_points` function in ' +
    '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    raise NotImplementedError('adaptive non-maximal suppression in ' +
    '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y, confidences, scales, orientations

