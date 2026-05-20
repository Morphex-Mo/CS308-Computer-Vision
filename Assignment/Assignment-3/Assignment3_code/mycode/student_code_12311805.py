import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from IPython.core.debugger import set_trace
from PIL import Image
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time


def get_tiny_images(image_paths):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    To build a tiny image feature, simply resize the original image to a very
    small square resolution, e.g. 16x16. You can either resize the images to
    square while ignoring their aspect ratio or you can crop the center
    square portion out of each image. Making the tiny images zero mean and
    unit length (normalizing them) will increase performance modestly.

    Useful functions:
    -   cv2.resize
    -   use load_image(path) to load a RGB images and load_image_gray(path) to
        load grayscale images

    Args:
    -   image_paths: list of N elements containing image paths

    Returns:
    -   feats: N x d numpy array of resized and then vectorized tiny images
              e.g. if the images are resized to 16x16, d would be 256
    """
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    for image_path in image_paths:
        image = load_image_gray(image_path)
        tiny_image = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
        tiny_image = tiny_image.astype(np.float32).reshape(-1)
        tiny_image -= np.mean(tiny_image)
        norm = np.linalg.norm(tiny_image)
        if norm > 0:
            tiny_image /= norm
        feats.append(tiny_image)

    feats = np.asarray(feats, dtype=np.float32)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def build_vocabulary(image_paths, vocab_size, step=15, sample_per_image=100):
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
            -  frames is a N x 2 matrix of locations, which can be thrown away
            here (but possibly used for extra credit in get_bags_of_sifts if
            you're making a "spatial pyramid").
            -  descriptors is a N x 128 matrix of SIFT features
          Note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster to
          compute. Also, be sure not to use the default value of step size. It
          will be very slow and you'll see relatively little performance gain
          from extremely dense sampling. You are welcome to use your own SIFT
          feature code! It will probably be slower, though.
    -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
            http://www.vlfeat.org/matlab/vl_kmeans.html
              -  X is a N x d numpy array of sampled SIFT features, where N is
                 the number of features sampled. N should be pretty large!
              -  K is the number of clusters desired (vocab_size)
                 cluster_centers is a K x d matrix of cluster centers. This is
                 your vocabulary.

    Args:
    -   image_paths: list of image paths.
    -   vocab_size: size of vocabulary

    Returns:
    -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
        cluster center / visual word
    """
    dim = 128
    vocab = np.zeros((vocab_size, dim))

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    rng = np.random.default_rng(0)
    descriptors_list = []

    for image_path in image_paths:
        image = load_image_gray(image_path)
        _, descriptors = dsift(image, step=step, fast=True)
        if descriptors is None or descriptors.size == 0:
            continue
        sample_count = min(sample_per_image, descriptors.shape[0])
        sample_indices = rng.choice(descriptors.shape[0], size=sample_count, replace=False)
        descriptors_list.append(descriptors[sample_indices])

    if len(descriptors_list) == 0:
        raise ValueError('No SIFT descriptors were extracted to build the vocabulary')

    descriptors_stack = np.vstack(descriptors_list).astype(np.float32)
    vocab = kmeans(descriptors_stack, vocab_size).astype(np.float32)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return vocab


def get_bags_of_sifts(image_paths, vocab_filename, step=15):
    """
    This feature representation is described in the handout, lecture
    materials, and Szeliski chapter 14.
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
            http://www.vlfeat.org/matlab/vl_dsift.html
          frames is a M x 2 matrix of locations, which can be thrown away here
            (but possibly used for extra credit in get_bags_of_sifts if you're
            making a "spatial pyramid").
          descriptors is a M x 128 matrix of SIFT features
            note: there are step, bin size, and smoothing parameters you can
            manipulate for dsift(). We recommend debugging with the 'fast'
            parameter. This approximate version of SIFT is about 20 times faster
            to compute. Also, be sure not to use the default value of step size.
            It will be very slow and you'll see relatively little performance
            gain from extremely dense sampling. You are welcome to use your own
            SIFT feature code! It will probably be slower, though.
    -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
            finds the cluster assigments for features in data
              -  data is a M x d matrix of image features
              -  vocab is the vocab_size x d matrix of cluster centers
              (vocabulary)
              -  assignments is a Mx1 array of assignments of feature vectors to
              nearest cluster centers, each element is an integer in
              [0, vocab_size)

    Args:
    -   image_paths: paths to N images
    -   vocab_filename: Path to the precomputed vocabulary.
            This function assumes that vocab_filename exists and contains an
            vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
            or visual word. This ndarray is saved to disk rather than passed in
            as a parameter to avoid recomputing the vocabulary every run.

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
            feature representation. In this case, d will equal the number of
            clusters or equivalently the number of entries in each image's
            histogram (vocab_size) below.
    """
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    for image_path in image_paths:
        image = load_image_gray(image_path)
        _, descriptors = dsift(image, step=step, fast=True)

        if descriptors is None or descriptors.size == 0:
            histogram = np.zeros(vocab.shape[0], dtype=np.float32)
        else:
            assignments = vlfeat.kmeans.kmeans_quantize(descriptors.astype(np.float32), vocab)
            histogram = np.bincount(assignments.ravel(), minlength=vocab.shape[0]).astype(np.float32)
            if histogram.sum() > 0:
                histogram /= histogram.sum()

        feats.append(histogram)

    feats = np.asarray(feats, dtype=np.float32)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
                              metric='euclidean'):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which will increase
    performance (although you need to pick a reasonable value for k).

    Useful functions:
    -   D = sklearn_pairwise.pairwise_distances(X, Y)
          computes the distance matrix D between all pairs of rows in X and Y.
            -  X is a N x d numpy array of d-dimensional features arranged along
            N rows
            -  Y is a M x d numpy array of d-dimensional features arranged along
            N rows
            -  D is a N x M numpy array where d(i, j) is the distance between row
            i of X and row j of Y

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating
            the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of
            the feature representation. You can assume N = M, unless you have changed
            the starter code
    -   metric: (optional) metric to be used for nearest neighbor.
            Can be used to select different distance functions. The default
            metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
            well for histograms

    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    train_image_feats = np.asarray(train_image_feats)
    test_image_feats = np.asarray(test_image_feats)

    # If features look like histograms and caller kept default metric,
    # switch to chi2 automatically.
    is_hist_like = (np.min(train_image_feats) >= -1e-8 and np.min(test_image_feats) >= -1e-8)
    if metric == 'euclidean' and is_hist_like:
        metric = 'chi2'

    # Use k-NN voting for histogram features, 1-NN for tiny-image style features.
    if metric == 'chi2':
        k = 9
    else:
        k = 1

    if metric == 'chi2':
        distances = np.zeros((test_image_feats.shape[0], train_image_feats.shape[0]), dtype=np.float32)
        eps = 1e-10
        for i, test_feat in enumerate(test_image_feats):
            numerator = (train_image_feats - test_feat) ** 2
            denominator = train_image_feats + test_feat + eps
            distances[i] = 0.5 * np.sum(numerator / denominator, axis=1)
    else:
        distances = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats, metric=metric)

    nearest_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    train_labels_np = np.asarray(train_labels)
    all_categories = np.unique(train_labels_np)

    for neighbors in nearest_indices:
        labels = train_labels_np[neighbors]
        if k == 1:
            test_labels.append(labels[0])
            continue

        # Majority vote with distance-weighted tie-break.
        best_label = None
        best_votes = -1
        best_weight = -1.0
        for cat in all_categories:
            mask = labels == cat
            votes = int(np.sum(mask))
            if votes == 0:
                continue
            weight = float(np.sum(1.0 / (distances[len(test_labels), neighbors[mask]] + 1e-10)))
            if votes > best_votes or (votes == best_votes and weight > best_weight):
                best_votes = votes
                best_weight = weight
                best_label = cat
        test_labels.append(best_label)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels


def svm_classify(train_image_feats, train_labels, test_image_feats):
    """
    This function will train a linear SVM for every category (i.e. one vs all)
    and then use the learned linear classifiers to predict the category of
    every test image. Every test feature will be evaluated with all 15 SVMs
    and the most confident SVM will "win". Confidence, or distance from the
    margin, is W*X + B where '*' is the inner product or dot product and W and
    B are the learned hyperplane parameters.

    Useful functions:
    -   sklearn LinearSVC
          http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    -   svm.fit(X, y)
    -   set(l)

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating
            the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of
            the feature representation. You can assume N = M, unless you have changed
            the starter code
    Returns:
    -   test_labels: M element list, where each entry is a string indicating
            the predicted category for each testing image
    """
    categories = sorted(list(set(train_labels)))

    train_image_feats = np.asarray(train_image_feats, dtype=np.float32)
    test_image_feats = np.asarray(test_image_feats, dtype=np.float32)

    # Use raw BoW histograms with standard scaling and a tuned linear SVM.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_image_feats)
    test_scaled = scaler.transform(test_image_feats)

    svms = {cat: LinearSVC(random_state=0, tol=1e-4, loss='squared_hinge', C=1.0,
                          max_iter=50000, dual=True)
            for cat in categories}

    for category, svm in svms.items():
        binary_labels = np.array([1 if label == category else -1 for label in train_labels])
        svm.fit(train_scaled, binary_labels)

    scores = []
    for category in categories:
        scores.append(svms[category].decision_function(test_scaled))

    scores = np.vstack(scores).T
    best_indices = np.argmax(scores, axis=1)
    test_labels = [categories[index] for index in best_indices]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels
