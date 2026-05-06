import cv2
import numpy as np
import pickle
from utils import load_image_gray

try:
    import cyvlfeat as vlfeat
except ImportError:
    vlfeat = None

import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC


def get_tiny_images(image_paths):
    """
    Build tiny image features by resizing to 16x16, then zero-mean and L2 norm.
    """
    tiny_size = (16, 16)
    feats = np.zeros((len(image_paths), tiny_size[0] * tiny_size[1]), dtype=np.float32)

    for idx, path in enumerate(image_paths):
        img = load_image_gray(path)
        tiny = cv2.resize(img, tiny_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        tiny = tiny.reshape(-1)
        tiny -= np.mean(tiny)
        norm = np.linalg.norm(tiny)
        if norm > 1e-12:
            tiny /= norm
        feats[idx, :] = tiny

    return feats


def _extract_local_descriptors(gray_img, step=8):
    """
    Extract local descriptors with dsift when available; otherwise fallback to OpenCV SIFT.
    """
    gray_img = gray_img.astype(np.float32)

    if vlfeat is not None:
        _, descriptors = vlfeat.sift.dsift(gray_img, fast=True, step=step)
        descriptors = descriptors.astype(np.float32)
    else:
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute((gray_img * 255).astype(np.uint8), None)
        if descriptors is None:
            descriptors = np.zeros((0, 128), dtype=np.float32)
        else:
            descriptors = descriptors.astype(np.float32)

    return descriptors


def build_vocabulary(image_paths, vocab_size):
    """
    Sample descriptors from training images and cluster into visual words.
    """
    rng = np.random.default_rng(42)
    num_images_for_vocab = min(len(image_paths), 400)
    samples_per_image = 250

    if len(image_paths) > num_images_for_vocab:
        chosen_idx = rng.choice(len(image_paths), size=num_images_for_vocab, replace=False)
        selected_paths = [image_paths[i] for i in chosen_idx]
    else:
        selected_paths = image_paths

    descriptor_list = []
    for path in selected_paths:
        img = load_image_gray(path)
        descriptors = _extract_local_descriptors(img, step=8)
        if descriptors.shape[0] == 0:
            continue

        if descriptors.shape[0] > samples_per_image:
            pick = rng.choice(descriptors.shape[0], size=samples_per_image, replace=False)
            descriptors = descriptors[pick]

        descriptor_list.append(descriptors)

    if len(descriptor_list) == 0:
        raise RuntimeError('No descriptors extracted while building vocabulary.')

    all_desc = np.vstack(descriptor_list).astype(np.float32)

    if vlfeat is not None:
        vocab = vlfeat.kmeans.kmeans(all_desc, vocab_size)
        vocab = vocab.astype(np.float32)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
        attempts = 5
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(
            all_desc,
            vocab_size,
            None,
            criteria,
            attempts,
            flags,
        )
        _ = compactness
        _ = labels
        vocab = centers.astype(np.float32)

    return vocab


def get_bags_of_sifts(image_paths, vocab_filename):
    """
    Represent each image as normalized histogram of visual words.
    """
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    vocab = vocab.astype(np.float32)
    vocab_size = vocab.shape[0]
    feats = np.zeros((len(image_paths), vocab_size), dtype=np.float32)

    for idx, path in enumerate(image_paths):
        img = load_image_gray(path)
        descriptors = _extract_local_descriptors(img, step=6)
        if descriptors.shape[0] == 0:
            continue

        if vlfeat is not None:
            assignments = vlfeat.kmeans.kmeans_quantize(descriptors, vocab)
        else:
            dists = sklearn_pairwise.pairwise_distances(descriptors, vocab, metric='euclidean')
            assignments = np.argmin(dists, axis=1)

        hist, _ = np.histogram(assignments, bins=np.arange(vocab_size + 1), density=False)
        hist = hist.astype(np.float32)
        hsum = np.sum(hist)
        if hsum > 0:
            hist /= hsum
        feats[idx, :] = hist

    return feats


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, metric='euclidean'):
    """
    Predict labels with k-NN (k=5 by default).
    """
    test_labels = []
    k = 5

    if metric == 'chi2':
        eps = 1e-10
        n_test = test_image_feats.shape[0]
        n_train = train_image_feats.shape[0]
        dists = np.zeros((n_test, n_train), dtype=np.float32)
        for i in range(n_test):
            x = test_image_feats[i][None, :]
            num = (train_image_feats - x) ** 2
            den = train_image_feats + x + eps
            dists[i] = 0.5 * np.sum(num / den, axis=1)
    else:
        dists = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats, metric=metric)

    for i in range(dists.shape[0]):
        nn_idx = np.argsort(dists[i])[:k]
        nn_labels = [train_labels[j] for j in nn_idx]
        values, counts = np.unique(nn_labels, return_counts=True)
        test_labels.append(values[np.argmax(counts)])

    return test_labels


def svm_classify(train_image_feats, train_labels, test_image_feats):
    """
    Train one-vs-all linear SVMs and predict by max confidence.
    """
    categories = sorted(list(set(train_labels)))
    classifiers = {
        cat: LinearSVC(random_state=0, tol=1e-4, loss='hinge', C=5.0, max_iter=20000)
        for cat in categories
    }

    y_train = np.array(train_labels)
    for cat in categories:
        y_binary = np.where(y_train == cat, 1, -1)
        classifiers[cat].fit(train_image_feats, y_binary)

    scores = np.zeros((test_image_feats.shape[0], len(categories)), dtype=np.float32)
    for cidx, cat in enumerate(categories):
        scores[:, cidx] = classifiers[cat].decision_function(test_image_feats)

    best = np.argmax(scores, axis=1)
    test_labels = [categories[i] for i in best]
    return test_labels
