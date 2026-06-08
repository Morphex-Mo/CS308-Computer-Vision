import cv2
import numpy as np
import os.path as osp
from glob import glob
from sklearn.svm import LinearSVC
import cyvlfeat as vlfeat
from utils import *


def get_positive_features(train_path_pos, feature_params):
        """
        This function should return all positive training examples (faces) from
        36x36 images in 'train_path_pos'. Each face should be converted into a
        HoG template according to 'feature_params'.
        """
        win_size = feature_params.get('template_size', 36)
        cell_size = feature_params.get('hog_cell_size', 6)

        positive_files = glob(osp.join(train_path_pos, '*.jpg'))
        n_cell = int(np.ceil(win_size / cell_size))
        feat_dim = n_cell * n_cell * 31

        feats = np.zeros((len(positive_files) * 2, feat_dim), dtype=np.float32)
        feat_idx = 0
        for file_path in positive_files:
                im = load_image_gray(file_path)
                hog = np.asarray(vlfeat.hog.hog(im, cell_size))
                feats[feat_idx, :] = hog.ravel()
                feat_idx += 1

                im_flipped = np.fliplr(im)
                hog_flipped = np.asarray(vlfeat.hog.hog(im_flipped, cell_size))
                feats[feat_idx, :] = hog_flipped.ravel()
                feat_idx += 1

        return feats[:feat_idx]


def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
        """
        Randomly sample negative training examples from non-face scenes.
        """
        win_size = feature_params.get('template_size', 36)
        cell_size = feature_params.get('hog_cell_size', 6)

        negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))
        n_cell = int(np.ceil(win_size / cell_size))
        feat_dim = n_cell * n_cell * 31

        feats = []
        for file_path in negative_files:
                if len(feats) >= num_samples:
                        break

                im = load_image_gray(file_path)
                h, w = im.shape
                if h < win_size or w < win_size:
                        continue

                hog = np.asarray(vlfeat.hog.hog(im, cell_size))
                hog_h, hog_w = hog.shape[:2]
                max_y = hog_h - n_cell + 1
                max_x = hog_w - n_cell + 1
                if max_y <= 0 or max_x <= 0:
                        continue

                samples_per_image = min(10, num_samples - len(feats))
                ys = np.random.randint(0, max_y, samples_per_image)
                xs = np.random.randint(0, max_x, samples_per_image)
                for yy, xx in zip(ys, xs):
                        feats.append(hog[yy:yy + n_cell, xx:xx + n_cell, :].ravel())

        if not feats:
                return np.zeros((0, feat_dim), dtype=np.float32)

        feats = np.vstack(feats).astype(np.float32)
        return feats[:num_samples]


def train_classifier(features_pos, features_neg, C):
        """
        Train a linear SVM classifier.
        """
        X = np.vstack((features_pos, features_neg))
        y = np.hstack((np.ones(len(features_pos)), -np.ones(len(features_neg))))

        svm = LinearSVC(C=C, max_iter=10000, dual=False)
        svm.fit(X, y)
        return svm


def mine_hard_negs(non_face_scn_path, svm, feature_params):
        """
        Mine false-positive windows from non-face scenes.
        """
        win_size = feature_params.get('template_size', 36)
        cell_size = feature_params.get('hog_cell_size', 6)

        negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))
        n_cell = int(np.ceil(win_size / cell_size))
        feat_dim = n_cell * n_cell * 31

        hard_feats = []
        for file_path in negative_files:
                im = load_image_gray(file_path)
                h, w = im.shape
                if h < win_size or w < win_size:
                        continue

                hog = np.asarray(vlfeat.hog.hog(im, cell_size))
                hog_h, hog_w = hog.shape[:2]
                max_y = hog_h - n_cell + 1
                max_x = hog_w - n_cell + 1
                if max_y <= 0 or max_x <= 0:
                        continue

                for yy in range(max_y):
                        for xx in range(max_x):
                                feat = hog[yy:yy + n_cell, xx:xx + n_cell, :].ravel().reshape(1, -1)
                                score = float(svm.decision_function(feat)[0])
                                if score > 0:
                                        hard_feats.append(feat.ravel())

        if not hard_feats:
                return np.zeros((0, feat_dim), dtype=np.float32)

        return np.vstack(hard_feats).astype(np.float32)


def run_detector(test_scn_path, svm, feature_params, verbose=False):
        """
        Run a multi-scale sliding-window face detector.
        """
        im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
        bboxes = np.empty((0, 4))
        confidences = np.empty(0)
        image_ids = []

        topk = feature_params.get('topk', 15)
        win_size = feature_params.get('template_size', 36)
        cell_size = feature_params.get('hog_cell_size', 6)
        scale_factor = feature_params.get('scale_factor', 0.65)
        conf_threshold = feature_params.get('conf_threshold', -2.0)
        template_size = int(win_size / cell_size)
        pre_nms_topk = feature_params.get('pre_nms_topk', topk)

        for im_filename in im_filenames:
                print('Detecting faces in {:s}'.format(im_filename))
                im = load_image_gray(im_filename)
                im_id = osp.split(im_filename)[-1]
                im_shape = im.shape

                cur_bboxes = []
                cur_confidences = []

                scale = 1.0
                im_scaled = im.copy()
                while True:
                        h_s, w_s = im_scaled.shape
                        if h_s < win_size or w_s < win_size:
                                break

                        hog = np.asarray(vlfeat.hog.hog(im_scaled, cell_size))
                        hog_h, hog_w = hog.shape[:2]
                        max_y = hog_h - template_size + 1
                        max_x = hog_w - template_size + 1
                        scale_y = h_s / float(im.shape[0])
                        scale_x = w_s / float(im.shape[1])

                        if max_y > 0 and max_x > 0:
                                for yy in range(max_y):
                                        for xx in range(max_x):
                                                feat = hog[yy:yy + template_size, xx:xx + template_size, :].ravel().reshape(1, -1)
                                                score = float(svm.decision_function(feat)[0])
                                                if score >= conf_threshold:
                                                        x_min = int(np.round((xx * cell_size) / scale_x))
                                                        y_min = int(np.round((yy * cell_size) / scale_y))
                                                        x_max = int(np.round(((xx + template_size) * cell_size) / scale_x))
                                                        y_max = int(np.round(((yy + template_size) * cell_size) / scale_y))
                                                        cur_bboxes.append([x_min, y_min, x_max, y_max])
                                                        cur_confidences.append(score)

                        scale *= scale_factor
                        new_w = max(1, int(np.round(im.shape[1] * scale)))
                        new_h = max(1, int(np.round(im.shape[0] * scale)))
                        if new_w < win_size or new_h < win_size:
                                break
                        im_scaled = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

                if len(cur_confidences) == 0:
                        cur_bboxes = np.zeros((0, 4), dtype=int)
                        cur_confidences = np.array([])
                else:
                        cur_bboxes = np.vstack(cur_bboxes)
                        cur_confidences = np.array(cur_confidences)

                        if len(cur_confidences) > pre_nms_topk:
                                idsort = np.argsort(-cur_confidences)[:pre_nms_topk]
                                cur_bboxes = cur_bboxes[idsort]
                                cur_confidences = cur_confidences[idsort]

                        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences, im_shape, verbose=verbose)

                        print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))
                        cur_bboxes = cur_bboxes[is_valid_bbox]
                        cur_confidences = cur_confidences[is_valid_bbox]

                if len(cur_confidences) == 0:
                        print('NMS done, 0 detections passed')

                bboxes = np.vstack((bboxes, cur_bboxes))
                confidences = np.hstack((confidences, cur_confidences))
                image_ids.extend([im_id] * len(cur_confidences))

        return bboxes, confidences, image_ids
