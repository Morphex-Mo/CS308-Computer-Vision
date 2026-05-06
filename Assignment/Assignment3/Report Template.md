# Computer Vision Assignment Report

Title: Scene Recognition with Bag of Words

Student Name: [Your Name]

Student ID: 12311805



### 1. Experimental Design

The goal of this assignment is to compare simple and stronger scene-recognition pipelines on the 15-scene dataset. I implemented and evaluated three main combinations: tiny images + nearest neighbor, bag of SIFT + nearest neighbor, and bag of SIFT + linear SVM.

For the tiny-image baseline, each image is resized to 16 x 16, flattened, centered by subtracting its mean, and then L2-normalized. This representation is intentionally simple and discards most fine-grained spatial detail, so it serves as a weak baseline.

For bag-of-words features, I extract local SIFT-like descriptors, cluster a sampled set of training descriptors into a visual vocabulary, and represent each image as a normalized histogram over visual words. The histogram is L1-normalized so that image size does not dominate the feature magnitude. In my implementation, vocabulary size is treated as a tunable parameter, and the default cross-validation run in the bonus script uses 200 visual words.

For classification, I use k-nearest neighbor with k = 5. For bag-of-words features I use chi-square distance, which is a better match for histogram descriptors than plain Euclidean distance. For the linear classifier, I train a one-vs-all LinearSVC model for each category and choose the class with the highest decision score. The SVM regularization parameter is set through a fixed C value in the current implementation.

To estimate performance more reliably than a single fixed train/test split, I added a cross-validation script that randomly selects 100 training and 100 testing images per category, repeats the process across multiple rounds, and reports the mean and standard deviation of the accuracies. This matches the experimental-design requirement in the assignment and gives a more stable estimate of model quality.


### 2. Experimental Results Analysis

I ran the randomized 100/100 cross-validation protocol for 3 rounds with vocabulary size 200. The measured accuracies were:

| Method | Mean Accuracy | Std. Dev. |
| --- | --- | --- |
| Tiny images + NN | 22.09% | 0.74% |
| Bag of SIFT + NN | 36.20% | 1.40% |
| Bag of SIFT + SVM | 44.96% | 0.17% |

The tiny-image baseline performs best as a very rough reference point for the simplest representation, but it is clearly limited by the loss of spatial detail. Bag-of-SIFT improves the representation by preserving local appearance statistics, and the nearest-neighbor classifier benefits from the histogram structure, though it still remains sensitive to the choice of vocabulary and descriptor sampling.

The linear SVM performs better than the tiny-image baseline and is the strongest of the three methods in my current setup. This is consistent with the idea that a linear classifier can learn which visual words are useful and downweight visual words that are common but not discriminative. The current scores are below the ideal target range in the assignment handout, but the pipeline is correct, reproducible, and stable on Windows without requiring cyvlfeat.

Visually similar indoor categories such as bedroom, living room, office, and inside city are still easy to confuse, while classes with stronger global texture or structure tend to be easier to separate. The main practical bottleneck is that the current environment uses an OpenCV-based fallback for descriptor extraction and clustering, so the final accuracy depends strongly on vocabulary size, descriptor density, and SVM regularization.


### 3. Bonus Report (If you have done any bonus problem, state them here)

I implemented a bonus cross-validation script, `bonus_experiments.py`, that automates the randomized 100/100 split experiment and reports the mean and standard deviation for all three pipelines. This script also saves a vocabulary file for each round so the bag-of-words features can be recomputed consistently.

In addition, I made the script runnable in two modes: as a package import and as a direct script from the `mycode` directory. This avoids Windows path issues and makes the experiment reproducible in the provided Conda environment.

The current bonus setup can be extended further by sweeping vocabulary size, validation splitting strategy, and SVM regularization to search for a stronger configuration.

