import argparse
import os
import pickle
from glob import glob

import numpy as np

try:
    import Assignment3_code.mycode.student_code_12311805 as sc
except ModuleNotFoundError:
    import student_code_12311805 as sc


CATEGORIES = [
    'Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial',
    'InsideCity', 'Kitchen', 'LivingRoom', 'Mountain', 'Office',
    'OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding'
]


def collect_all_paths(data_root, fmt='jpg'):
    by_cat = {}
    for cat in CATEGORIES:
        train_p = glob(os.path.join(data_root, 'train', cat, f'*.{fmt}'))
        test_p = glob(os.path.join(data_root, 'test', cat, f'*.{fmt}'))
        by_cat[cat] = train_p + test_p
    return by_cat


def random_split_100_100(by_cat, rng, n_train=100, n_test=100):
    train_paths, test_paths = [], []
    train_labels, test_labels = [], []

    for cat in CATEGORIES:
        all_paths = np.array(by_cat[cat])
        perm = rng.permutation(len(all_paths))
        all_paths = all_paths[perm]

        tr = all_paths[:n_train]
        te = all_paths[n_train:n_train + n_test]

        train_paths.extend(tr.tolist())
        test_paths.extend(te.tolist())
        train_labels.extend([cat] * len(tr))
        test_labels.extend([cat] * len(te))

    return train_paths, test_paths, train_labels, test_labels


def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(y_true == y_pred))


def run_three_combinations(train_paths, test_paths, train_labels, test_labels, vocab_size, vocab_file):
    tiny_train = sc.get_tiny_images(train_paths)
    tiny_test = sc.get_tiny_images(test_paths)
    tiny_pred = sc.nearest_neighbor_classify(tiny_train, train_labels, tiny_test)
    tiny_acc = accuracy(test_labels, tiny_pred)

    vocab = sc.build_vocabulary(train_paths, vocab_size=vocab_size)
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)

    bow_train = sc.get_bags_of_sifts(train_paths, vocab_file)
    bow_test = sc.get_bags_of_sifts(test_paths, vocab_file)

    bow_nn_pred = sc.nearest_neighbor_classify(bow_train, train_labels, bow_test, metric='chi2')
    bow_nn_acc = accuracy(test_labels, bow_nn_pred)

    bow_svm_pred = sc.svm_classify(bow_train, train_labels, bow_test)
    bow_svm_acc = accuracy(test_labels, bow_svm_pred)

    return tiny_acc, bow_nn_acc, bow_svm_acc


def experiment_cross_validation(data_root, rounds, vocab_size, seed):
    rng = np.random.default_rng(seed)
    by_cat = collect_all_paths(data_root)

    tiny_scores, bow_nn_scores, bow_svm_scores = [], [], []

    for i in range(rounds):
        trp, tep, trl, tel = random_split_100_100(by_cat, rng)
        vocab_file = f'vocab_cv_round_{i}.pkl'
        t, bnn, bsvm = run_three_combinations(trp, tep, trl, tel, vocab_size, vocab_file)
        tiny_scores.append(t)
        bow_nn_scores.append(bnn)
        bow_svm_scores.append(bsvm)
        print(f'[Round {i + 1}] tiny={t:.4f}, bow+nn={bnn:.4f}, bow+svm={bsvm:.4f}')

    print('\n[Cross Validation Summary]')
    print(f'tiny+nn: mean={np.mean(tiny_scores):.4f}, std={np.std(tiny_scores):.4f}')
    print(f'bow+nn: mean={np.mean(bow_nn_scores):.4f}, std={np.std(bow_nn_scores):.4f}')
    print(f'bow+svm: mean={np.mean(bow_svm_scores):.4f}, std={np.std(bow_svm_scores):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--vocab_size', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    experiment_cross_validation(
        data_root=args.data_root,
        rounds=args.rounds,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
