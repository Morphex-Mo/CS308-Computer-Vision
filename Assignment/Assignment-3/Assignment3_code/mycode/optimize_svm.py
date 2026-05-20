import os
import os.path as osp
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from student_code_12311805 import get_bags_of_sifts
from utils import get_image_paths

OUTDIR = 'outputs/opt_svm'
os.makedirs(OUTDIR, exist_ok=True)

categories = ['Kitchen','Store','Bedroom','LivingRoom','Office','Industrial','Suburb',
              'InsideCity','TallBuilding','Street','Highway','OpenCountry','Coast',
              'Mountain','Forest']
num_train_per_cat = 100
vocab_filename = 'vocab.pkl'

# Paths
data_path = osp.join('..', 'data')
train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path, categories, num_train_per_cat)

print('Computing bag-of-sift features...')
train_bags = get_bags_of_sifts(train_image_paths, vocab_filename)
test_bags = get_bags_of_sifts(test_image_paths, vocab_filename)

# Tune C
C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
results = []

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(train_bags, train_labels))

X_train = train_bags[train_idx]
y_train = [train_labels[i] for i in train_idx]
X_val = train_bags[val_idx]
y_val = [train_labels[i] for i in val_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

for C in C_values:
    print(f'Testing C={C}...')
    svms = {}
    for cat in categories:
        y_binary = np.array([1 if l == cat else 0 for l in y_train])
        svm = LinearSVC(C=C, random_state=0, tol=1e-4, loss='hinge', max_iter=10000, dual=True)
        svm.fit(X_train_scaled, y_binary)
        svms[cat] = svm
    
    scores_val = []
    for cat in categories:
        scores_val.append(svms[cat].decision_function(X_val_scaled))
    scores_val = np.vstack(scores_val).T
    preds_val_idx = np.argmax(scores_val, axis=1)
    preds_val = [categories[i] for i in preds_val_idx]
    
    cm_val = confusion_matrix([categories.index(l) for l in y_val], [categories.index(p) for p in preds_val], labels=list(range(len(categories))))
    mean_diag_val = np.mean(np.diag(cm_val) / cm_val.sum(axis=1))
    print(f'Val mean diag for C={C}: {mean_diag_val}')
    results.append((C, mean_diag_val))

# pick best C
best_C, best_score = max(results, key=lambda x: x[1])
print('Best C on validation:', best_C, 'score=', best_score)

# retrain on full training set with best C and evaluate on test
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(train_bags)
X_test_scaled = scaler_full.transform(test_bags)

svms_full = {}
for cat in categories:
    y_binary = np.array([1 if l == cat else 0 for l in train_labels])
    svm = LinearSVC(C=best_C, random_state=0, tol=1e-4, loss='hinge', max_iter=10000, dual=True)
    svm.fit(X_full_scaled, y_binary)
    svms_full[cat] = svm

scores_test = []
for cat in categories:
    scores_test.append(svms_full[cat].decision_function(X_test_scaled))
scores_test = np.vstack(scores_test).T
preds_idx_test = np.argmax(scores_test, axis=1)
preds_test = [categories[i] for i in preds_idx_test]

cm_test = confusion_matrix([categories.index(l) for l in test_labels], [categories.index(p) for p in preds_test], labels=list(range(len(categories))))
mean_diag_test = np.mean(np.diag(cm_test) / cm_test.sum(axis=1))
print('Test mean diag with best C=', best_C, 'is', mean_diag_test)

# save results
np.savez(osp.join(OUTDIR, 'svm_search_results.npz'), results=results, best_C=best_C, best_val=best_score, test_score=mean_diag_test)

plt.figure(figsize=(10,8))
plt.imshow(cm_test.astype(float)/cm_test.sum(axis=1)[:,None], cmap='viridis')
plt.title(f'Bag-SVM tuned C={best_C} mean diag={mean_diag_test:.3f}')
plt.colorbar()
plt.tight_layout()
plt.savefig(osp.join(OUTDIR, f'confusion_bag_svm_tuned_C{best_C}.png'))
plt.close()

with open(osp.join(OUTDIR, 'svm_search_log.txt'), 'w') as f:
    f.write('results (C, val_mean_diag):\n')
    for r in results:
        f.write(str(r) + '\n')
    f.write('\nbest_C=' + str(best_C) + '\n')
    f.write('test_mean_diag=' + str(mean_diag_test) + '\n')

print('Saved optimization outputs to', OUTDIR)

if __name__ == '__main__':
    pass
