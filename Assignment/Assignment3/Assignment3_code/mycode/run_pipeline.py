import os
import os.path as osp
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from student_code_12311805 import get_tiny_images, build_vocabulary, get_bags_of_sifts, nearest_neighbor_classify, svm_classify
from utils import load_image, get_image_paths

OUTDIR = 'outputs'
os.makedirs(OUTDIR, exist_ok=True)

# Parameters (adjustable)
categories = ['Kitchen','Store','Bedroom','LivingRoom','Office','Industrial','Suburb',
              'InsideCity','TallBuilding','Street','Highway','OpenCountry','Coast',
              'Mountain','Forest']
num_train_per_cat = 100
vocab_filename = 'vocab.pkl'
vocab_size = 200

# Paths
data_path = osp.join('..', 'data')
train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path, categories, num_train_per_cat)
print('Loaded', len(train_image_paths), 'train and', len(test_image_paths), 'test images')

# Tiny images
print('Computing tiny image features...')
train_tiny = get_tiny_images(train_image_paths)
test_tiny = get_tiny_images(test_image_paths)

# Nearest neighbor (tiny)
print('Running NN on tiny images...')
preds_tiny_nn = nearest_neighbor_classify(train_tiny, train_labels, test_tiny)
cm_tiny = confusion_matrix([categories.index(l) for l in test_labels], [categories.index(p) for p in preds_tiny_nn], labels=list(range(len(categories))))
acc_tiny = np.mean(np.diag(cm_tiny) / cm_tiny.sum(axis=1))
print('Tiny NN accuracy (mean diag):', acc_tiny)

plt.figure(figsize=(10,8))
plt.imshow(cm_tiny.astype(float)/cm_tiny.sum(axis=1)[:,None], cmap='viridis')
plt.title(f'Tiny-NN confusion (mean diag={acc_tiny:.3f})')
plt.colorbar()
plt.tight_layout()
plt.savefig(osp.join(OUTDIR,'confusion_tiny_nn.png'))
plt.close()

# Build or load vocab
if not osp.isfile(vocab_filename):
    print('Building visual vocabulary (this may take a while)')
    vocab = build_vocabulary(train_image_paths, vocab_size)
    with open(vocab_filename,'wb') as f:
        pickle.dump(vocab, f)
    print('Saved',vocab_filename)
else:
    print('Loading existing vocab')
    with open(vocab_filename,'rb') as f:
        vocab = pickle.load(f)

# Bag of SIFT
print('Computing bag-of-sift features...')
train_bags = get_bags_of_sifts(train_image_paths, vocab_filename)
test_bags = get_bags_of_sifts(test_image_paths, vocab_filename)

# NN on bags
print('Running NN on bag-of-sift...')
preds_bag_nn = nearest_neighbor_classify(train_bags, train_labels, test_bags, metric='chi2')
cm_bag_nn = confusion_matrix([categories.index(l) for l in test_labels], [categories.index(p) for p in preds_bag_nn], labels=list(range(len(categories))))
acc_bag_nn = np.mean(np.diag(cm_bag_nn) / cm_bag_nn.sum(axis=1))
print('Bag-NN accuracy (mean diag):', acc_bag_nn)
plt.figure(figsize=(10,8))
plt.imshow(cm_bag_nn.astype(float)/cm_bag_nn.sum(axis=1)[:,None], cmap='viridis')
plt.title(f'Bag-NN confusion (mean diag={acc_bag_nn:.3f})')
plt.colorbar()
plt.tight_layout()
plt.savefig(osp.join(OUTDIR,'confusion_bag_nn.png'))
plt.close()

# SVM on bags
print('Running SVM on bag-of-sift (this may take time)')
preds_bag_svm = svm_classify(train_bags, train_labels, test_bags)
cm_bag_svm = confusion_matrix([categories.index(l) for l in test_labels], [categories.index(p) for p in preds_bag_svm], labels=list(range(len(categories))))
acc_bag_svm = np.mean(np.diag(cm_bag_svm) / cm_bag_svm.sum(axis=1))
print('Bag-SVM accuracy (mean diag):', acc_bag_svm)
plt.figure(figsize=(10,8))
plt.imshow(cm_bag_svm.astype(float)/cm_bag_svm.sum(axis=1)[:,None], cmap='viridis')
plt.title(f'Bag-SVM confusion (mean diag={acc_bag_svm:.3f})')
plt.colorbar()
plt.tight_layout()
plt.savefig(osp.join(OUTDIR,'confusion_bag_svm.png'))
plt.close()

# Save a few example result images (first 12 test images)
from PIL import Image, ImageDraw, ImageFont
EXAMPLES = 12
os.makedirs(osp.join(OUTDIR,'examples'), exist_ok=True)
for i, img_path in enumerate(test_image_paths[:EXAMPLES]):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    # resize for consistent display
    maxw = 300
    if w>maxw:
        img = img.resize((maxw, int(h*maxw/w)))
    draw = ImageDraw.Draw(img)
    true = test_labels[i]
    nn_pred = preds_bag_nn[i]
    svm_pred = preds_bag_svm[i]
    text = f'True: {true}\nNN: {nn_pred}\nSVM: {svm_pred}'
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((5,5), text, fill=(255,0,0), font=font)
    img.save(osp.join(OUTDIR,'examples', f'example_{i}_{true}.jpg'))

print('Saved outputs to', OUTDIR)

if __name__=='__main__':
    pass
