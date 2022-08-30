# works good!
import pickle
import numpy as np
np.random.seed(43)
import pandas as pd
from collections import defaultdict, Counter
import random
from tqdm import tqdm
from scipy.stats import binom_test
from scipy.stats import wilcoxon

from sklearn.model_selection import StratifiedKFold

from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# 0. Fix rng
import warnings
warnings.filterwarnings('ignore')
rng = np.random.default_rng(44)

# 1. Read embeddings 
# with open("is_correct.pickle", 'rb') as file:
#     is_correct = pickle.load(file)
    
with open("../all_pose_embeddings.pickle", 'rb') as file:
    all_pose_embeddings = pickle.load(file)
    
with open("../all_texts.pickle", 'rb') as file:
    all_texts = pickle.load(file)
    
# with open("text_lengths.pickle", 'rb') as file:
#     text_lengths = pickle.load(file)
    
with open("../all_languages.pickle", 'rb') as file:
    all_languages = pickle.load(file)
    
with open("../all_vid_ids.pickle", 'rb') as file:
    all_vid_ids = pickle.load(file)
    
with open("../all_images.pickle", 'rb') as file:
    all_images = pickle.load(file)
    
with open("id2lang.pickle", 'rb') as file:
    id2lang = pickle.load(file)
    
# 2. Some spanish poses had bad artifacts, remove them
bad_frames = list()
for ix, pose in enumerate(all_images):
    for frame in pose:
        if frame[14] < -1.5:
            bad_frames.append(ix)
            break
            
all_pose_embeddings = np.delete(all_pose_embeddings, bad_frames, axis=0)
all_texts = np.delete(all_texts, bad_frames, axis=0)
# text_lengths = np.delete(text_lengths, bad_frames, axis=0)
all_languages = np.delete(all_languages, bad_frames, axis=0)
all_vid_ids = np.delete(all_vid_ids, bad_frames, axis=0)
all_images = np.delete(all_images, bad_frames, axis=0)
# is_correct = np.delete(is_correct, bad_frames, axis=0)

# 3. Functions
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def stratified_group_k_fold(X, y, groups, k, seed=44):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices
        
# 4. Predict language
en_good_indices = rng.choice(np.where(all_languages == 'en')[0], (all_languages == 'es').sum(), replace=False)
es_indices = np.where(all_languages == 'es')[0]

# concatenate to get the full new index with balance
idx_for_subsample = np.concatenate([en_good_indices, es_indices]) # for recovering idcs of original data

# now subsample
all_pose_embeddings_subsampled = all_pose_embeddings
all_languages_subsampled = all_languages
# is_correct_subsampled = is_correct
all_vid_ids_subsampled = all_vid_ids
all_texts_subsampled = pd.Series(all_texts)
all_images_subsampled = all_images

overall = list()

for hjk in range(100):
    perf = list()
    scores = []
    raw_poses_scores = []
    baseline_scores = []
    precisions_recalls = []
    precisions_recalls_raw = []
    p_values = []
    p_values_against_raw = []

    # just another way to prepare y
    y=[id2lang[i] for i in all_vid_ids]

    assert (np.array(y) == all_languages).all()
    y = (np.array(y) == 'en').astype('int')

    for train_fold_idcs, test_fold_idcs in stratified_group_k_fold(X=all_vid_ids, y=y, groups=all_vid_ids, k=10):    

        unique_train_ids = np.unique(all_vid_ids_subsampled[train_fold_idcs])
        unique_test_ids = np.unique(all_vid_ids_subsampled[test_fold_idcs])

        assert not np.isin(unique_test_ids, unique_train_ids).any()
        assert not np.isin(all_vid_ids_subsampled[train_fold_idcs], all_vid_ids_subsampled[test_fold_idcs]).any()

        # now we get from videos to clips
        train_idx = np.isin(np.array(range(len(all_vid_ids_subsampled))), train_fold_idcs) # np.isin(all_vid_ids_subsampled, train_item_ids)
        test_idx = np.isin(np.array(range(len(all_vid_ids_subsampled))), test_fold_idcs) # np.isin(all_vid_ids_subsampled, test_item_ids)

        assert not np.isin(all_vid_ids_subsampled[train_idx], all_vid_ids_subsampled[test_idx]).any()
#         print(f'Use {round(train_idx.sum()/len(train_idx), 4)} of the data for the training')

        # now select the training and validation data
        pose_embed_train = all_pose_embeddings_subsampled[train_idx]
        pose_embed_test = all_pose_embeddings_subsampled[test_idx]
        lang_train = all_languages_subsampled[train_idx]
        lang_test = all_languages_subsampled[test_idx]
        
#         is_correct_train = is_correct_subsampled[train_idx]
#         is_correct_test = is_correct_subsampled[test_idx]
        vid_ids_train = all_vid_ids_subsampled[train_idx]
        vid_ids_test = all_vid_ids_subsampled[test_idx]
        texts_train = all_texts_subsampled[train_idx]
        texts_test = all_texts_subsampled[test_idx]

        raw_poses_train = all_images_subsampled[train_idx]
        raw_poses_test = all_images_subsampled[test_idx]
        raw_poses_train = raw_poses_train.reshape(sum(train_idx), -1)
        raw_poses_test = raw_poses_test.reshape(sum(test_idx), -1)
        
        ########################### NEW SECTION _ SUBSAMPLING MOVED HERE ############################
        #
        #############################################################################################

        random_zero_indices_train = np.random.choice(np.where(lang_train == 'en')[0], min((lang_train == 'es').sum(), (lang_train == 'en').sum()), replace=False)
        random_zero_indices_test = np.random.choice(np.where(lang_test == 'en')[0], min((lang_test == 'es').sum(), (lang_test == 'en').sum()), replace=False)

        one_indices_train = np.where(lang_train == 'es')[0]
        one_indices_test = np.where(lang_test == 'es')[0]

        idx_for_subsample_train = np.concatenate([random_zero_indices_train, one_indices_train]) # for recovering idcs of original data
        idx_for_subsample_test = np.concatenate([random_zero_indices_test, one_indices_test]) # for recovering idcs of original data

        pose_embed_train = pose_embed_train[idx_for_subsample_train]
        pose_embed_test = pose_embed_test[idx_for_subsample_test]
#         sent_labels_train = sent_labels_train[idx_for_subsample_train]
#         sent_labels_test = sent_labels_test[idx_for_subsample_test]
        raw_poses_train = raw_poses_train[idx_for_subsample_train]
        raw_poses_test = raw_poses_test[idx_for_subsample_test]
        
        lang_train = lang_train[idx_for_subsample_train]
        lang_test = lang_test[idx_for_subsample_test]
        
#         n_obs_fold.append(sent_labels_train.shape[0])
#         if sum(sent_labels_test) == 0:
#             continue
        
#         print(pd.Series(lang_train).shape)
#         print(pd.Series(lang_test).shape)
        scaler = StandardScaler()
        scaler.fit(pose_embed_train)
#         print(pose_embed_train.shape[0])
        # fir the logistic regression with the default params
        lin_clf = LogisticRegression(random_state=42)
        lin_clf.fit(scaler.transform(pose_embed_train), lang_train)

        # predict on the test data
        preds = lin_clf.predict(scaler.transform(pose_embed_test))
        # accuracy - ok because perfectly balanced dataset
        acc = sum(preds == lang_test)/len(preds)
        scores.append(acc)

        precisions_recalls.append(precision_recall_fscore_support(preds, lang_test))

        # NOW tHE SAME BUT FOR RAW AND MAJORITY
        # 
        raw_scaler = StandardScaler()
        raw_scaler.fit(raw_poses_train)

        # fir the logistic regression with the default params
        lin_clf_raw = LogisticRegression(random_state=42)
        lin_clf_raw.fit(raw_scaler.transform(raw_poses_train), lang_train)

        # predict on the test data
        raw_preds = lin_clf_raw.predict(raw_scaler.transform(raw_poses_test))
        # accuracy - ok because perfectly balanced dataset

        raw_acc = sum(raw_preds == lang_test)/len(raw_preds)

        x = preds == lang_test
        y = np.ones_like(x) == (lang_test == 'en').astype('int8')

        p = wilcoxon(x=x.astype('int8'), y=y.astype('int8'), alternative = 'greater').pvalue
        p_values.append(p)

        y = raw_preds == lang_test
        p_against_raw = wilcoxon(x=x.astype('int8'), y=y.astype('int8'), alternative = 'greater').pvalue
        p_values_against_raw.append(p_against_raw)
        # print(binom_test((preds == lang_test).sum(), n=len(preds), p=raw_acc))

        raw_poses_scores.append(raw_acc)
        baseline_acc = pd.Series(lang_test).value_counts(normalize=True).iloc[0]
        baseline_scores.append(baseline_acc)
        precisions_recalls_raw.append(precision_recall_fscore_support(raw_preds, lang_test))
        
        overall.append({
            'embeddings': acc,
            'raw': raw_acc,
            'majority': baseline_acc,
            'run': hjk
        })

print(f"Embeddings: {pd.DataFrame(overall)['embeddings'].mean()}")
print(f"Raw: {pd.DataFrame(overall)['raw'].mean()}")
print(f"Majority: {pd.DataFrame(overall)['majority'].mean()}")

pd.DataFrame(overall).to_csv("lang_pred_task_10x100.csv", index=False)