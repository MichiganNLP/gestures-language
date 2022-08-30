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
with open("is_correct.pickle", 'rb') as file:
    is_correct = pickle.load(file)
    
with open("all_pose_embeddings.pickle", 'rb') as file:
    all_pose_embeddings = pickle.load(file)
    
with open("all_texts.pickle", 'rb') as file:
    all_texts = pickle.load(file)
    
with open("text_lengths.pickle", 'rb') as file:
    text_lengths = pickle.load(file)
    
with open("all_languages.pickle", 'rb') as file:
    all_languages = pickle.load(file)
    
with open("all_vid_ids.pickle", 'rb') as file:
    all_vid_ids = pickle.load(file)
    
with open("all_images.pickle", 'rb') as file:
    all_images = pickle.load(file)
    
with open("id2lang_train.pickle", 'rb') as file:
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
text_lengths = np.delete(text_lengths, bad_frames, axis=0)
all_languages = np.delete(all_languages, bad_frames, axis=0)
all_vid_ids = np.delete(all_vid_ids, bad_frames, axis=0)
all_images = np.delete(all_images, bad_frames, axis=0)
is_correct = np.delete(is_correct, bad_frames, axis=0)

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
        
# 4. Predict liwc category
for_lang = 'es'
liwc_path = "/home/public/DICTIONARIES/LingEthnography/LIWC/LIWC.all.txt"
spanish_liwc_path = "/home/abzaliev/LIWC.Spanish.all.txt"
dictionary = pd.read_csv(liwc_path, header=None, names=('word', 'category'))
spanish_dict = pd.read_csv(spanish_liwc_path, header=None, names=('word', 'category'))
spanish_dict['category'] = spanish_dict['category'].apply(lambda x: x.strip())
words_cleaned = dictionary['word'].apply(lambda x: x.replace('*', '').strip())
words_cleaned_spanish = spanish_dict['word'].apply(lambda x: x.replace('*', '').strip())

d = dict(zip(words_cleaned, dictionary['category']))
# d_spanish = dict(zip(words_cleaned_spanish, spanish_dict['category']))
d_spanish = pd.DataFrame(zip(words_cleaned_spanish, spanish_dict['category'])).groupby(0)[1].unique().to_dict()

unique_categories = set(dictionary['category'].unique())

idx_for_subsample = np.where(all_languages == for_lang)[0]

# now subsample
all_pose_embeddings_subsampled_gi = all_pose_embeddings[idx_for_subsample]
all_languages_subsampled_gi = all_languages[idx_for_subsample]
is_correct_subsampled_gi = is_correct[idx_for_subsample]
all_vid_ids_subsampled_gi = all_vid_ids[idx_for_subsample]
all_texts_subsampled_gi = pd.Series(all_texts)[idx_for_subsample]
all_images_subsampled_gi = all_images[idx_for_subsample]

liwc_categories = list()

for u, la in tqdm(zip(all_texts_subsampled_gi, all_languages_subsampled_gi)):
    single_cat = set()
    for w in u.split():
        if la == 'en':
            raise
        elif la == 'es':
            c = d_spanish.get(w)
        if c is not None:
            for sev_c in c:
                single_cat.add(sev_c)

    liwc_categories.append(single_cat)


all_performances = list()
for hjk in range(10):
    perf = list()
    for cat in spanish_dict['category'].unique():
        y = [cat in c for c in liwc_categories]
        y = np.array(y).astype('int')

        if sum(y) < 30:
            continue

        scores = []
        raw_poses_scores = []
        baseline_scores = []
        precisions_recalls = []
        precisions_recalls_raw = []

        for _ in range(1):
    #         zero_indices = np.random.choice(np.where(y == 0)[0], min((y == 1).sum(), (y == 0).sum()), replace=False)
    #         one_indices = np.where(y == 1)[0]
    #         idx_for_subsample = np.concatenate([zero_indices, one_indices]) # for recovering idcs of original data

            # now subsample
            all_pose_embeddings_subsampled = all_pose_embeddings_subsampled_gi #[idx_for_subsample]
            all_languages_subsampled = all_languages_subsampled_gi #[idx_for_subsample]
            is_correct_subsampled = is_correct_subsampled_gi #[idx_for_subsample]
            all_vid_ids_subsampled = all_vid_ids_subsampled_gi #[idx_for_subsample]
            all_texts_subsampled = pd.Series(all_texts_subsampled_gi) #[idx_for_subsample]
            all_images_subsampled = all_images_subsampled_gi #[idx_for_subsample]
            y_l = y#[idx_for_subsample]

            # see they are balanced # 1 for english, 2 if both, 0 if nothing
            if len(pd.Series(all_languages_subsampled).value_counts().unique()) == 0:
                continue

            n_obs_fold = list()
            p_values_fold = list()

            for train_fold_idcs, test_fold_idcs in stratified_group_k_fold(X=all_vid_ids_subsampled, y=y_l, groups=all_vid_ids_subsampled, k=30): #kfold.split(X=all_vid_ids_subsampled, y=y ):    

                unique_train_ids = np.unique(all_vid_ids_subsampled[train_fold_idcs])
                unique_test_ids = np.unique(all_vid_ids_subsampled[test_fold_idcs])

                assert not np.isin(unique_test_ids, unique_train_ids).any()
                assert not np.isin(all_vid_ids_subsampled[train_fold_idcs], all_vid_ids_subsampled[test_fold_idcs]).any()

                # now we get from videos to clips
                train_idx = np.isin(np.array(range(len(all_vid_ids_subsampled))), train_fold_idcs) # np.isin(all_vid_ids_subsampled, train_item_ids)
                test_idx = np.isin(np.array(range(len(all_vid_ids_subsampled))), test_fold_idcs) # np.isin(all_vid_ids_subsampled, test_item_ids)

                assert not np.isin(all_vid_ids_subsampled[train_idx], all_vid_ids_subsampled[test_idx]).any()
    #             print(f'Use {round(train_idx.sum()/len(train_idx), 4)} of the data for the training')

                # now select the training and validation data
                pose_embed_train = all_pose_embeddings_subsampled[train_idx]
                pose_embed_test = all_pose_embeddings_subsampled[test_idx]

                if pose_embed_train.shape[0] == 0 or pose_embed_test.shape[0] == 0:
                    continue

                vid_ids_train = all_vid_ids_subsampled[train_idx]
                vid_ids_test = all_vid_ids_subsampled[test_idx]

                sent_labels_train = np.array(y_l)[train_idx]
                sent_labels_test = np.array(y_l)[test_idx]

                raw_poses_train = all_images_subsampled[train_idx]
                raw_poses_test = all_images_subsampled[test_idx]
                raw_poses_train = raw_poses_train.reshape(sum(train_idx), -1)
                raw_poses_test = raw_poses_test.reshape(sum(test_idx), -1)

                ########################### NEW SECTION _ SUBSAMPLING MOVED HERE ############################
                #
                #############################################################################################

                random_zero_indices_train = np.random.choice(np.where(sent_labels_train == 0)[0], min((sent_labels_train == 1).sum(), (sent_labels_train == 0).sum()), replace=False)
                random_zero_indices_test = np.random.choice(np.where(sent_labels_test == 0)[0], min((sent_labels_test == 1).sum(), (sent_labels_test == 0).sum()), replace=False)

                one_indices_train = np.where(sent_labels_train == 1)[0]
                one_indices_test = np.where(sent_labels_test == 1)[0]

                idx_for_subsample_train = np.concatenate([random_zero_indices_train, one_indices_train]) # for recovering idcs of original data
                idx_for_subsample_test = np.concatenate([random_zero_indices_test, one_indices_test]) # for recovering idcs of original data

                pose_embed_train = pose_embed_train[idx_for_subsample_train]
                pose_embed_test = pose_embed_test[idx_for_subsample_test]
                sent_labels_train = sent_labels_train[idx_for_subsample_train]
                sent_labels_test = sent_labels_test[idx_for_subsample_test]
                raw_poses_train = raw_poses_train[idx_for_subsample_train]
                raw_poses_test = raw_poses_test[idx_for_subsample_test]

                n_obs_fold.append(sent_labels_train.shape[0])
                if sum(sent_labels_test) == 0:
                    continue

                # print(pd.Series(y).value_counts())
                scaler = StandardScaler()
                scaler.fit(pose_embed_train)

                # fir the logistic regression with the default params
                lin_clf = LogisticRegression(random_state=42)
                try:
                    lin_clf.fit(scaler.transform(pose_embed_train), sent_labels_train)
                except:
                    continue

                # predict on the test data
                preds = lin_clf.predict(scaler.transform(pose_embed_test))

                # accuracy - ok because perfectly balanced dataset
                acc = sum(preds == sent_labels_test)/len(preds) # 1_score(preds, sent_labels_test)#
                scores.append(acc)

                precisions_recalls.append(precision_recall_fscore_support(preds, sent_labels_test))

                # NOW THE SAME BUT FOR RAW AND MAJORITY
                raw_scaler = StandardScaler()
                raw_scaler.fit(raw_poses_train)

                # fir the logistic regression with the default params
                lin_clf_raw = LogisticRegression(random_state=42)
                lin_clf_raw.fit(raw_scaler.transform(raw_poses_train), sent_labels_train)

                # predict on the test data
                raw_preds = lin_clf_raw.predict(raw_scaler.transform(raw_poses_test))

                # accuracy - ok because perfectly balanced dataset
                raw_acc = sum(raw_preds == sent_labels_test)/len(raw_preds)
    #             raw_acc = f1_score(raw_preds, sent_labels_test)

                raw_poses_scores.append(raw_acc)

                baseline_acc = sum(sent_labels_test == 1)/len(sent_labels_test)
    #             baseline_acc = f1_score([pd.Series(sent_labels_test).value_counts(normalize=True).index[0]] * len(sent_labels_test), sent_labels_test)
                baseline_scores.append(baseline_acc)
                precisions_recalls_raw.append(precision_recall_fscore_support(raw_preds, sent_labels_test))

                #   x = preds == sent_labels_test
                #   y_local = np.ones_like(x) == sent_labels_test
                #   p_value = wilcoxon(x=x.astype('int8'), y=y.astype('int8'), alternative = 'greater').pvalue

                p_value = binom_test((preds == sent_labels_test).sum(), n=len(preds), p=pd.Series(sent_labels_test).value_counts(normalize=True).iloc[0])
                p_values_fold.append(p_value)

                perf.append({
                    'cat': cat,
                    'embeddings': acc,
                    'raw': raw_acc,
                    'majority': baseline_acc,
                    'n_obs': sent_labels_train.shape[0],
            #             'p-value': p_values_fold,
#                     'wilcoxon_against_baseline': wilcoxon(x=scores, y=baseline_scores, alternative = 'greater')[1],
#                     'wilcoxon_against_raw': wilcoxon(x=scores, y=raw_poses_scores, alternative = 'greater')[1],
                    'run': hjk
                })

        print(f'Category: {cat}')
        print(f"Embeddings: {np.mean(scores)}")
        print(f"Raw: {np.mean(raw_poses_scores)}")
        print(f"Majority: {np.mean(baseline_scores)}")
    all_performances.append(pd.DataFrame(perf))
# v2 means added proper categories, not just dictionary
pd.concat(all_performances).to_csv("liwc_pred_task_es_v2.csv", index=False)