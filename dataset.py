# heavily based on https://github.com/youngwoo-yoon/Co-Speech_Gesture_Generation, with modifications
from transformers import XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import logging
import lmdb as lmdb
import numpy as np
from collections import defaultdict
import pyarrow
import os

from scipy.interpolate import interp1d
import math

def normalize_skeleton(data, resize_factor=None):
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    anchor_pt = (data[1 * 2], data[1 * 2 + 1])  # neck
    if resize_factor is None:
        neck_height = float(abs(data[1] - data[1 * 2 + 1]))
        shoulder_length = distance(data[1 * 2], data[1 * 2 + 1], data[2 * 2], data[2 * 2 + 1]) + \
                          distance(data[1 * 2], data[1 * 2 + 1], data[5 * 2], data[5 * 2 + 1])
        resized_neck_height = neck_height / float(shoulder_length)
        if resized_neck_height > 0.6:
            resize_factor = shoulder_length * resized_neck_height / 0.6
        else:
            resize_factor = shoulder_length

    normalized_data = data.copy()
    for i in range(0, len(data), 2):
        normalized_data[i] = (data[i] - anchor_pt[0]) / resize_factor
        normalized_data[i + 1] = (data[i + 1] - anchor_pt[1]) / resize_factor
    
    return normalized_data, resize_factor


class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = np.array(skeletons)
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.has_missing_frames():
            # works well and does what it should
            self.skeletons = []
            self.filtering_message = "too many missing frames"

        # fill missing joints
        if self.skeletons != []:
            self.fill_missing_joints()
            if self.skeletons is None or np.isnan(self.skeletons).any():
                self.filtering_message = "failed to fill missing joints"
                self.skeletons = []
            

        # filtering
        if self.skeletons != []:
            if self.is_static():
                self.skeletons = []
                self.filtering_message = "static motion"
            elif self.has_jumping_joint():
                self.skeletons = []
                self.filtering_message = "jumping joint"
        
        # preprocessing
        if self.skeletons != []:

            self.smooth_motion()

            is_side_view = False
            self.skeletons = self.skeletons.tolist()
            for i, frame in enumerate(self.skeletons):
                self.skeletons[i], _ = normalize_skeleton(frame) # translate and scale

                # assertion: missing joints
                assert not np.isnan(self.skeletons[i]).any()

                # side view check
                if (self.skeletons[i][0] < min(self.skeletons[i][2 * 2],
                                               self.skeletons[i][5 * 2]) or
                    self.skeletons[i][0] > max(self.skeletons[i][2 * 2],
                                               self.skeletons[i][5 * 2])):
                    is_side_view = True
                    break

            if len(self.skeletons) == 0 or is_side_view:
                self.filtering_message = "sideview"
                self.skeletons = []

        return self.skeletons, self.filtering_message

    def is_static(self, verbose=False):
        def joint_angle(p1, p2, p3):
            v1 = p1 - p2
            v2 = p3 - p2
            ang1 = np.arctan2(*v1[::-1])
            ang2 = np.arctan2(*v2[::-1])
            return np.rad2deg((ang1 - ang2) % (2 * np.pi))

        def get_joint_variance(skeleton, index1, index2, index3):
            angles = []

            for i in range(skeleton.shape[0]):
                x1, y1 = skeleton[i, index1 * 2], skeleton[i, index1 * 2 + 1]
                x2, y2 = skeleton[i, index2 * 2], skeleton[i, index2 * 2 + 1]
                x3, y3 = skeleton[i, index3 * 2], skeleton[i, index3 * 2 + 1]
                angle = joint_angle(np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3]))
                angles.append(angle)

            variance = circvar(angles, low=0, high=360)
            return variance

        left_arm_var = get_joint_variance(self.skeletons, 2, 3, 4)
        right_arm_var = get_joint_variance(self.skeletons, 5, 6, 7)

        th = 150
        if left_arm_var < th and right_arm_var < th:
            print('too static - left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print('not static - left var {}, right var {}'.format(left_arm_var, right_arm_var))
            return False

    def has_jumping_joint(self, verbose=False):
        frame_diff = np.squeeze(self.skeletons[1:, :16] - self.skeletons[:-1, :16])
        diffs = abs(frame_diff.flatten())
        width = max(self.skeletons[0, :16:2]) - min(self.skeletons[0, :16:2])

        if max(diffs) > width / 2.0:
            print('jumping joint - diff {}, width {}'.format(max(diffs), width))
            return True
        else:
            if verbose:
                print('no jumping joint - diff {}, width {}'.format(max(diffs), width))
            return False

    def has_missing_frames(self):
        n_empty_frames = 0
        n_frames = self.skeletons.shape[0]
        for i in range(n_frames):
            if np.sum(self.skeletons[i]) == 0:
                n_empty_frames += 1

        ret = n_empty_frames > n_frames * 0.1
        if ret:
            print('missing frames - {} / {}'.format(n_empty_frames, n_frames))
        return ret

    def smooth_motion(self):
        for i in range(16):
            self.skeletons[:, i] = savgol_filter(self.skeletons[:, i], 5, 2)

    def fill_missing_joints(self):
        # actually works super well! just checked!
        skeletons = self.skeletons
        n_joints = 8  # only upper body
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        for i in range(n_joints):
            xs, ys = skeletons[:, i * 2], skeletons[:, i * 2 + 1]
#             xs[xs == 0] = np.nan # because they are already normalized no need to filter out zeros
#             ys[ys == 0] = np.nan

            if sum(np.isnan(xs)) > len(xs) / 2:
                skeletons = None
                break

            if sum(np.isnan(ys)) > len(ys) / 2:
                skeletons = None
                break

            if np.isnan(xs).any():
                nans, t = nan_helper(xs)
                xs[nans] = np.interp(t(nans), t(~nans), xs[~nans])
                skeletons[:, i * 2] = xs

            if np.isnan(ys).any():
                nans, t = nan_helper(ys)
                ys[nans] = np.interp(t(nans), t(~nans), ys[~nans])
                skeletons[:, i * 2 + 1] = ys

        return skeletons
    
import lmdb as lmdb
from collections import defaultdict

def calc_spectrogram_length_from_motion_length(n_frames, fps):
    ret = (n_frames / fps * 16000 - 1024) / 512 + 1
    return int(round(ret))

def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
    expected_n = duration_in_sec * fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    if hasattr(poses, 'dtype'):
        interpolated_y = interpolated_y.astype(poses.dtype)
    return interpolated_y


class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, clip_lmdb_dir_spanish, out_lmdb_dir, n_poses, subdivision_stride,
                 pose_resampling_fps, disable_filtering=False):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.disable_filtering = disable_filtering

        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']
            
        self.src_lmdb_env_spanish = lmdb.open(clip_lmdb_dir_spanish, readonly=True, lock=False)
        with self.src_lmdb_env_spanish.begin() as txn_spanish:
            self.n_videos_spanish = txn_spanish.stat()['entries']

        self.spectrogram_sample_length = calc_spectrogram_length_from_motion_length(self.n_poses, self.skeleton_resampling_fps)
        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        map_size = 1024 * 50  # in MB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0
        
    def run(self):
        n_filtered_out = defaultdict(int)
        src_txn = self.src_lmdb_env.begin(write=False)
        src_txn_spanish = self.src_lmdb_env_spanish.begin(write=False)
        
        # sampling and normalization for english
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                filtered_result = self._sample_from_clip(vid, clip, 'en')
                for type in filtered_result.keys():
                    n_filtered_out[type] += filtered_result[type]
        
        # sampling and normalization for spanish
        cursor = src_txn_spanish.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                filtered_result = self._sample_from_clip(vid, clip, 'es')
                for type in filtered_result.keys():
                    n_filtered_out[type] += filtered_result[type]
                    
        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples: ', txn.stat()['entries'])
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                print('{}: {}'.format(type, n_filtered))
                n_total_filtered += n_filtered
            print('no. of excluded samples: {} ({:.1f}%)'.format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()['entries'] + n_total_filtered)))
            
        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    def _sample_from_clip(self, vid, clip, lang):
        clip_skeleton = clip['skeletons'] # changed from 3d
        # don't use audio for now
        # clip_audio = clip['audio_feat']
        # clip_audio_raw = clip['audio_raw']
        clip_word_list = clip['words']
        clip_s_f, clip_e_f = clip['start_frame_no'], clip['end_frame_no']
        clip_s_t, clip_e_t = clip['start_time'], clip['end_time']

        n_filtered_out = defaultdict(int)

        # skeleton resampling
        # just takes 25 fps and downsamples to 15 fps
        clip_skeleton = resample_pose_seq(clip_skeleton, clip_e_t - clip_s_t, self.skeleton_resampling_fps)

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_words_list = []

        num_subdivision = math.floor(
            (len(clip_skeleton) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1
        expected_audio_length = calc_spectrogram_length_from_motion_length(len(clip_skeleton), self.skeleton_resampling_fps)

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            subdivision_start_time = clip_s_t + start_idx / self.skeleton_resampling_fps
            subdivision_end_time = clip_s_t + fin_idx / self.skeleton_resampling_fps
            sample_words = self.get_words_in_time_range(word_list=clip_word_list,
                                                        start_time=subdivision_start_time,
                                                        end_time=subdivision_end_time)

            if len(sample_words) >= 2:
                # filtering motion skeleton data
                sample_skeletons, filtering_message = MotionPreprocessor(sample_skeletons).get()

#                 filtering_message = 'Turned off'
                is_correct_motion = (sample_skeletons != [])
                motion_info = {'vid': vid,
                               'start_frame_no': clip_s_f + start_idx,
                               'end_frame_no': clip_s_f + fin_idx,
                               'start_time': subdivision_start_time,
                               'end_time': subdivision_end_time,
                               'is_correct_motion': is_correct_motion, 'filtering_message': filtering_message}

                if is_correct_motion or self.disable_filtering:
                    sample_skeletons_list.append(sample_skeletons)
                    sample_words_list.append(sample_words)
                    aux_info.append(motion_info)
                else:
                    n_filtered_out[filtering_message] += 1

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for words, poses, aux in zip(sample_words_list, sample_skeletons_list, aux_info):
                    
                    # preprocessing for poses
                    poses = np.asarray(poses)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    v = [words, poses, {'lang': lang, 'vid': vid}]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1

        return n_filtered_out

    @staticmethod
    def get_words_in_time_range(word_list, start_time, end_time):
        words = []

        for word in word_list:
            _, word_s, word_e = word[0], word[1], word[2]

            if word_s >= end_time+2:
                break

            if word_e <= start_time-2:
                continue

            words.append(word)

        return words

class SpeechMotionDataset(Dataset):
    def __init__(self, lmdb_dir, lmdb_dir_spanish, n_poses=10, subdivision_stride=15, pose_resampling_fps=15,
                 speaker_model=None, remove_word_timing=False, input_resolution=224):
        
        super(SpeechMotionDataset, self).__init__()
        
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.remove_word_timing = remove_word_timing

        self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))
        self.expected_spectrogram_length = calc_spectrogram_length_from_motion_length(
            n_poses, pose_resampling_fps)
        
#         self._tokenizer = SimpleTokenizer()
#         self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', use_fast=False)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', use_fast=False)
        
        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            logging.info('Creating the dataset cache...')
            n_poses_extended = int(round(n_poses * 1.25))  # some margin
            data_sampler = DataPreprocessor(lmdb_dir, lmdb_dir_spanish, preloaded_dir, n_poses_extended,
                                            subdivision_stride, pose_resampling_fps)
            data_sampler.run()
        else:
            logging.info('Found the cache {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self):
        return self.n_samples
    
    @staticmethod
    def normalize_dir_vec(dir_vec, mean_dir_vec):
        return dir_vec - mean_dir_vec
    
    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, aux_info = sample

        def flatten_word_list(words, end_time=None):
            # just flattens the words
            indexes = []
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(word[0])
            return indexes
        # to tensors
        word_seq = flatten_word_list(word_seq)

        text_input = self.tokenizer(" ".join(word_seq), padding='max_length', max_length=10, truncation=True, return_tensors='pt')
        pose_input = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
        
        if any(np.isnan(pose_input).numpy().ravel()):
            raise
        
        return pose_input, text_input