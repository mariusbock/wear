import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from utils import majorityVoting, combine_predictions, convert_samples_to_segments, convert_segments_to_samples, ANETdetection


path_to_inertial = 'experiments/main_experiments/inertial/aandd/30_frames_15_stride'
path_to_camera = 'experiments/main_experiments/camera/tridet/30_frames_15_stride'
combined = True
path_to_combined = 'experiments/main_experiments/combined/tridet/30_frames_15_stride'
seeds = [1, 2, 3]
majority_filter = 501
score_threshold = 0.1
sampling_rate = 50
nb_classes = 19
json_files = [
    'data/wear/annotations/60fps/wear_split_1.json', 
    'data/wear/annotations/60fps/wear_split_2.json', 
    'data/wear/annotations/60fps/wear_split_3.json',
    'data/wear/annotations/60fps/wear_split_4.json', 
    'data/wear/annotations/60fps/wear_split_5.json', 
    'data/wear/annotations/60fps/wear_split_6.json',
    'data/wear/annotations/60fps/wear_split_7.json', 
    'data/wear/annotations/60fps/wear_split_8.json', 
    'data/wear/annotations/60fps/wear_split_9.json',
    'data/wear/annotations/60fps/wear_split_10.json', 
    'data/wear/annotations/60fps/wear_split_11.json', 
    'data/wear/annotations/60fps/wear_split_12.json',
    'data/wear/annotations/60fps/wear_split_13.json', 
    'data/wear/annotations/60fps/wear_split_14.json', 
    'data/wear/annotations/60fps/wear_split_15.json',
    'data/wear/annotations/60fps/wear_split_16.json',
    'data/wear/annotations/60fps/wear_split_17.json',
    'data/wear/annotations/60fps/wear_split_18.json',
    ]

print("Data Loading....")
all_mAP = np.zeros((len(seeds), 5))
all_recall = np.zeros((len(seeds), nb_classes))
all_prec = np.zeros((len(seeds), nb_classes))
all_f1 = np.zeros((len(seeds), nb_classes))
for s_pos, seed in enumerate(seeds):
    all_preds = np.array([])
    all_gt = np.array([])
    
    for i, j in enumerate(json_files):
        with open(j) as fi:
            file = json.load(fi)
            anno_file = file['database']
            labels = ['null'] + list(file['label_dict'])
            label_dict = dict(zip(labels, list(range(len(labels)))))
            val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']
                
        v_data = np.empty((0, 12 + 2))
        for sbj in val_sbjs:
            data = pd.read_csv(os.path.join('data/wear/raw/inertial/{}hz'.format(sampling_rate), sbj + '.csv'), index_col=False).replace({"label": label_dict}).fillna(0).to_numpy()
            v_data = np.append(v_data, data, axis=0)

        preds_inertial = np.array([])
        v_orig_preds = np.load(os.path.join(path_to_inertial, 'seed_' + str(seed), 'unprocessed_results/v_preds_wear_split_{}.npy'.format(i + 1)))

        for sbj in val_sbjs:
            sbj_pred = v_orig_preds[v_data[:, 0] == int(sbj.split("_")[-1])]
            sbj_pred = [majorityVoting(i, sbj_pred.astype(int), majority_filter) for i in range(len(sbj_pred))]
            preds_inertial = np.append(preds_inertial, sbj_pred)
            
        print("Converting to Samples....")
        v_seg_camera = pd.read_csv(os.path.join(path_to_camera, 'seed_' + str(seed), 'unprocessed_results/v_seg_wear_split_{}.csv'.format(int(i) + 1, seed)), index_col=None)
        v_seg_camera = v_seg_camera.rename(columns={"video_id": "video-id", "t_start": "t-start", "t_end": "t-end"})
        preds_camera, gt, _ = convert_segments_to_samples(v_seg_camera, v_data, sampling_rate, threshold=score_threshold)
        preds = combine_predictions(preds_inertial, preds_camera, gt)
        if combined:
            v_seg_combined = pd.read_csv(os.path.join(path_to_combined, 'seed_' + str(seed), 'unprocessed_results/v_seg_wear_split_{}.csv'.format(int(i) + 1, seed)), index_col=None)
            v_seg_combined = v_seg_combined.rename(columns={"video_id": "video-id", "t_start": "t-start", "t_end": "t-end"})
            preds_combined, gt, _ = convert_segments_to_samples(v_seg_combined, v_data, sampling_rate, threshold=score_threshold)
            preds = combine_predictions(preds, preds_combined, gt)
        v_seg = convert_samples_to_segments(v_data[:, 0], preds, sampling_rate)
        all_preds = np.concatenate((all_preds, preds))
        all_gt = np.concatenate((all_gt, gt))
        labels = range(nb_classes)

        det_eval = ANETdetection(j, 'validation', tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7])
                        
        print("Evaluating {}....".format(j))
        v_mAP, _ = det_eval.evaluate(v_seg)
        v_prec = precision_score(gt, preds, average=None, labels=labels)
        v_rec = recall_score(gt, preds, average=None, labels=labels)
        v_f1 = f1_score(gt, preds, average=None, labels=labels)

        all_prec[s_pos, :] += v_prec
        all_recall[s_pos, :] += v_rec
        all_f1[s_pos, :] += v_f1
        all_mAP[s_pos, :] += v_mAP
    if seed == 1:
        comb_conf = confusion_matrix(all_gt, all_preds, normalize='true')
        comb_conf = np.around(comb_conf, 2)
        comb_conf[comb_conf == 0] = np.nan

        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        sns.heatmap(comb_conf, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens, cbar=False, annot_kws={'fontsize': 16})
        pred_name = "oracle"
        _.savefig(pred_name + ".pdf")
        np.save(pred_name, all_preds)

print("Individual mAP:")
print(np.around(np.mean(all_mAP, axis=0) / len(json_files), 4) * 100)

print("Average mAP:")
print("{:.4} (+/-{:.4})".format(np.mean(all_mAP) / len(json_files) * 100, np.std(np.mean(all_mAP, axis=1) / len(json_files)) * 100))

print("Average Precision:")
print("{:.4} (+/-{:.4})".format(np.mean(all_prec) / len(json_files) * 100, np.std(np.mean(all_prec, axis=1) / len(json_files)) * 100))

print("Average Recall:")
print("{:.4} (+/-{:.4})".format(np.mean(all_recall) / len(json_files) * 100, np.std(np.mean(all_recall, axis=1) / len(json_files)) * 100))

print("Average F1:")
print("{:.4} (+/-{:.4})".format(np.mean(all_f1) / len(json_files) * 100, np.std(np.mean(all_f1, axis=1) / len(json_files)) * 100))
