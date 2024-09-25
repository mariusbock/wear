import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns

from utils import ANETdetection, convert_segments_to_samples

# postprocessing parameters
save = False
test = True
sensor = 'inertial'
algorithm = 'tridet' 
experiment = '60_frames_30_stride'
path_to_preds = ['experiments/test_experiments/{}/{}/{}'.format(sensor, algorithm, experiment)]
seeds = [1, 2, 3]
score_thres = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
sampling_rate = 50
nb_classes = 19
if test:
    json_files = [
        'data/wear/annotations/60fps/wear_test_split_1.json',
        'data/wear/annotations/60fps/wear_test_split_2.json',
        'data/wear/annotations/60fps/wear_test_split_3.json',
        'data/wear/annotations/60fps/wear_test_split_4.json',
        'data/wear/annotations/60fps/wear_test_split_5.json',
        'data/wear/annotations/60fps/wear_test_split_6.json',
        ]
else:
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

for path in path_to_preds:
    for f in score_thres:
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
                if test:
                    v_seg = pd.read_csv(os.path.join(path, 'seed_' + str(seed), 'unprocessed_results/v_seg_wear_test.csv'.format(int(i) + 1, seed)), index_col=None)
                else:
                    v_seg = pd.read_csv(os.path.join(path, 'seed_' + str(seed), 'unprocessed_results/v_seg_wear_split_{}.csv'.format(int(i) + 1, seed)), index_col=None)

                for sbj in val_sbjs:
                    data = pd.read_csv(os.path.join('data/wear/raw/inertial/{}hz'.format(sampling_rate), sbj + '.csv'), index_col=False, low_memory=False).replace({"label": label_dict}).fillna(0).to_numpy()
                    v_data = np.append(v_data, data, axis=0)
            
                v_seg = v_seg.rename(columns={"video_id": "video-id", "t_start": "t-start", "t_end": "t-end"})
                #v_seg = v_seg[v_seg['video-id'] == val_sbjs[0]]
                preds, gt, _ = convert_segments_to_samples(v_seg, v_data, sampling_rate, threshold=f)
                all_preds = np.concatenate((all_preds, preds))
                all_gt = np.concatenate((all_gt, gt))
                labels = range(nb_classes)
                v_seg = v_seg[v_seg.score > f]
                det_eval = ANETdetection(j, 'validation', tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7])
                            
                v_mAP, _ = det_eval.evaluate(v_seg)
                v_prec = precision_score(gt, preds, average=None, labels=labels)
                v_rec = recall_score(gt, preds, average=None, labels=labels)
                v_f1 = f1_score(gt, preds, average=None, labels=labels)
                """print(v_mAP)
                print(np.mean(v_prec))
                print(np.mean(v_rec))
                print(np.mean(v_f1))"""
                all_prec[s_pos, :] += v_prec
                all_recall[s_pos, :] += v_rec
                all_f1[s_pos, :] += v_f1
                all_mAP[s_pos, :] += v_mAP
            if seed == 1 and save:
                if f == 0.1 or f == 0.0:
                    comb_conf = confusion_matrix(all_gt, all_preds, normalize='true')
                    comb_conf = np.around(comb_conf, 2)
                    comb_conf[comb_conf == 0] = np.nan

                    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
                    sns.heatmap(comb_conf, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens, cbar=False, annot_kws={
                            'fontsize': 16,
                        })
                    pred_name = path.split('/')[-2]
                    _.savefig(algorithm + '_' + sensor + '_' + str(f) + "_" + experiment + ".pdf")
                    np.save(algorithm + '_' + sensor + '_' + str(f) + "_" + experiment, all_preds)
            
        print("Prediction for {} with threshold {}:".format(path_to_preds, f))
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
