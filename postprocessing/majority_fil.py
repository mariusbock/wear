import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns

from utils import majorityVoting, ANETdetection, convert_samples_to_segments

# postprocessing parameters
path_to_preds = ['path/to/preds/to/be/analysed']
seeds = [1, 2, 3]
sampling_rate = 50
majority_filters = [1, 251, 501, 751, 1001, 1251]
json_files = [
    'data/wear/annotations/wear_split_1.json', 
    'data/wear/annotations/wear_split_2.json', 
    'data/wear/annotations/wear_split_3.json'
    ]


print("Data Loading....")
for path in path_to_preds:
    for f in majority_filters:
        all_mAP = np.zeros((len(seeds), 5))
        all_recall = np.zeros((len(seeds), 19))
        all_prec = np.zeros((len(seeds), 19))
        all_f1 = np.zeros((len(seeds), 19))
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
                    data = pd.read_csv(os.path.join('data/wear/raw/inertial/' + str(sampling_rate) + 'hz', sbj + '.csv'), index_col=False).replace({"label": label_dict}).fillna(0).to_numpy()
                    v_data = np.append(v_data, data, axis=0)
                    
                v_preds = np.array([])
                v_orig_preds = np.load(os.path.join(path, 'seed_' + str(seed), 'unprocessed_results/v_preds_wear_split_{}.npy'.format(i + 1)))
                
                for sbj in val_sbjs:
                    sbj_pred = v_orig_preds[v_data[:, 0] == int(sbj.split("_")[-1])]
                    sbj_pred = [majorityVoting(i, sbj_pred.astype(int), f) for i in range(len(sbj_pred))]
                    v_preds = np.append(v_preds, sbj_pred)

                print("Converting to Segments....")
                seg_data = convert_samples_to_segments(v_data[:, 0], v_preds, sampling_rate)
                det_eval = ANETdetection(j, 'validation', tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7])
                all_preds = np.concatenate((all_preds, v_preds))
                all_gt = np.concatenate((all_gt, v_data[:, -1]))

                print("Evaluating....")
                v_mAP, _ = det_eval.evaluate(seg_data)
                v_prec = precision_score(v_data[:, -1], v_preds, average=None)
                v_rec = recall_score(v_data[:, -1], v_preds, average=None)
                v_f1 = f1_score(v_data[:, -1], v_preds, average=None)
                    
                all_prec[s_pos, :] += v_prec
                all_recall[s_pos, :] += v_rec
                all_f1[s_pos, :] += v_f1
                all_mAP[s_pos, :] += v_mAP

            if seed == 1:
                comb_conf = confusion_matrix(all_gt, all_preds, normalize='true')
                comb_conf = np.around(comb_conf, 2)
                comb_conf[comb_conf == 0] = np.nan

                _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
                sns.heatmap(comb_conf, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens, cbar=False, annot_kws={
                            'fontsize': 16,
                        })
                ax.set_title('Confusion Matrix')
                pred_name = path.split('/')[-2]
                _.savefig(pred_name + ".pdf")
                np.save("viz", all_preds)
                    
        print("Prediction for {} with threshold {}:".format(path_to_preds, f))
        print("Individual mAP:")
        print(np.around(np.mean(all_mAP, axis=0) / len(json_files), 4) * 100)

        print("Average mAP:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_mAP) / len(json_files) * 100, np.std(np.mean(all_mAP, axis=1) / len(json_files)) * 100))

        print("Individual Precision:")
        print(np.around(np.mean(all_prec, axis=0) / len(json_files), 4) * 100)

        print("Average Precision:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_prec) / len(json_files) * 100, np.std(np.mean(all_prec, axis=1) / len(json_files)) * 100))

        print("Individual Recall:")
        print(np.around(np.mean(all_recall, axis=0) / len(json_files), 4) * 100)

        print("Average Recall:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_recall) / len(json_files) * 100, np.std(np.mean(all_recall, axis=1) / len(json_files)) * 100))

        print("Individual F1:")
        print(np.around(np.mean(all_f1, axis=0) / len(json_files), 4) * 100)

        print("Average F1:")
        print("{:.4} (+/-{:.4})".format(np.mean(all_f1) / len(json_files) * 100, np.std(np.mean(all_f1, axis=1) / len(json_files)) * 100))