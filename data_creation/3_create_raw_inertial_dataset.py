import itertools
import os

from glob import glob

import pandas as pd
import numpy as np

from utils import append_srt_to_dataframe

sbjs = ['sbj_0', 'sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5', 'sbj_6', 'sbj_7', 'sbj_8', 'sbj_9', 'sbj_10', 'sbj_11', 'sbj_12', 'sbj_13', 'sbj_14', 'sbj_15', 'sbj_16', 'sbj_17']
sens_header = ['sbj_id', 'right_arm_acc_x', 'right_arm_acc_y', 'right_arm_acc_z', 'right_leg_acc_x', 'right_leg_acc_y','right_leg_acc_z', 'left_leg_acc_x', 'left_leg_acc_y', 'left_leg_acc_z', 'left_arm_acc_x', 'left_arm_acc_y', 'left_arm_acc_z', 'label']

    
for i, subject in enumerate(sbjs):
    print('PROCESSING: {}'.format(subject))
    sbj_sens_files = sorted(glob(os.path.join(os.path.join('data_creation/recordings', subject + '_*', subject + '_*'))))
    sbj_sens_data = None

    # create sensor datasets
    for f in sbj_sens_files:
        data = pd.read_csv(f)
        if sbj_sens_data is None:
            sbj_sens_data = data
        else:
            sbj_sens_data = pd.concat((sbj_sens_data, data), axis=0, ignore_index=True)

    sbj_sens_data = append_srt_to_dataframe(sbj_sens_data, os.path.join('data_creation/video_annotations', subject + '.srt'), 50)
    sbj_id = pd.Series(np.full(len(sbj_sens_data), int(subject.split('_')[-1])), name='sbj_id')
    sbj_sens_data = pd.concat((sbj_id, sbj_sens_data), axis=1, ignore_index=True)
    sbj_sens_data.columns = sens_header
    sbj_sens_data.to_csv(os.path.join('data/wear/raw/inertial', subject + '.csv'), index=False)
        
        
