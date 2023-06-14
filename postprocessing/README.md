# Postprocessing of experiments

## Inertial-based Architecture Experiments
Run the `majority_fil.py` script by changing the `path_to_preds` path-variable pointing towards a folder containing the logged experiments as separate folders, i.e.: 
- If you ran three experiments with varying different seeds place all three folders in a directory.
- Name each folder following the name structure `seed_X` where `X` is the employed seed of each experiment.
- Define the `seeds` and `majority_filters` you want to test.

## Camera-based Architecture Experiments
Run the `score_thres.py` script by changing the `path_to_preds` path-variable pointing towards a folder containing the logged experiments as separate folders, i.e.: 
- If you ran three experiments with varying different seeds place all three folders in a directory.
- Name each folder following the name structure `seed_X` where `X` is the employed seed of each experiment.
- Define the `seeds` and `score_thresholds` you want to test.

## Oracle-based Architecture Experiments
Run the `oracle.py` script by changing the `path_to_inertial_preds` and `path_to_camera_preds` path-variables pointing towards the best inertial- and camera-based approaches which you previously used during postprocessing.
