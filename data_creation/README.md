# How-To: Record your own data

The following guide will detail how to record your own data and contribute to the WEAR dataset with your own workout session. To start, please read the [recording plan](https://docs.google.com/document/d/15YZndREj4cDvrFpxDFYV-2q6co2NlVdPZfJpOLrcLJQ/edit?usp=share_link). The recording plan details all the sports activities part of the WEAR dataset as well as gives you an overview of the equipment you need to record the dataset.

## Hardware setup

In order to record data for the WEAR dataset you need: 
1. One headmounted camera (preferably being able to record in a large field-of-view). We recommend action cameras such as a [GoPro](https://gopro.com) as they are lightweight and often come with stable headmounts. During recording of the original WEAR dataset, we set the headmounted camera to record at 1080p with a frame rate of 60FPS and large field-of-view.
2. A second camera and tripod which you can place securely in the proximity of your workout location. While placing the camera is important that the field-of-view of the camera captures as much area as possible you are going to perfrom the activities in.  
3. Four wearable acceleration sensors which are able to be placed securely on your wrists and ankles. It is important that the sensor provides you with raw, uncompressed accleration data. We recommend using the [Bangle.js Version 1 smartwatches](https://shop.espruino.com/ble/banglejs) as they are a cost-effective, open-source smartwatch. If you opt to use the Bangle.js smartwatch please install the latest version of the [Bangle.js Activate Loader](https://github.com/kristofvl/BangleApps) on all watches. During recording of the original WEAR dataset, we set the smartwatches to record at 50Hz with a sensitivity of ±8G.


## Data Collection of a Session
In order to record a session, please equip all sensors as detailled in the [recording plan](https://docs.google.com/document/d/15YZndREj4cDvrFpxDFYV-2q6co2NlVdPZfJpOLrcLJQ/edit?usp=share_link). Before commencing any activities, you have to perform synchronization jumps. These will help you identify the start and end times of your workout in the inertial sensor recordings and are essential for synchronizing the camera and inertial data. To do so, stand still (i.e. not moving your limbs) in front of the tripod-mounted camera for at least 10 seconds, make 3 jumps with rasinig your arms during the jump and stand still for another 10 seconds. After having finished ALL activities of your recording session, repeat the synchronization jump procedure one more time.

## Postprocessing
Once having performed a session you will need to synchronize the camera and inertial data. We recommend creating a `recordings` folder which contains subfolders, each belonging to a different recording session (e.g. `recordings/sbj_1_session_1`. 

1. Depending on whether you are using the Bangle.js smartwatches use the `1_identify_sync_points.py` script to visualize all four sensor streams. The script reads all relevant files of the sensors you provide and concatenates the chunked bin files into one continious sensor stream per smartwatch. For each sensor axis, note down the peak data points belonging to your first and last synchronization jump. Afterwards, load your videos into a video editing tool of your choice (e.g. FinalCut) and note down the two timestamps belonging to the first and last synchronization jump.
2. Using the timestamps in both data streams use the `2_create_synced_files.py` script to create four `.wav`-files, each representing the magnitude of a smartwatch. Load the four files into your video editing tool and verify your synchronization by comparing the visual representation of the acceleration data with your video stream (NOTE: this is a tedious process, but will get easier over time).
3. Once you have verified that all smartwatches are synced with your video stream, crop the video stream to be the exact length as the `.wav`-files and export the video. Next annotate the video using subtitles with the activity labels. Once finished, export the subtitles as an `.srt`-file.
4. Run the `3_create_raw_inertial_dataset.py` to create a final `.csv`-file per subject containing all sensor axis as well as labels.
5. Compute I3D features for all created videos using clip lenghts of 0.5, 1 and 2 seconds. Concatenate rgb and flow features such that you are left with one `.npy` file per subject containing a feature vector of size `2048` per clip.
6. Finally, having created all relevant files run the `4_create_features_and_annotations_dataset.py`. It will create all processed files such as the vectorized inertial features and combined features (inertial + camera) used during the multimodal experiments.

## Contact
If you have any questions about the data collection process and would like to contribute to our dataset, please do not hesitate to contact us via marius.bock@uni-siegen.de.