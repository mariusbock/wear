# WEAR: A Multimodal Dataset for Wearable and Egocentric Video Activity Recognition

![](teaser.gif)

## Abstract
Though research has shown the complementarity of camera- and inertial-based data, datasets which offer both modalities remain scarce. In this paper we introduce WEAR, a multimodal benchmark dataset for both vision- and wearable-based Human Activity Recognition (HAR). The dataset comprises data from 18 participants performing a total of 18 different workout activities with untrimmed inertial (acceleration) and camera (egocentric video) data recorded at 10 different outside locations. WEAR features a diverse set of activities which are low in inter-class similarity and, unlike previous egocentric datasets, not defined by human-object-interactions nor originate from inherently distinct activity categories. Provided benchmark results reveal that single-modality architectures have different strengths and weaknesses in their prediction performance. Further, in light of the recent success of transformer-based video action detection models, we demonstrate their versatility by applying them in a plain fashion using vision, inertial and combined (vision + inertial) features as input. Results show that vision transformers are not only able to produce competitive results using only inertial data, but also can function as an architecture to fuse both modalities by means of simple concatenation, with the multimodal approach being able to produce the highest average mAP, precision and close-to-best F1-scores. Up until now, vision-based transformers have neither been explored in inertial nor in multimodal human activity recognition, making our approach the first to do so. An arXiv of our paper can be found at this [link](https://arxiv.org/abs/2304.05088).

## Changelog
- 18/04/2023: provided code to reproduce experiments.
- 12/04/2023: initial commit and [arXiv](https://arxiv.org/abs/2304.05088) uploaded.

## Installation
Please follow instructions mentioned in the [INSTALL.md](/Install.md) file.

## Download
The full dataset can be downloaded via https://bit.ly/wear_dataset

The download folder is divided into 3 subdirectories
- **annotations (> 1MB)**: JSON-files containing annotations per-subject using the THUMOS14-style
- **processed (15GB)**: precomputed I3D, inertial and combined per-subject features
- **raw (130GB)**: Raw, per-subject video and inertial data

## Reproduce Experiments
Once having installed requirements, one can rerun experiments by running the `main.py` script:

````
python main.py --config ./configs/60_frames_30_stride/actionformer_combined.yaml --seed 1 --eval_type split
````

Each config file represents one type of experiment. Each experiment was run three times using three different random seeds (i.e. `1, 2, 3`).

### Logging using Neptune.ai

In order to log experiments to [Neptune.ai](https://neptune.ai) please provide `project`and `api_token` information in your local deployment (see lines `33-34` in `main.py`)

## Contact
Marius Bock (marius.bock@uni-siegen.de)

## Cite as
```
@article{bock2023wear,
  title={WEAR: A Multimodal Dataset for Wearable and Egocentric Video Activity Recognition},
  author={Bock, Marius and Moeller, Michael and Van Laerhoven, Kristof and Kuehne, Hilde},
  volume={abs/2304.05088},
  journal={CoRR},
  year={2023},
  url={https://arxiv.org/abs/2304.05088}
}
```
