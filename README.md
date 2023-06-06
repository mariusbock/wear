# WEAR: An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition

<img src="teaser.gif" width="100%"/>

## Abstract
Though research has shown the complementarity of camera- and inertial-based data, datasets which offer both modalities remain scarce. In this paper, we introduce WEAR, an outdoor sports dataset for both vision- and inertial-based Human Activity Recognition (HAR). The dataset comprises data from 18 participants performing a total of 18 different workout activities with untrimmed inertial (acceleration) and camera (egocentric video) data recorded at 10 different outside locations. Unlike previous egocentric datasets, WEAR provides a challenging prediction scenario marked by purposely introduced variations of certain activities as well as an overall small information overlap across modalities. Provided benchmark results reveal that single-modality architectures each have different strengths and weaknesses in their prediction performance. Further, in light of the recent success of transformer-based temporal action localization models, we demonstrate their versatility by applying them in a plain fashion using vision, inertial and combined (vision + inertial) features as input. Results show that vision transformers are not only able to produce competitive results using only inertial data, but also can function as an architecture to fuse both modalities by means of simple concatenation, with the combined approach (vision + inertial features) being able to produce the highest mean average precision and \MB{close-to-best} F1-score. Our approach is not only the first to demonstrate the applicability of vision-based transformers for inertial-based, but also their capabilities of combining inertial- and vision-based data, in the context of Human Activity Recognition. The code to reproduce experiments is publicly available [here](https://github.com/mariusbock/wear). An arXiv version of our paper is available [here](https://arxiv.org/abs/2304.05088).

## Changelog
- 18/04/2023: provided code to reproduce experiments.
- 12/04/2023: initial commit and [arXiv](https://arxiv.org/abs/2304.05088) uploaded.

## Installation
Please follow instructions mentioned in the [INSTALL.md](/INSTALL.md) file.

## Download
The full dataset can be downloaded [here](https://bit.ly/wear_dataset)

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

In order to log experiments to [Neptune.ai](https://neptune.ai) please provide `project`and `api_token` information in your local deployment (see lines `34-35` in `main.py`)

## Contact
Marius Bock (marius.bock@uni-siegen.de)

## Cite as
```
@article{bock2023wear,
  title={WEAR: An Outdoor Sports for Wearable and Egocentric Video Activity Recognition},
  author={Bock, Marius and Kuehne, Hilde and Van Laerhoven, Kristof and Moeller, Michael},
  volume={abs/2304.05088},
  journal={CoRR},
  year={2023},
  url={https://arxiv.org/abs/2304.05088}
}
```
