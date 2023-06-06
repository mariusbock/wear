# Installation Guide
Clone repository:

```
git clone git@github.com:mariusbock/wear.git
cd wear
```

Create [Anaconda](https://www.anaconda.com/products/distribution) environment:

```
conda create -n wear python==3.10.4
conda activate wear
```

Install PyTorch distribution:

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install other requirements:
```
pip install -r requirements.txt
```

Compile C++ distribution of NMS (used by Tridet and ActionFormer)

```
cd camera_baseline/actionformer/libs/utils
python setup.py install --user
cd ../../../..
cd camera_baseline/tridet/libs/utils
python setup.py install --user
cd ../../../..
```