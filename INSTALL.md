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

Install PyTorch distribution (we used version==2.3.1):

```
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install other requirements:
```
pip install -r requirements.txt
```

Compile C++ distributions of NMS (used by ActionFormer, TemporalMaxer and Tridet)

```
cd camera_baseline/actionformer/libs/utils
python setup.py install --user
cd ../../../..
cd camera_baseline/tridet/libs/utils
python setup.py install --user
cd ../../../..
```