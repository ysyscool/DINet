# DINet
This repository contains the reference code for our TMM paper:
[arXiv Paper Version](https://arxiv.org/pdf/1904.03571.pdf)

Please cite with the following Bibtex code:
```
@article{yang2019dilated,
  title={A Dilated Inception Network for Visual Saliency Prediction},
  author={Yang, Sheng and Lin, Guosheng and Jiang, Qiuping and Lin, Weisi},
  journal={arXiv preprint arXiv:1904.03571},
  year={2019}
}
```

## Requirements
* Python 2.7
* Keras 2.1.2
* Tensorflow-gpu 1.3.0
* opencv-python


## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/ysyscool/DINet
cd DINet
mkdir models
```

- Download weights from [Google Drive](https://drive.google.com/file/d/1o8RTCUpP08iDO7XdDbbiz3bW8vJfb6Yw/view?usp=sharing).
Put the weights into 
```bash
cd models
```

### Train/Test
Download the SALICON 2015 dataset and modify the paths in config.yaml
And then using the following command to train the model
```bash
python main.py --phase=train --batch_size=10
```

For testing, modify the variables of weightfile (in line 217) and imgs_test_path (in line 220) in the main.py.
And then using the following command to test the model
```bash
python main.py --phase=test
```

### Evaluation on SALICON dataset
Please refer to this [link](https://github.com/NUS-VIP/salicon-evaluation). 

## Acknowledgments
Code largely benefits from [sam](https://github.com/marcellacornia/sam). 
