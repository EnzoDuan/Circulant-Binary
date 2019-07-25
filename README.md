# Test code for BONN
## Prepare
* install the PyTorch (0.4.0), torchvision, tensorboardX, python3
* download this code
* get the CIFAR dataset ready

##Install Convolutional Module and Binary Module
* cd install
* sh install.sh
* cd BinActivateFunc_PyTorch
* sh install.sh

## Train and Evaluation
* run ```python CIFAR.py --dataset_dir [your dataset path] --gpu 0``` 

## Please cite

```
@inproceedings{gu2019bonn,
  title={Circulant Binary Convolutional Networks: Enhancing the Performance of 1-bit
DCNNs with Circulant Back Propagation},
  author={Liu, ChunLei and Ding, Wenrui and Xia, Xin and Zhang, Baochang and Gu, Jiaxin  and Liu, Jianzhuang and Ji, Rongrong and David, Doermann },
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```