# Tensorflow and Matlab code for the papers: 
*[RealVAD: A Real-world Dataset for Voice Activity Detection](https://ieeexplore.ieee.org/document/9133504)

*[Voice Activity Detection by Upper Body Motion Analysis and Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content_ICCVW_2019/html/HBU/Shahid_Voice_Activity_Detection_by_Upper_Body_Motion_Analysis_and_Unsupervised_ICCVW_2019_paper.html)

## Overview

The code in this repository allows 

1. Training a ResNet50 model with the pre-trained weights used for network initialization

2. Using the trained ResNet50 model for Voice Activity Detection (VAD) labels: 0: not-speaking, 1: speaking

3. Test performance evalution is in terms of True Positive, True Negative, False Positive and False Negative Rates and F1-score
 
4. Extracting and saving fc features into a mat. file

5. Training and testing unsupervised domain adaptation with fc features extracted.

## Sub-directories and Files
There are three sub-directories described as follows:

### Images
Containes some sample dynamic images

### RealVAD
Containes some sample train and validation set 

### Resnet-Finetuning

``Train_Main``: To train ResNet model on a given dataset 

``Test_Main.py``: Test the trained model on a single image

``Model_Evaluation.py``: To evaluate the trained model on a complete test set

``FcMatWritter.py``: To write extracted fc features into a .mat file

``Resnet.py``: ResNet model definition

``datageneratorBalancedBatch.py``: Image batch generator with balanced number of samples from each class

``datageneratorSequenceBatch.py``: Sequential image batch generator

### Unsupervised-DomainAdaptation-Matlab

``VAD_Domain_Adaptation.m``: Training and test unsupervised domain adaptation when the ResNet fc features are the input

Some pre-trained ResNet50 model can be downloaded from this link (https://drive.google.com/drive/folders/1dHYOMuzXHL46P1zDgDyDj9NgYzV1nNSS?usp=sharing)

## Dependencies
* Python 3.5
* Tensorflow 1.12
* Opencv 3.0
* Natsort 7.0.1
* Matlab 2017b

## How it works
1- Obtain your target datasets e.g.  RealVAD (https://github.com/IIT-PAVIS/Voice-Activity-Detection)

2- Generate and save the dynamic image by using (https://github.com/hbilen/dynamic-image-nets) 

3- Define your training and test folds in the text files (example files as given as trainRealVAD1.txt and testRealVAD1.txt in RealVAD sub-directory)

4- Change paths and parameters in Train_Main.py to train ResNet model

5- Evaluate trained model on test set by using Model_Evaluation.py

6- Any single dynamic image can be tested by using Test_Main.py 

7- Save fc feature for training and test data in .mat file using FcMatWritter.py

8- Run VAD_Domain_Adaptation.m in matlab to perform and test Unsupervised Domain Adaptation component

## Reference

**RealVAD: A Real-world Dataset for Voice Activity Detection**  
Cigdem Beyan, Muhammad Shahid and Vittorio Murino,
```
@ARTICLE{Beyan2020TMM,
  author={C. {Beyan} and M. {Shahid} and V. {Murino}},
  journal={IEEE Transactions on Multimedia},
  title={RealVAD: A Real-world Dataset and A Method for Voice Activity Detection by Body Motion Analysis},
  year={2020},
  volume={},
  number={},
  pages={1-1},}
```
**Voice Activity Detection by Upper Body Motion Analysis and Unsupervised Domain Adaptation**  
Muhammad Shahid, Cigdem Beyan and Vittorio Murino, ICCVW 2019
```
@inproceedings{shahid2019voice,
  title={Voice Activity Detection by Upper Body Motion Analysis and Unsupervised Domain Adaptation},
  author={Shahid, Muhammad and Beyan, Cigdem and Murino, Vittorio},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
```
