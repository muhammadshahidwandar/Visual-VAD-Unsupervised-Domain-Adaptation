# Tensorflow and Matlab Code for the paper [RealVAD: A Real-world Dataset for Voice Activity Detection]()[Voice Activity Detection by Upper Body Motion Analysis and Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content_ICCVW_2019/html/HBU/Shahid_Voice_Activity_Detection_by_Upper_Body_Motion_Analysis_and_Unsupervised_ICCVW_2019_paper.html)

## Overview

The code in this repository allows 

1. Training of Resent50 model using pretrained weights initialization.

2. Testing of Resnet50 model for VAD label (0: not speaking, 1: speaking).

3. Full model evalution interms of TP,TN,FP,FN and F1score measure. 
 
4. Saving of fc features and label in matlab(.mat) file for further processing. 

5. Training and testing of unsuperwised domain adaptation using fc features.

## Sub Directories and Files
There are three sub directories discribed as follows

###images
         This subdirectory containes some sample dynamic images
### Resnet-Training-Tensorflow

``Train_Main``: To train Resnet model on given dataset 

``Test_Main.py``: Test the trained model on signle image

``Model_Evaluation.py``: To Evaluate the trained model on complete test set

``FcMatWritter.py``: Resnet fc feature writter in .mat file format

``Resnet.py``: Resnet Model defination

``datageneratorBalancedBatch.py``: image batch generator with balance number of samples from each class

``datageneratorSequenceBatch.py``: sequential image batch generator

### Unsuperwised-DomainAdaptation-Matlab

``VAD_Domain_Adaptation.m``: Training and Testing of unsuperwised domain adaptation using resnet fc features

Some pretrained resnet50 model can be downloaded from this link

## Dependencies
* Python 3.5
* Tensorflow 1.12
* Opencv 3.0
* natsort 7.0.1

## How it works
1- Obtain your target datasets e.g.  RealVAD()

2- Generate and Save Dynamic Image using (https://github.com/hbilen/dynamic-image-nets) 

3- Define your training and test folds in text files 

4- Change paths and parameters in Train_Main.py and Train Resnet50 model

5- Evaluate trained model on test set using Model_Evaluation.py

6- Any signle dynamic image can be tested using Test_Main.py 

7- Save fc feature for training and test data in .mat file using FcMatWritter.py

8- Run VAD_Domain_Adaptation.m in matlab to perform and test Unsuperwised domain adaptation

## Reference

**RealVAD: A Real-world Dataset for Voice Activity Detection**  
Cigdem Beyan, Muhammad Shahid and Vittorio Murino, in Press 2020
```
    @InProceedings{
    }
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