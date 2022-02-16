"""
Copyright (c) 2022-present, Abdullah Abuolaim
This code is the implementation of the multi-task DP network (MDP) for single
image defocus deblurring published in WACV'22. Paper title: Improving Single-
Image Defocus Deblurring: How Dual-Pixel Images Help Through Multi-Task
Learning.

This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

Note: this code is the implementation of the "Learning to Reduce Defocus Blur
by Realistically Modeling Dual-Pixel Data" paper published in ICCV 2021.
Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel

Email: abdullah.abuolaim@gmail.com
"""

import numpy as np
import os
import math
import cv2
import random
# from skimage import measure
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_absolute_error
from copy import deepcopy
import datetime

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, Callback
from keras import backend, applications
import tensorflow as tf
import argparse

# parse args
parser = argparse.ArgumentParser(description='Keras Multi-Task DP (MDP) Network')
parser.add_argument('--training_phase', default='phase_2', type=str,  help='training phase: phase_1 or phase_2')
parser.add_argument('--op_phase', default='train', type=str,  help='phase operation: train or test')
parser.add_argument('--phase_1_checkpoint_model', default='mdp_phase_1_wacv', type=str,  help='pretrained model from phase 1')
parser.add_argument('--test_model', default='mdp_phase_2_wacv', type=str,  help='test model name')
parser.add_argument('--path_to_data', default='../dd_dp_dataset_canon/', type=str,  help='dataset directory for the training phase')
parser.add_argument('--img_mini_b', default=8, type=int, help='image mini batch size')
parser.add_argument('--patch_size', default=512, type=int, help='patch size')
parser.add_argument('--epoch', default=80, type=int, help='number of train epoches')
parser.add_argument('--lr', default=6e-4, type=float, help='initial learning rate')
parser.add_argument('--schedule_lr_rate', default=8, type=int, help='after how many epochs you change learning rate')
parser.add_argument('--bit_depth', default=16, type=int, help='image bit depth datatype, uint16 or uint8')
parser.add_argument('--downscale_ratio', default=1, type=float, help='downscale input test image in case the gpu memory is not sufficient')
parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate')

args = parser.parse_args()

op_phase = args.op_phase
test_model = args.test_model

img_mini_b = args.img_mini_b

patch_h, patch_w = args.patch_size, args.patch_size

# number of epochs
nb_epoch = args.epoch

#initial learning rate
init_lr = args.lr

# after how many epochs you change learning rate
scheduling_rate = args.schedule_lr_rate

dropout_rate = args.dropout_rate

# paths to read data
path_read=args.path_to_data #  phase 1: 'dp_data_png/' (DLDP) phase 2: 'dd_dp_dataset_canon/' (DPDD)
#########################################################################
# READ & WRITE DATA PATHS									            #
#########################################################################
if args.training_phase == 'phase_2':
    continue_checkpoint=True
    deblurring_branch_trainable=True
    # loss function weights
    loss_weights_all =  [1,0.2,0.2,0.2]
    # paths to sub directories
    blurry_img='_c/source/'
    sharp_img='_c/target/'
    left_img='_l/source/'
    right_img='_r/source/'
else: #phase_1
    continue_checkpoint=False
    deblurring_branch_trainable=False
    # loss function weights
    loss_weights_all =  [1,1,0.2,0.2]
    # paths to sub directories
    blurry_img='/c/'
    sharp_img='/c/'
    left_img='/l/'
    right_img='/r/'
    # number of epochs
    nb_epoch = 60
    #initial learning rate
    init_lr = 3e-4

if continue_checkpoint:
    path_to_pretrained_model='./ModelCheckpoints/'+args.phase_1_checkpoint_model+'.hdf5'

# activation function used after each conv layer
acti_str = 'relu'

filter_patch=False
filter_num=2
    
if op_phase=='train':
    date_time='mdp_'+args.training_phase+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M")
    # path to write results
    path_write='./results/res_'+date_time+'/'
    # path to save model
    path_save_model='./ModelCheckpoints/'+date_time+'.hdf5'
    
    # path to tensorboard log
    log_path='./logs/'+ date_time
else:
    # path to write results
    path_write='./results/res_'+args.test_model+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M")+'/'
    path_read_model='./ModelCheckpoints/'+args.test_model+'.hdf5'

#########################################################################
# NUMBER OF IMAGES IN THE TRAINING, VALIDATION, AND TESTING SETS		#
#########################################################################
if op_phase == 'train':
    total_nb_train = len([path_read + 'train' + blurry_img + f for f
                    in os.listdir(path_read + 'train' + blurry_img)
                    if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))])
    
    total_nb_val = len([path_read + 'val' + blurry_img + f for f
                    in os.listdir(path_read + 'val' + blurry_img)
                    if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))])
elif op_phase == 'test':
    total_nb_test = len([path_read + 'test' + blurry_img + f for f
                    in os.listdir(path_read + 'test' + blurry_img)
                    if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))])
else:
    total_nb_test = len([path_read + f for f
                    in os.listdir(path_read)
                    if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))])
    
#########################################################################
# MODEL PARAMETERS & TRAINING SETTINGS									#
#########################################################################

# input image size
img_w_real=1680
img_h_real=1120

# mean value pre-claculated
src_mean=0
trg_mean=0

# output patch size
patch_w_out = patch_w
patch_h_out = patch_h

nb_ch=3

# number of out channels
nb_ch_out=12

# color flag:"1" for 3-channel 8-bit image or "0" for 1-channel 8-bit grayscale
# or "-1" to read image as it including bit depth
color_flag=-1

norm_val=(2**args.bit_depth)-1

if op_phase == 'train':
    # number of training image batches
    nb_train = int(total_nb_train)
    # number of validation image batches
    nb_val = int(total_nb_val)

# generate learning rate array
lr_=[]

lr_.append(init_lr)
for i in range(int(nb_epoch/scheduling_rate)):
    lr_.append(lr_[i]*0.5)

train_set, val_set, test_set =  [], [], []

mse_list_l, psnr_list_l, ssim_list_l, mae_list_l = [], [], [], []
mse_list_r, psnr_list_r, ssim_list_r, mae_list_r = [], [], [], []
mse_list_s, psnr_list_s, ssim_list_s, mae_list_s = [], [], [], []