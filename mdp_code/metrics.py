"""
This module is for evaluation metrics calculation i.e., MSE, MAE, PSNR, SSIM.

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

from config import *

def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def MSE_PSNR_SSIM(img1, img2):
    mse_ = np.mean( (img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    PIXEL_MAX = 1
    return mse_, 10 * math.log10(PIXEL_MAX / mse_), structural_similarity(img1,
                                                    img2, data_range=PIXEL_MAX,
                                                    multichannel=True) #measure.compare_ssim