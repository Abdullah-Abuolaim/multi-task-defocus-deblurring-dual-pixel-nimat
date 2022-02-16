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

from config import *
from model import *
from metrics import *

def check_dir(_path):
    if not os.path.exists(_path):
        try:
            os.makedirs(_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
def filter_shapness_measure(img_,kernelSize):
    convX = cv2.Sobel(img_,cv2.CV_64F,1,0,ksize=kernelSize)
    convY = cv2.Sobel(img_,cv2.CV_64F,0,1,ksize=kernelSize)
    tempArrX=convX*convX
    tempArrY=convY*convY
    tempSumXY=tempArrX+tempArrY
    tempSumXY=np.sqrt(tempSumXY)
    return np.sum(tempSumXY)

def schedule_learning_rate(epoch):
    lr=lr_[int(epoch/scheduling_rate)]
    return lr

def data_random_shuffling(temp_type):
    global train_set, val_set, test_set
    
    images_c_src = [path_read + temp_type + blurry_img + f for f in os.listdir(path_read + temp_type + blurry_img) if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))]
    images_c_src.sort()
    
    images_s_trg = [path_read + temp_type + sharp_img + f for f in os.listdir(path_read + temp_type + sharp_img) if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))]
    images_s_trg.sort()
    
    images_l_src = [path_read + temp_type + left_img + f for f in os.listdir(path_read + temp_type + left_img) if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))]
    images_l_src.sort()
    
    images_r_src = [path_read + temp_type + right_img + f for f in os.listdir(path_read + temp_type + right_img) if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))]
    images_r_src.sort()

    if temp_type != 'test':
        tempInd=np.arange(len(images_c_src))
        random.shuffle(tempInd)
        
        images_c_src=np.asarray(images_c_src)[tempInd]
        images_s_trg=np.asarray(images_s_trg)[tempInd]
        
        images_l_src=np.asarray(images_l_src)[tempInd]
        images_r_src=np.asarray(images_r_src)[tempInd]

    for i in range(len(images_c_src)):
        if temp_type =='train':
            train_set.append([images_c_src[i],images_l_src[i],images_r_src[i],images_s_trg[i]])
        elif temp_type =='val':
            val_set.append([images_c_src[i],images_l_src[i],images_r_src[i],images_s_trg[i]])
        elif temp_type =='test':
            test_set.append([images_c_src[i],images_l_src[i],images_r_src[i],images_s_trg[i]])
        else:
            raise NotImplementedError
    
def test_generator_image(temp_test_img):
    global op_phase
    
    if op_phase=='test':
        temp_img = cv2.resize(cv2.imread(temp_test_img[0],color_flag),None,fy=args.downscale_ratio,fx=args.downscale_ratio)
    else:
        temp_img = cv2.resize(cv2.imread(temp_test_img,color_flag),None,fy=args.downscale_ratio,fx=args.downscale_ratio)
        
    if temp_img.dtype == 'uint16':
        temp_norm_val=(2**16)-1
    else:
        temp_norm_val=(2**8)-1
    
    num_maxpool_levels=4
    pad_mod=2**num_maxpool_levels
    
    padding_h, padding_w = 0, 0
    if temp_img.shape[0] % pad_mod != 0:
        padding_h=pad_mod-(temp_img.shape[0] % pad_mod)
    if temp_img.shape[1] % pad_mod != 0:
        padding_w=pad_mod-(temp_img.shape[1] % pad_mod)
        
    test_image = np.zeros((1, temp_img.shape[0]+padding_h,temp_img.shape[1]+padding_w, nb_ch))
    gt_image = np.zeros((1, temp_img.shape[0]+padding_h,temp_img.shape[1]+padding_w, nb_ch_out))
    
    test_image[0, padding_h:,padding_w:,:] = temp_img/temp_norm_val
    if op_phase=='test':
        gt_image[0, padding_h:,padding_w:,0:3] = cv2.imread(temp_test_img[1],color_flag)/temp_norm_val
        gt_image[0, padding_h:,padding_w:,3:6] = cv2.imread(temp_test_img[2],color_flag)/temp_norm_val
        gt_image[0, padding_h:,padding_w:,6:9] = cv2.imread(temp_test_img[3],color_flag)/temp_norm_val
    return test_image, gt_image, temp_norm_val, [padding_h,padding_w]


def rotate_image(in_img, angle):
  image_center = tuple(np.array(in_img.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  rot_img = cv2.warpAffine(in_img, rot_mat, in_img.shape[1::-1], flags=cv2.INTER_LINEAR)
  return rot_img

def test_generator_image_nimat(temp_test_img):
    temp_img = cv2.resize(cv2.imread(temp_test_img,color_flag),None,fy=args.downscale_ratio,fx=args.downscale_ratio)
    org_img_h, org_img_w = temp_img.shape[0], temp_img.shape[1]
    img_dia =int(math.ceil(math.sqrt(org_img_h**2 + org_img_w**2)))
        
    if temp_img.dtype == 'uint16':
        temp_norm_val=(2**16)-1
        temp_img_00=(np.zeros([img_dia, img_dia,3])).astype(np.uint16)
    else:
        temp_norm_val=(2**8)-1
        temp_img_00=(np.zeros([img_dia, img_dia,3])).astype(np.uint8)
    
    offset_h=img_dia-org_img_h
    offset_w=img_dia-org_img_w

    temp_img_00[offset_h//2:org_img_h+offset_h//2,offset_w//2:org_img_w+offset_w//2,:] = temp_img
    
    num_maxpool_levels=4
    pad_mod=2**num_maxpool_levels
    
    padding = 0
    if img_dia % pad_mod != 0:
        padding=pad_mod-(img_dia % pad_mod)
    
    test_image = np.zeros((4, img_dia+padding,img_dia+padding, nb_ch))
    
    test_image[0, padding:,padding:,:] = temp_img_00/temp_norm_val
    test_image[1, padding:,padding:,:] = rotate_image(temp_img_00,45)/temp_norm_val
    test_image[2, padding:,padding:,:] = rotate_image(temp_img_00,90)/temp_norm_val
    test_image[3, padding:,padding:,:] = rotate_image(temp_img_00,135)/temp_norm_val
    
    return test_image, temp_norm_val, padding, [offset_h//2,org_img_h+offset_h//2,offset_w//2,org_img_w+offset_w//2]

def generator(phase_gen='train'):
    if phase_gen == 'train':       
        data_set_temp=train_set
        nb_total=total_nb_train
    elif phase_gen == 'val':
        data_set_temp=val_set
        nb_total=total_nb_val
    else:
        raise NotImplementedError
        
    img_count = 0
    src_ims = np.zeros((img_mini_b, patch_h, patch_w, nb_ch))
    trg_ims = np.zeros((img_mini_b, patch_h_out, patch_w_out, nb_ch_out))
    
    while True:
        img_data_src_c = data_set_temp[img_count % nb_total][0]
        
        img_data_trg_l = data_set_temp[img_count % nb_total][1]
        img_data_trg_r = data_set_temp[img_count % nb_total][2]
        img_data_trg_s = data_set_temp[img_count % nb_total][3]
        
        img_src_c=cv2.imread(img_data_src_c,color_flag)/norm_val
        img_trg_l=cv2.imread(img_data_trg_l,color_flag)/norm_val
        img_trg_r=cv2.imread(img_data_trg_r,color_flag)/norm_val
        img_trg_s=cv2.imread(img_data_trg_s,color_flag)/norm_val
        
        ##################################################Rmove patches with lowest sharpness
        patch_sh=[]
        pts_patch_sp=[]
        if filter_patch:
            temp_img_mini_b = img_mini_b + filter_num
        else:
            temp_img_mini_b = img_mini_b
        
        for _pts in range(0, temp_img_mini_b):
            s_p=[random.randint(0,img_h_real-patch_h),random.randint(0,img_w_real-patch_w)]
            e_p=[s_p[0]+patch_h,s_p[1]+patch_w]
            
            if filter_patch:
                test_filt_img=img_trg_s[s_p[0]:e_p[0],s_p[1]:e_p[1],1]
                patch_sh.append(filter_shapness_measure(test_filt_img,5))
            
            pts_patch_sp.append(s_p)
            
        if filter_patch:
            for _pts in range(0, filter_num):
                rmv_ind=np.argmin(patch_sh)
                patch_sh.pop(rmv_ind)
                pts_patch_sp.pop(rmv_ind)
        ####################################################
    
        for i in range(0, img_mini_b):
            s_p=pts_patch_sp[i]
            e_p=[s_p[0]+patch_h,s_p[1]+patch_w]
            src_ims[i] = img_src_c[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
            trg_ims[i, :,:,0:3] = img_trg_l[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
            trg_ims[i, :,:,3:6] = img_trg_r[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
            trg_ims[i, :,:,6:9] = img_trg_s[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
            trg_ims[i, :,:,9:12] = img_src_c[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
    
        yield (src_ims,trg_ims)
        img_count += 1

            
def save_eval_predictions(path_to_save,test_image,predictions,gt_image,img_name,temp_norm_val,temp_padding):
    global op_phase
    
    if type(predictions) is list: 
            predictions = np.asarray(predictions[0])
    if op_phase=='test':
        mse_l, psnr_l, ssim_l = MSE_PSNR_SSIM((gt_image[0,:,:,0:3]).astype(np.float64), (predictions[0,:,:,0:3]).astype(np.float64))
        mse_r, psnr_r, ssim_r = MSE_PSNR_SSIM((gt_image[0,:,:,3:6]).astype(np.float64), (predictions[0,:,:,3:6]).astype(np.float64))
        mse_s, psnr_s, ssim_s = MSE_PSNR_SSIM((gt_image[0,:,:,6:9]).astype(np.float64), (predictions[0,:,:,6:9]).astype(np.float64))
        
        mae_l = MAE((gt_image[0,:,:,0:3]).astype(np.float64), (predictions[0,:,:,0:3]).astype(np.float64))
        mae_r = MAE((gt_image[0,:,:,3:6]).astype(np.float64), (predictions[0,:,:,3:6]).astype(np.float64))
        mae_s = MAE((gt_image[0,:,:,6:9]).astype(np.float64), (predictions[0,:,:,6:9]).astype(np.float64))
        
        mse_list_l.append(mse_l)
        mse_list_r.append(mse_r)
        mse_list_s.append(mse_s)
        
        psnr_list_l.append(psnr_l)
        psnr_list_r.append(psnr_r)
        psnr_list_s.append(psnr_s)
        
        ssim_list_l.append(ssim_l)
        ssim_list_r.append(ssim_r)
        ssim_list_s.append(ssim_s)
        
        mae_list_l.append(mae_l)
        mae_list_r.append(mae_r)
        mae_list_s.append(mae_s)
    
    if temp_norm_val == (2**8)-1:
        temp_in_img=((test_image*temp_norm_val)+src_mean).astype(np.uint8)
        temp_out_l_img=((predictions[0,:,:,0:3]*temp_norm_val)+src_mean).astype(np.uint8)
        temp_out_r_img=((predictions[0,:,:,3:6]*temp_norm_val)+src_mean).astype(np.uint8)
        temp_out_s_img=((predictions[0,:,:,6:9]*temp_norm_val)+src_mean).astype(np.uint8)
        
        # if op_phase=='test':
        #     temp_gt_l_img=((gt_image[0,:,:,0:3]*temp_norm_val)+src_mean).astype(np.uint8)
        #     temp_gt_r_img=((gt_image[0,:,:,3:6]*temp_norm_val)+src_mean).astype(np.uint8)
        #     temp_gt_s_img=((gt_image[0,:,:,6:9]*temp_norm_val)+src_mean).astype(np.uint8)
        
    elif temp_norm_val == (2**16)-1:
        temp_in_img=((test_image*temp_norm_val)+src_mean).astype(np.uint16)
        temp_out_l_img=((predictions[0,:,:,0:3]*temp_norm_val)+src_mean).astype(np.uint16)
        temp_out_r_img=((predictions[0,:,:,3:6]*temp_norm_val)+src_mean).astype(np.uint16)
        temp_out_s_img=((predictions[0,:,:,6:9]*temp_norm_val)+src_mean).astype(np.uint16)
        
        # if op_phase=='test':
        #     temp_gt_l_img=((gt_image[0,:,:,0:3]*temp_norm_val)+src_mean).astype(np.uint16)
        #     temp_gt_r_img=((gt_image[0,:,:,3:6]*temp_norm_val)+src_mean).astype(np.uint16)
        #     temp_gt_s_img=((gt_image[0,:,:,6:9]*temp_norm_val)+src_mean).astype(np.uint16)
    
    cv2.imwrite(path_to_save+str(img_name)+'_p_l.png',temp_out_l_img[temp_padding[0]:,temp_padding[1]:,:])
    cv2.imwrite(path_to_save+str(img_name)+'_p_r.png',temp_out_r_img[temp_padding[0]:,temp_padding[1]:,:])
    cv2.imwrite(path_to_save+str(img_name)+'_p_s.png',temp_out_s_img[temp_padding[0]:,temp_padding[1]:,:])
    
    # if op_phase=='test':
    #     cv2.imwrite(path_to_save+str(img_name)+'_g_l.png',temp_gt_l_img[temp_padding[0]:,temp_padding[1]:,:])
    #     cv2.imwrite(path_to_save+str(img_name)+'_g_r.png',temp_gt_r_img[temp_padding[0]:,temp_padding[1]:,:])
    #     cv2.imwrite(path_to_save+str(img_name)+'_g_s.png',temp_gt_s_img[temp_padding[0]:,temp_padding[1]:,:])

def save_eval_predictions_nimat(path_to_save,predictions,img_name,temp_norm_val,temp_padding,tmp_s_co):
    if type(predictions) is list: 
            predictions = np.asarray(predictions[0])
    
    img_order=[[6,2],[5,1],[4,0],[3,7]]
    for i in range(4):
        if temp_norm_val == (2**8)-1:
            temp_out_l_img=((predictions[i,:,:,0:3]*temp_norm_val)+src_mean).astype(np.uint8)
            temp_out_r_img=((predictions[i,:,:,3:6]*temp_norm_val)+src_mean).astype(np.uint8)
            
        elif temp_norm_val == (2**16)-1:
            temp_out_l_img=((predictions[i,:,:,0:3]*temp_norm_val)+src_mean).astype(np.uint16)
            temp_out_r_img=((predictions[i,:,:,3:6]*temp_norm_val)+src_mean).astype(np.uint16)
        
        temp_l_rot_img=temp_out_l_img[temp_padding:,temp_padding:,:]
        temp_r_rot_img=temp_out_r_img[temp_padding:,temp_padding:,:]
        
        temp_l_img=rotate_image(temp_l_rot_img,-1*i*45)
        temp_r_img=rotate_image(temp_r_rot_img,-1*i*45)
        
        out_l_img=temp_l_img[tmp_s_co[0]:tmp_s_co[1],tmp_s_co[2]:tmp_s_co[3],:]
        out_r_img=temp_r_img[tmp_s_co[0]:tmp_s_co[1],tmp_s_co[2]:tmp_s_co[3],:]
        
        cv2.imwrite(path_to_save+str(img_name)+'_'+str(img_order[i][0])+'.png',out_l_img)
        cv2.imwrite(path_to_save+str(img_name)+'_'+str(img_order[i][1])+'.png',out_r_img)
        
      
def save_results_list(list_prefix,temp_mse_list,temp_psnr_list,temp_ssim_list,temp_mae_list):
    np.save(path_write+'mse'+list_prefix,np.asarray(temp_mse_list))
    np.save(path_write+'psnr'+list_prefix,np.asarray(temp_psnr_list))
    np.save(path_write+'ssim'+list_prefix,np.asarray(temp_ssim_list))
    np.save(path_write+'mae'+list_prefix,np.asarray(temp_mae_list))
    np.save(path_write+'all'+list_prefix,[np.mean(np.asarray(temp_mse_list)),
                                          np.mean(np.asarray(temp_psnr_list)),
                                          np.mean(np.asarray(temp_ssim_list)),
                                          np.mean(np.asarray(temp_mae_list))])