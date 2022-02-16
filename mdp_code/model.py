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

def get_val_imgs(num_val_imgs):
    src_ims = np.zeros((num_val_imgs, patch_h, patch_w, nb_ch))
    trg_ims = np.zeros((num_val_imgs, patch_h_out, patch_w_out, nb_ch_out))
    
    indx_rand=random.randint(0,total_nb_val-1)
    img_data_src_c = val_set[indx_rand][0]
    
    img_data_trg_l = val_set[indx_rand][1]
    img_data_trg_r = val_set[indx_rand][2]
    img_data_trg_s = val_set[indx_rand][3]
    
    img_src_c=cv2.imread(img_data_src_c,color_flag)/norm_val
    img_trg_l=cv2.imread(img_data_trg_l,color_flag)/norm_val
    img_trg_r=cv2.imread(img_data_trg_r,color_flag)/norm_val
    img_trg_s=cv2.imread(img_data_trg_s,color_flag)/norm_val
    
    ##################################################Rmove patches with lowest sharpness
    for i in range(0, num_val_imgs):
        s_p=[random.randint(0,img_h_real-patch_h),random.randint(0,img_w_real-patch_w)]
        e_p=[s_p[0]+patch_h,s_p[1]+patch_w]
        src_ims[i] = img_src_c[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
        trg_ims[i, :,:,0:3] = img_trg_l[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
        trg_ims[i, :,:,3:6] = img_trg_r[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
        trg_ims[i, :,:,6:9] = img_trg_s[s_p[0]:e_p[0],s_p[1]:e_p[1],:]
    return src_ims,trg_ims

class TensorBoardImage(Callback):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        num_val_imgs=4
        src_ims_val, trg_ims_val = get_val_imgs(num_val_imgs)
        pred_val=self.model.predict(src_ims_val,1)
       
        file_writer = tf.summary.create_file_writer(log_path)
        
        if type(pred_val) is list: 
            pred_val=np.asarray(pred_val[0])
       
        temp_imgs_l=pred_val[:,:,:,0:3]
        temp_imgs_r=pred_val[:,:,:,3:6]
        temp_imgs_s=pred_val[:,:,:,6:9]
        
        src_ims_val=src_ims_val[:,:,:,::-1]
        temp_imgs_l=temp_imgs_l[:,:,:,::-1]
        temp_imgs_r=temp_imgs_r[:,:,:,::-1]
        temp_imgs_s=temp_imgs_s[:,:,:,::-1]
        with file_writer.as_default():
          # Don't forget to reshape.
          val_images_in = np.reshape(src_ims_val[0:num_val_imgs], (-1, patch_w, patch_h, nb_ch))
          tf.summary.image('val_epoch_'+str(epoch+1).zfill(3), val_images_in, max_outputs=num_val_imgs, step=0)
          val_images_l = np.reshape(temp_imgs_l[0:num_val_imgs], (-1, patch_w, patch_h, nb_ch))
          tf.summary.image('val_epoch_'+str(epoch+1).zfill(3), val_images_l, max_outputs=num_val_imgs, step=1)
          val_images_r = np.reshape(temp_imgs_r[0:num_val_imgs], (-1, patch_w, patch_h, nb_ch))
          tf.summary.image('val_epoch_'+str(epoch+1).zfill(3), val_images_r, max_outputs=num_val_imgs, step=2)
          val_images_s = np.reshape(temp_imgs_s[0:num_val_imgs], (-1, patch_w, patch_h, nb_ch))
          tf.summary.image('val_epoch_'+str(epoch+1).zfill(3), val_images_s, max_outputs=num_val_imgs, step=3)
        return
    
def mse_s(y_true, y_pred):
      return backend.mean(backend.square(y_pred[:, :,:,6:9] - y_true[:, :,:,6:9]),axis=-1)
 
def mse_l_r(y_true, y_pred):
    return backend.mean(backend.square(y_pred[:, :,:,0:6] - y_true[:, :,:,0:6]),axis=-1)

def L_C_loss(y_true, y_pred):
    return backend.mean(backend.square(y_true[:, :,:,9:12] - (y_pred[:, :,:,0:3]+y_pred[:, :,:,3:6])/2), axis=-1)
    
def L_D_loss(y_true, y_pred):
    return backend.mean(backend.square((y_pred[:, :,:,0:3]-y_pred[:, :,:,3:6])-(y_true[:, :,:,0:3]-y_true[:, :,:,3:6])), axis=-1)

def unet(_in_data):
    conv1 = Conv2D(64, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(_in_data)
    conv1 = Conv2D(64, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    #############################left view branch - early
    conv5 = Conv2D(1024, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    up6 = Conv2D(512, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(merge6)

    up7 = Conv2D(256, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    #############################right view branch - early
    conv10 = Conv2D(1024, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    up11 = Conv2D(512, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv10))
    merge11 = concatenate([conv4,up11], axis = 3)
    conv11 = Conv2D(512, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(merge11)

    up12 = Conv2D(256, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv11))
    merge12 = concatenate([conv3,up12], axis = 3)
    conv12 = Conv2D(256, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(merge12)
    conv12 = Conv2D(256, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv12)

    up13 = Conv2D(128, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv12))

    #############################defocus deblurring branch - early
    conv15 = Conv2D(1024, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    conv15.trainable=deblurring_branch_trainable
    conv15=conv15(pool4)
    
    up16 = Conv2D(512, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    up16.trainable=deblurring_branch_trainable
    up16=up16(UpSampling2D(size = (2,2))(conv15))
    
    merge16 = concatenate([conv4,up16], axis = 3)
    conv16 = Conv2D(512, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    conv16.trainable=deblurring_branch_trainable
    conv16=conv16(merge16)

    up17 = Conv2D(256, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    up17.trainable=deblurring_branch_trainable
    up17=up17(UpSampling2D(size = (2,2))(conv16))
    
    merge17 = concatenate([conv3,up17], axis = 3)
    conv17 = Conv2D(256, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    conv17.trainable=deblurring_branch_trainable
    conv17=conv17(merge17)
    
    conv17_ = Conv2D(256, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    conv17_.trainable=deblurring_branch_trainable
    conv17_=conv17_(conv17)

    up18 = Conv2D(128, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    up18.trainable=deblurring_branch_trainable
    up18= up18(UpSampling2D(size = (2,2))(conv17_))
    
    #############################left view branch - late
    merge8 = concatenate([conv2,up8,up13,up18], axis = 3)
    conv8 = Conv2D(128, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    
    conv9 = Conv2D(3, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv9) #2,3
    conv_l = Conv2D(nb_ch, 1, activation = 'sigmoid')(conv9)
        
    #############################right view branch - late
    merge13 = concatenate([conv2,up8,up13,up18], axis = 3)
    conv13 = Conv2D(128, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(merge13)
    conv13 = Conv2D(128, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv13)

    up14 = Conv2D(64, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv13))
    merge14 = concatenate([conv1,up14], axis = 3)
    conv14 = Conv2D(64, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(merge14)
    conv14 = Conv2D(64, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv14)
    
    conv14 = Conv2D(3, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')(conv14) #2,3
    conv_r = Conv2D(nb_ch, 1, activation = 'sigmoid')(conv14) 
    
    #############################defocus deblurring branch - late
    merge18 = concatenate([conv2,up8,up13,up18], axis = 3)
    conv18 = Conv2D(128, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    conv18.trainable=deblurring_branch_trainable
    conv18=conv18(merge18)
    
    conv18_ = Conv2D(128, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    conv18_.trainable=deblurring_branch_trainable
    conv18_=conv18_(conv18)

    up19 = Conv2D(64, 2, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')  
    up19.trainable=deblurring_branch_trainable
    up19=up19(UpSampling2D(size = (2,2))(conv18_)) 
    
    merge19 = concatenate([conv1,up19], axis = 3)
    conv19 = Conv2D(64, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    conv19.trainable=deblurring_branch_trainable
    conv19=conv19(merge19)
    
    conv19_ = Conv2D(64, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal')
    conv19_.trainable=deblurring_branch_trainable
    conv19_=conv19_(conv19)
    
    
    conv19__ = Conv2D(3, 3, activation = acti_str, padding = 'same', kernel_initializer = 'he_normal') #2,3
    conv19__.trainable=deblurring_branch_trainable
    conv19__=conv19__(conv19_)
    
    conv_s = Conv2D(nb_ch, 1, activation = 'sigmoid')
    conv_s.trainable=deblurring_branch_trainable
    conv_s=conv_s(conv19__)

    conv_out=concatenate([conv_l,conv_r,conv_s], axis = 3)

    return [conv_out,conv_out,conv_out,conv_out]