"""
This code imports the modules and starts the implementation based on the
configurations in config.py module.

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

from model import *
from config import *
from data import *

if op_phase=='train':
    check_dir(path_write)
    check_dir(log_path)
    
    data_random_shuffling('train')
    data_random_shuffling('val')
    in_data = Input(batch_shape=(None, patch_h, patch_w, nb_ch))

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        m = Model(inputs=in_data, outputs=unet(in_data))
    
        if continue_checkpoint:
            m.load_weights(path_to_pretrained_model)
            print('***************Model: A Pre-Trained Checkpoint Model Has Been Loaded***************')
            
        m.summary()
        
        m.compile(optimizer = Adam(lr = lr_[0]), loss = [mse_s,mse_l_r,L_C_loss,L_D_loss],loss_weights=loss_weights_all)
        print('**********************************Network loss: MSE(s), MSE(l,r), LC, LD*********************************')
        
    # training callbacks
    model_checkpoint = ModelCheckpoint(path_save_model, monitor='loss',
                            verbose=1, save_best_only=True)
    tensorboard_callback = TensorBoard(log_path, histogram_freq=1, write_graph=False, profile_batch=0)
    m._get_distribution_strategy = lambda: None
    tbi_callback = TensorBoardImage('Image Example')
    l_r_scheduler_callback = LearningRateScheduler(schedule=schedule_learning_rate)

    history = m.fit_generator(generator('train'), nb_train, nb_epoch,
                        validation_data=generator('val'),
                        validation_steps=nb_val,callbacks=[model_checkpoint,
                        l_r_scheduler_callback, tensorboard_callback,tbi_callback])
    
    np.save(path_write+'train_loss_arr',history.history['loss'])
    np.save(path_write+'val_loss_arr',history.history['val_loss'])

elif op_phase=='test':
    check_dir(path_write)
    data_random_shuffling('test')

    in_data = Input(batch_shape=(None, None, None, nb_ch))

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        m = Model(inputs=in_data, outputs=unet(in_data))
        m.load_weights(path_read_model)
        
        for img_path_count in range(total_nb_test):
            img_name=(test_set[img_path_count][0]).split('/')[-1]
            img_ext=img_name.split('.')[-1]
            img_name=img_name.split('.'+img_ext)[0]
            
            print('********************Image: '+img_name+',  #'+str(img_path_count+1)+' out of ' +str(total_nb_test)+'********************')
            test_image, gt_image, temp_norm_val, temp_padding = test_generator_image(test_set[img_path_count])
            
            prediction_img = m.predict(test_image,batch_size=1,verbose=1)                   
            save_eval_predictions(path_write,test_image,prediction_img,gt_image,img_name,temp_norm_val,temp_padding)

    save_results_list('_l',mse_list_l,psnr_list_l,ssim_list_l,mae_list_l)
    save_results_list('_r',mse_list_r,psnr_list_r,ssim_list_r,mae_list_r)
    save_results_list('_s',mse_list_s,psnr_list_s,ssim_list_s,mae_list_s)

elif op_phase=='test_quick':
    check_dir(path_write)
    in_data = Input(batch_shape=(None, None, None, nb_ch))
    
    images_to_test = [path_read + f for f in os.listdir(path_read) if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))]
    images_to_test.sort()
    
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        m = Model(inputs=in_data, outputs=unet(in_data))
        m.load_weights(path_read_model)
        
        for img_path_count in range(total_nb_test):
            img_name=(images_to_test[img_path_count]).split('/')[-1]
            img_ext=img_name.split('.')[-1]
            img_name=img_name.split('.'+img_ext)[0]
            
            print('********************Image: '+img_name+',  #'+str(img_path_count+1)+' out of ' +str(total_nb_test)+'*************')
            test_image, gt_image, temp_norm_val, temp_padding = test_generator_image(images_to_test[img_path_count])
            
            prediction_img = m.predict(test_image,batch_size=1,verbose=1)                   
            save_eval_predictions(path_write,test_image,prediction_img,gt_image,img_name,temp_norm_val,temp_padding)

elif op_phase=='nimat':
    check_dir(path_write)
    in_data = Input(batch_shape=(None, None, None, nb_ch))
    
    images_to_test = [path_read + f for f in os.listdir(path_read) if f.endswith(('.jpg','.JPG', '.jpeg','.JPEG', '.png', '.PNG', '.TIF'))]
    images_to_test.sort()
    
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        m = Model(inputs=in_data, outputs=unet(in_data))
        m.load_weights(path_read_model)
        
        for img_path_count in range(total_nb_test):
            img_name=(images_to_test[img_path_count]).split('/')[-1]
            img_ext=img_name.split('.')[-1]
            img_name=img_name.split('.'+img_ext)[0]
            
            print('********************Image: '+img_name+',  #'+str(img_path_count+1)+' out of ' +str(total_nb_test)+'*************')
            test_images, temp_norm_val, temp_padding, temp_size_coordinates = test_generator_image_nimat(images_to_test[img_path_count])
            
            prediction_img = m.predict(test_images,batch_size=1,verbose=1)                   
            save_eval_predictions_nimat(path_write,prediction_img,img_name,temp_norm_val,temp_padding,temp_size_coordinates)