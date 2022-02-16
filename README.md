# Improving Single-Image Defocus Deblurring: How Dual-Pixel Images Help Through Multi-Task Learning

<p align="center">
												<a href="https://abuolaim.nowaty.com/">Abdullah Abuolaim</a>
			&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 	<a href="https://sites.google.com/view/mafifi/">Mhamoud Afifi</a>
			&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;	<a href="https://www.eecs.yorku.ca/~mbrown/">Michael S. Brown</a>
	<br>
	York University
</p>

<img src="./figures/MDP-NIMAT-fast-2.gif" width="100%" alt="teaser gif">

Reference github repository for the paper [Improving Single-Image Defocus Deblurring: How Dual-Pixel Images Help Through Multi-Task Learning](https://arxiv.org/pdf/2108.05251.pdf). Abuolaim et al., proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2022 ([YouTube presentation](https://www.youtube.com/watch?v=eE9i81D3_Os)). If you use our dataset or code, please cite our paper:
```
@inproceedings{abuolaim2022improving,
  title={Improving Single-Image Defocus Deblurring: How Dual-Pixel Images Help Through Multi-Task Learning},
  author={Abuolaim, Abdullah and Afifi, Mahmoud and Brown, Michael S},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1231--1239},
  year={2022}
}
```
## DLDP Dataset
We collected a new diverse and large Dual-Pixel (DLDP) dataset of 2353 scenes.
This dataset consists of 7059 images i.e., 2353 images with their 4706 dual-pixel (DP) sub-aperture views.

* Dataset

    * [2090 images used for training](https://ln5.sync.com/dl/9eccac1a0/e62jckvx-kt5cv3gf-gebdm4am-bcfzfhm3) (processed to an sRGB and encoded with a lossless 16-bit depth).
    * [263 images used for testing](https://ln5.sync.com/dl/6ca5eb1f0/37nq8ssu-7vb9iwtz-agajda74-66trakjn) (processed to an sRGB and encoded with a lossless 16-bit depth).

* Training and testing sets
    * The dataset is divided randomly into:
		* 89% training and 11% testing.
	* Each set has a balanced number of indoor/outdoor scenes and aperture sizes.
	* The 500 scenes of the [DPDD dataset [1]](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) are added to the training set.

## Our New Image Motion Attribute (NIMAT) Effect
<img src="./figures/nimat.gif" width="100%" alt="NIMAT effect">

## Code and Trained Models
### Prerequisites
* The code tested with:
	* Python 3.8.3
	* TensorFlow 2.2.0
	* Keras 2.4.3
	* Numpy 1.19.1
	* Scipy 1.5.2
	* Scikit-image 0.16.2
	* Scikit-learn 0.23.2
	* OpenCV 4.4.0
	
	<i>Despite not tested, the code may work with library versions other than the specified</i>

### Installation
* Clone with HTTPS this project to your local machine 
```bash
git clone https://github.com/Abdullah-Abuolaim/multi-task-defocus-deblurring-dual-pixel-nimat.git
cd ./multi-task-defocus-deblurring-dual-pixel-nimat/mdp_code/
```

### Testing
* Download the final trained model of the second phase used in the main paper i.e., [mdp_phase_2_wacv](https://ln5.sync.com/dl/fc47788b0/fj48pyz3-4kg2k8hr-qiu2ae4b-9d8kt9jc)

* Place the downloaded `.hdf5` model inside `ModelCheckpoints` for testing

* Download the [DPDD dataset [1]](https://ln2.sync.com/dl/c45358c50/r7kpybwk-xw8hhszh-qkj249ap-y8k2344d), or visit project [GitHub](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)

* Run `main.py` in `mdp_code` directory as follows:
	```bash
	python main.py   --op_phase test   --path_to_data $PATH_TO_DPDD_DATA$   --test_model  mdp_phase_2_wacv
	```
	* <i>--training_phase</i>: training phase i.e., phase_1 or phase_2
	* <i>--op_phase</i>: operation phase training or testing
	* <i>--path_to_data</i>: path to the directory that has the DPDD data e.g., `./dd_dp_dataset_canon/`
	* <i>--test_model</i>: test model name
	* <i>--downscale_ratio</i>: downscale input test image in case the GPU memory is not sufficient, or use CPU instead
	* The results of the tested models will be saved in the `results` directory that will be created inside `mdp_code`

* The above testing is for evaluating  the DPDD dataset quantitively and quantitatively
	* This evaluation is for data that has ground truth
	* This testing is designed based on the directory structure from the [DPDD GitHub](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)

<i>Recall that you might need to change </i>

#### Quick Testing
* For quick qualitative testing of images in a directory, run the following command:
	```bash
	python main.py   --op_phase test_quick   --path_to_data $PATH_TO_DATA$   --test_model  mdp_phase_2_wacv
	```
	* <i>--path_to_data</i>: path to the directory that has the images (no ground truth)
	* <i>--downscale_ratio</i>: downscale input test image in case the GPU memory is not sufficient, or use CPU instead
	* The results of the tested models will be saved in the `results` directory that will be created inside `mdp_code`
	
<i>Recall that you might need to change </i>

### Training
* As described in the main paper, we train our multi-task DP network (MDP) in two phases:
	* First phase: freezing the weights of the deblurring <i>Decoder</i>, then, training with image patches from our new [DLDP dataset](https://ln5.sync.com/dl/9eccac1a0/e62jckvx-kt5cv3gf-gebdm4am-bcfzfhm3)  to optimize for the DP-view synthesis task.
	* Second phase: unfreezing the weights of the deblurring <i>Decoder</i>, then, fine-tuning using images from the [DPDD dataset [1]](https://ln2.sync.com/dl/c45358c50/r7kpybwk-xw8hhszh-qkj249ap-y8k2344d) to optimize jointly for both the defocus deblurring and DP-view synthesis tasks.

#### Phase 1 Training
* Run `main.py` in `mdp_code` directory to start phase 1 training:
	```bash
	python main.py   --training_phase  phase_1   --op_phase train   --path_to_data $PATH_TO_DLDP_DATA$
	```
	* <i>--path_to_data</i>: path to the directory that has the DLDP data e.g., `./dldp_data_png/`
	* The results of the tested models will be saved in the `results` directory that will be created inside `mdp_code`
	* The trained model and checkpoints will be saved in `ModelCheckpoints` after each epoch
	* A `TensorBoard` log for each training  session will be created at `logs` to provide the visualization and tooling needed to monitor the training

<i>Recall that you might need to change </i>

#### Phase 2 Training
* Download the final trained model of the first phase used in the main paper i.e., [mdp_phase_1_wacv](https://ln5.sync.com/dl/42a8a0ad0/8rxzfjiw-ckyzzauf-x3g7q244-bz8zx2ti)
* Place the downloaded `.hdf5` model inside `ModelCheckpoints` for training
* You need the first phase trained model to strat phase 2 training
* Run `main.py` in `mdp_code` directory to start phase 2 training:
	```bash
	python main.py   --training_phase  phase_2   --op_phase train   --path_to_data $PATH_TO_DPDD_DATA$   --phase_1_checkpoint_model  mdp_phase_1_wacv
	```
	* <i>--path_to_data</i>: path to the directory that has the DPDD data e.g., `./dd_dp_dataset_canon/`
	* <i>--phase_1_checkpoint_model</i>: the name of the pretrained model from phase 1 e.g., mdp_phase_1_wacv
	* The results of the tested models will be saved in the `results` directory that will be created inside `mdp_code`
	* The trained model and checkpoints will be saved in `ModelCheckpoints` after each epoch
	* A `TensorBoard` log for each training  session will be created at `logs` to provide the visualization and tooling needed to monitor the training

<i>Recall that you might need to change </i>

#### Other Training Options
* <i>--patch_size</i>: training patch size
* <i>--img_mini_b</i>: image mini-batch size
* <i>--epoch</i>: number of training epochs
* <i>--lr</i>: initial learning rate
* <i>--schedule_lr_rate</i>: learning rate scheduler (after how many epochs to decrease)
* <i>--bit_depth</i>: image bit depth datatype, 16 for `uint16` or 8 for `uint8`. Recall that we train with 16-bit images
* <i>--dropout_rate</i>: the dropout rate of the `conv` unit at the network bottleneck

### NIMAT Effect
* To generate eight DP-like sub-aperture views (i.e., NIMAT) for each image in a directory, run the following command:
	```bash
	python main.py   --op_phase nimat   --test_model  mdp_phase_2_wacv   --path_to_data $PATH_TO_DATA$
	```
	* <i>--path_to_data</i>: path to the directory that has the images (no ground truth)
	* <i>--downscale_ratio</i>: downscale input test image in case the GPU memory is not sufficient, or use CPU instead
	* The results of the tested models will be saved in the `results` directory that will be created inside `mdp_code`
	
<i>Recall that you might need to change </i>

## Contact

Should you have any question/suggestion, please feel free to reach out:

[Abdullah Abuolaim](http://abuolaim.nowaty.com/) (abuolaim@eecs.yorku.ca).

## Related Links
* ECCV'18 paper: Revisiting Autofocus for Smartphone Cameras &nbsp; [[project page](https://abuolaim.nowaty.com/eccv_2018_autofocus/)]
* WACV'20 paper: Online Lens Motion Smoothing for Video Autofocus &nbsp; [[project page](https://abuolaim.nowaty.com/wacv_2020_autofocus_lens_motion/)] &nbsp; [[presentation](https://www.youtube.com/watch?v=85z075A3rI0)]
* ICCP'20 paper: Modeling Defocus-Disparity in Dual-Pixel Sensors &nbsp; [[github](https://github.com/abhijithpunnappurath/dual-pixel-defocus-disparity)] &nbsp; [[presentation](https://www.youtube.com/watch?v=Ow2ffrqjPiI)]
* ECCV'20 paper: Defocus Deblurring Using Dual-Pixel Data &nbsp; [[project page](https://abuolaim.nowaty.com/eccv_2020_dp_defocus_deblurring/)] &nbsp; [[github](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)] &nbsp; [[presentation](https://www.youtube.com/watch?v=xb12cFiB8ww)]
* ICCV'21 paper: Learning to Reduce Defocus Blur by Realistically Modeling Dual-Pixel Data &nbsp; [[github](https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel)] &nbsp; [[presentation](https://www.youtube.com/watch?v=SxLgE3xwBAQ)]
* CVPRW'21 paper: NTIRE 2021 Challenge for Defocus Deblurring Using Dual-pixel Images: Methods and Results &nbsp; [[pdf](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Abuolaim_NTIRE_2021_Challenge_for_Defocus_Deblurring_Using_Dual-Pixel_Images_Methods_CVPRW_2021_paper.pdf)] &nbsp; [[presentation](https://www.youtube.com/watch?v=OC52DLyz1lU)]
* WACVW'22 paper: Multi-View Motion Synthesis via Applying Rotated Dual-Pixel Blur Kernels &nbsp; [[pdf](https://arxiv.org/pdf/2111.07837.pdf)] &nbsp; [[presentation](https://youtu.be/eIquI76r0dw)]

## References
[1] Abdullah Abuolaim and Michael S. Brown. *Defocus Deblurring Using Dual-Pixel Data.* In ECCV, 2020.
