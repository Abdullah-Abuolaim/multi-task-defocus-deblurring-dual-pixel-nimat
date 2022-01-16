# Improving Single-Image Defocus Deblurring: How Dual-Pixel Images Help Through Multi-Task Learning

<p align="center">
												<a href="https://www.eecs.yorku.ca/~abuolaim/">Abdullah Abuolaim</a>
			&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 	<a href="https://sites.google.com/view/mafifi/">Mhamoud Afifi</a>
			&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;	<a href="https://www.eecs.yorku.ca/~mbrown/">Michael S. Brown</a>
	<br>
	York University
</p>

<img src="./figures/MDP-NIMAT-fast-2.gif" width="100%" alt="teaser gif">

Reference github repository for the paper [Improving Single-Image Defocus Deblurring: How Dual-Pixel Images Help Through Multi-Task Learning](https://arxiv.org/pdf/2108.05251.pdf). Abuolaim et al., proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2022. If you use our dataset or code, please cite our paper:
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
Will be available soon.

## Contact

Should you have any question/suggestion, please feel free to reach out:

[Abdullah Abuolaim](http://www.eecs.yorku.ca/~abuolaim/) (abuolaim@eecs.yorku.ca).

## Related Links
* ECCV'18 paper: Revisiting Autofocus for Smartphone Cameras &nbsp; [[project page](https://www.eecs.yorku.ca/~abuolaim/eccv_2018_autofocus/)]
* WACV'20 paper: Online Lens Motion Smoothing for Video Autofocus &nbsp; [[project page](https://www.eecs.yorku.ca/~abuolaim/wacv_2020_autofocus_lens_motion/)]
* ICCP'20 paper: Modeling Defocus-Disparity in Dual-Pixel Sensors &nbsp; [[github](https://github.com/abhijithpunnappurath/dual-pixel-defocus-disparity)]
* ECCV'20 paper: Defocus Deblurring Using Dual-Pixel Data &nbsp; [[project page](https://www.eecs.yorku.ca/~abuolaim/eccv_2020_dp_defocus_deblurring/)] &nbsp; [[github](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)]
* ICCV'21 paper: Learning to Reduce Defocus Blur by Realistically Modeling Dual-Pixel Data &nbsp; [[github](https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel)]
* WACVW'22 paper: Multi-View Motion Synthesis via Applying Rotated Dual-Pixel Blur Kernels &nbsp; [[pdf](https://arxiv.org/pdf/2111.07837.pdf)]

## References
[1] Abdullah Abuolaim and Michael S. Brown. *Defocus Deblurring Using Dual-Pixel Data.* In ECCV, 2020.
