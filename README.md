
# Self-supervised Representations in Reinforcement Learning on Pixel-Based Observations with Distractors

**Authors**: [Utkarsh A. Mishra](https://utkarshmishra04.github.io) and [Manan Tomar](https://manantomar.github.io)

**Description**:

This repository is a collection of widely used self-supervised auxiliary functions used for learning representations in reinforcement learning. Learning representations for pixel-based control has gained significant attention recently in reinforcement learning. A wide range of methods have been proposed to enable efficient learning. Such methods can be broadly classsified into few categories based on the auxiliary loss used, namely, with State metric ([DBC](https://github.com/facebookresearch/deep_bisim4control)), Reconstruction ([DREAMER](https://github.com/google-research/dreamer), [TIA](https://github.com/kyonofx/tia)), Contrastive ([CURL](https://github.com/MishaLaskin/curl)) and Non-Contrastives ([SPR](https://github.com/mila-iqia/spr)) losses. 

![Baseline Architecture](./assets/baseline.png)

The approach in the repository uses a baseline architecture as shown in figure above (left) which is a very simple arrangement of reward and transition prediction modules in addition to the SAC Actor-Critic losses. There are 8 different experiments performed on a particular type of environment data named as the following cases:

- Case 0: Baseline only SAC, no reward and transition losses
- Case 1: Baseline, Reward through Transition
- Case 2: Baseline with only Transition loss
- Case 3: Baseline with only Reward loss
- Case 4: Baseline, Independent Reward and Transition
- Case 5: Baseline + Value Aware loss 
- Case 6: Baseline + Reconstruction loss
- Case 7: Baseline + Contrastive loss
- Case 8: Baseline + Non-Contrastive loss

All the above experiments were further performed on 4 different environment data:

- Type 1: Simple Pixel [DM-Control](https://github.com/deepmind/dm_control)
- Type 2: Pixel DM-Control with [Natural Driving Distrators](https://github.com/Showmax/kinetics-downloader) (Figure above - right-top)
- Type 3: Pixel DM-Control + Natural Driving Distrators + Camera position and zoom offsets (Figure above - right-bottom)
- Type 4: Augmented Pixel DM-Control + Natural Driving Distrators


## Usage:



### Running the natural video setting

You can download the Kinetics 400 dataset and grab the driving_car label from the train dataset to replicate our setup. Some instructions for downloading the dataset can be found here: [https://github.com/Showmax/kinetics-downloader](https://github.com/Showmax/kinetics-downloader).


## Citation:

If you find this useful you can cite this work as follows:

```
@misc{representation-learning-pixels,
  author = {Mishra, Utkarsh A. and Tomar, Manan},
  title = {Learning Representation in Reinforcement Learning on Pixel-Based Observations with Distractors},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/UtkarshMishra04/representation_learning_pixels}
}
```

## Acknowledgement:

We thank the authors of [RAD](https://github.com/MishaLaskin/rad), [CURL](https://github.com/MishaLaskin/curl) and [SPR](https://github.com/mila-iqia/spr) for their well-structured open source code.