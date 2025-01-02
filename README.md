# PyTorch Glow: Generative Flow with Invertible 1x1 Convolutions

![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)
[![Paper](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/1807.03039)

Glow is a normalizing flow model introduced by OpenAI that uses an invertible generative architecture.
Glow’s flow blocks consist of 3 components: act norm, 1x1 invertible convolutions and affine coupling layers.
<br></br>
This repository contains the complete workflow for training and testing Glow. All code was developed during the GenAI UCU course.
Here are presented:
- model implementation from scratch
- train script with hydra configs
- tensorboard logging
- DDP trainer
- tests with pytest
- CI using github actions
