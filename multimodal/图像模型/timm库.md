
# 概览

- PyTorch Image Models (timm)
- 旨在将各种 SOTA 模型、图像实用工具、常用的优化器、训练策略等视觉相关常用函数的整合在一起，并具有复现ImageNet训练结果的能力。
- 图像模型（models）、层（layers）、实用程序（utilities）、优化器（optimizers）、调度器（schedulers）、数据加载/增强（data-loaders / augmentations）和参考训练/验证脚本（reference training / validation scripts）的集合

## 官方库
- https://github.com/huggingface/pytorch-image-models#introduction

## 官方文档
- https://fastai.github.io/timmdocs/

## 中文简单介绍
- https://blog.csdn.net/weixin_44966641/article/details/121364784
- https://blog.csdn.net/weixin_44966641/article/details/121364784
- https://jishuin.proginn.com/p/763bfbd64d09

# 创建模型API

- create_model 函数是用来创建一个网络模型（如 ResNet、ViT 等），timm 库本身可供直接调用的模型已有接近400个，用户也可以自己实现一些模型并注册进 timm，供自己调用
- list_models函数可以查看timm所提供的模型列表，即可直接创建、有预训练的模型列表

## create_model函数参数

-  pretrained ：True or False
   -  ue:timm直接根据对应的URL下载模型权重参数并加载到模型，注意，只有本地没有对应模型参数时才会下载，也就是说，通常在第一次运行时下载对应模型参数，之后会直接从本地加载模型权重参数。（报错经验：这里容易出现的报错，就是第一次下载时，可能因为网络等原因，下载不完全，因此，有时需要删除对应文件，重新下载）
