## Awesome-CIL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Geeks-Z.Awesome-CIL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Awesome-CIL" /> <img src="https://img.shields.io/github/license/Geeks-Z/Awesome-CIL" /></div>

## 🎉 Introduction

- **增量学习**：Continual Learning/Incremental Learning/Life-Long Learning

- **汇总**类增量学习（CIL，Class-Incremental Learning）的论文和代码，并对论文进行复现

- **论文阅读笔记**：[论文阅读](https://www.zhihu.com/people/icode1024/columns)

## 🚀 Survey

| Title                                                        | Venue | Year | Code                                                   |
| ------------------------------------------------------------ | ----- | ---- | ------------------------------------------------------ |
| [Class-Incremental Learning: A Survey](http://arxiv.org/abs/2302.03648) | TPAMI | 2024 | [Official](https://github.com/zhoudw-zdw/CIL_Surve)    |
| [Continual Learning with Pre-Trained Models: A Survey](http://arxiv.org/abs/2401.16386) | IJCAI | 2024 | [Official](https://github.com/sun-hailong/LAMDA-PILOT) |
| [PyCIL: A Python Toolbox for Class-Incremental Learning](https://arxiv.org/abs/2112.12533) |       |      | [Official](https://github.com/G-U-N/PyCIL)              |

## 🌟 Papers

| Title                                                        | Method      | Venue | Year | Type               | Code                                                    |
| ------------------------------------------------------------ | ----------- | ----- | ---- | ------------------ | ------------------------------------------------------- |
| [MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2412.09441) | MOS         | AAAI  | 2025 | PEFT Expansion     | [Official](https://github.com/sun-hailong/AAAI25-MOS)   |
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](http://arxiv.org/abs/2404.00228) | InfLoRA     | CVPR  | 2024 | PTM                | [Official](https://github.com/liangyanshuo/InfLoRA)     |
| [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2403.12030) | EASE        | CVPR  | 2024 | PEFT Expansion     | [Official](https://github.com/sun-hailong/CVPR24-Ease)  |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) | APER        | IJCV  | 2024 | PEFT Expansion     | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
| [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](http://arxiv.org/abs/2211.13218) | CODA-Prompt | CVPR  | 2023 | PEFT Expansion     | [Official](https://github.com/GT-RIPL/CODA-Prompt)      |
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | DualPrompt  | ECCV  | 2022 | PEFT Expansion     | [Official](https://github.com/google-research/l2p)      |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | L2P         | CVPR  | 2022 | PEFT Expansion     | [Official](https://github.com/google-research/l2p)      |
| [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) | FOSTER      | ECCV  | 2022 | Backbone Expansion | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      |
| [DER: Dynamically Expandable Representation for Class Incremental Learning](2021) | DER         | CVPR  | 2021 | Backbone Expansion | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      |
| [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725) | iCaRL       | CVPR  | 2017 | Memory             | [Official](https://github.com/srebuffi/iCaRL)           |

## 📚 Datasets

| Dataset       | training instances | testing instances | Classes | Link                                                         |
| ------------- | ------------------ | ----------------- | ------- | ------------------------------------------------------------ |
| CIFAR100      | 50,000             | 10,000            | 100     |                                                              |
| CUB200        | 9,430              | 2,358             | 200     |                                                              |
| ImageNet-R    | 24,000             | 6,000             | 200     |                                                              |
| ImageNet-A    | 5,981              | 1,519             | 200     |                                                              |
| ObjectNet     | 26,509             | 6,628             | 200     | [https://objectnet.dev/download.html](https://objectnet.dev/download.html) |
| Omnibenchmark | 89,697             | 5,983             | 300     |                                                              |
| VTAB          | 1,796              | 8,619             | 50      |                                                              |

## 📊 Reproduced Results

### Details

- **Code:** [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)

  <div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20241226204218.png" style="zoom: 80%;" /></div>

- ### Dataset split

  - `B-$m$ Inc-$n$' ：$m$代表初始增量阶段类别数量，$n$ 代表后续增量阶段类别数量；
  - LFH(learning from half)，表示在模型训练的初始阶段先用一半的类别进行训练，然后剩下一半的类别均匀分为 $N$ 个阶段进行训练；
  - LFS(learning from scratch)，表示所有的类别均匀地分为 $N$ 个阶段进行训练

- ### Backbone:`ViT-B/16-IN21K`

- ### Memory

  For exemplar parameters, DER, iCaRL and FOSTER set the `fixed_memory` option to false and retain the `memory_size` of 2000 for CIFAR100, while setting `fixed_memory` option to true and retaining the `memory_per_class` of 20 for ImageNet-R. On the contrary, other models are exemplar-free.

- **Dependencies**

  - pytorch 2.0.1
  - torchvision 0.15.2
  - timm 0.6.12
  - tqdm  4.65.0
  - numpy 1.21.5
  - scipy 1.10.1
  - easydict 1.13

### Results

> 实验结果：平均准确率（Accuracy）和标准差（Std）/ 论文中的结果

#### CIFAR-100



#### CUB-200



#### ImageNet-R



#### ImageNet-A



#### Omnibenchmark



#### VTAB



## 👨‍🏫 TODO

| Title                                                        | Venue | Year | Type            | Code                                                        |
| ------------------------------------------------------------ | ----- | ---- | --------------- | ----------------------------------------------------------- |
|                                                              |       |      |                 |                                                             |
| [FCS: Feature Calibration and Separation for Non-Exemplar Class Incremental Learning](https://ieeexplore.ieee.org/document/10657158/?arnumber=10657158) | CVPR  | 2024 | Feature Rectify | [Official](https://github.com/zhoujiahuan1991/CVPR2024-FCS) |
|                                                              |       |      |                 |                                                             |
|                                                              |       |      |                 |                                                             |

### Different PTMs

| PTM             | Pre-Trained Dataset | Finetuned Dataset |
| --------------- | ------------------- | ----------------- |
| ViT-B/16-IN1K   | ImageNet21K         | ImageNet1K        |
| ViT-B/16-IN21K  | ImageNet21K         | -                 |
| ViT-L/16-IN1K   | ImageNet21K         | ImageNet1K        |
| ViT-B/16-DINO   | ImageNet            | -                 |
| ViT-B/16-SAM    | SA-1B (Segment Anything 1 Billion Dataset) | - |
| ViT-B/16-MAE    | ImageNet21K         | -                 |
| ViT-B/16-CLIP   | OpenAI CLIP Dataset (a large corpus of text-image pairs) | - |
| ResNet18/50/152 | ImageNet1K          | -                 |

### Parameters

> 20Epoch batch_size=48 memory_size: 2000

| Method      | Tunable Parameters（Backbone） | All Parameters | Average Accuracy (%)<br />(CIFAR B0 Inc5) |
| ----------- | ------------------------------ | -------------- | ----------------------------------------- |
| Ease        |                                |                |                                           |
| SimpleCIL   |                                |                |                                           |
| CODA-Prompt |                                |                |                                           |
| DualPrompt  |                                |                |                                           |
| L2P         |                                |                |                                           |
| FOSTER      |                                |                |                                           |
| DER         |                                |                |                                           |
| iCaRL       |                                |                |                                           |
| Finetune    |                                |                |                                           |

## 🤗 Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
