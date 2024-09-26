## Awesome-CIL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Geeks-Z.Awesome-CIL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Awesome-CIL" /> <img src="https://img.shields.io/github/license/Geeks-Z/Awesome-CIL" /></div>

## 🎉 Introduction

CIL: Class-Incremental Learning
> Incremental Learning: Life-Long Learning/Continual Learning

## 🚀 Survey

| Title                                                        | Venue | Year | Code                                                   |
| ------------------------------------------------------------ | ----- | ---- | ------------------------------------------------------ |
| [Class-Incremental Learning: A Survey](http://arxiv.org/abs/2302.03648) | TPAMI | 2024 | [Official](https://github.com/zhoudw-zdw/CIL_Surve)    |
| [Continual Learning with Pre-Trained Models: A Survey](http://arxiv.org/abs/2401.16386) | IJCAI | 2024 | [Official](https://github.com/sun-hailong/LAMDA-PILOT) |
| [PyCIL: A Python Toolbox for Class-Incremental Learning](https://arxiv.org/abs/2112.12533) |       |      | [Official](https://github.com/G-U-N/PyCIL              |

## 🌟 Papers

| Title                                                        | Venue | Year | Type   | Code                                                    |
| ------------------------------------------------------------ | ----- | ---- | ------ | ------------------------------------------------------- |
| [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2403.12030) | CVPR  | 2024 | PTM    | [Official](https://github.com/sun-hailong/CVPR24-Ease)  |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) | IJCV  | 2024 | PTM    | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
| [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](http://arxiv.org/abs/2211.13218) | CVPR  | 2023 | PTM    | [Official](https://github.com/GT-RIPL/CODA-Prompt)      |
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | ECCV  | 2022 | PTM    | [Official](https://github.com/google-research/l2p)      |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | CVPR  | 2022 | PTM    | [Official](https://github.com/google-research/l2p)      |
| [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) | ECCV  | 2022 | Mixed  | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      |
| [DER: Dynamically Expandable Representation for Class Incremental Learning](2021) | CVPR  | 2021 | Mixed  | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      |
| [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725) | CVPR  | 2017 | Memory | [Official](https://github.com/srebuffi/iCaRL)           |

## 📚 Datasets

| Dataset       | training instances | testing instances | Classes | Link |
| ------------- | ------------------ | ----------------- | ------- | ---- |
| CIFAR100      | 50,000             | 10,000            | 100     |      |
| CUB200        | 9,430              | 2,358             | 200     |      |
| ImageNet-R    | 24,000             | 6,000             | 200     |      |
| ImageNet-A    | 5,981              | 1,519             | 200     |      |
| ObjectNet     | 26,509             | 6,628             | 200     |      |
| Omnibenchmark | 89,697             | 5,983             | 300     |      |
| VTAB          | 1,796              | 8,619             | 50      |      |

## 📝 Reproduced Results

- class split: `B-$m$ Inc-$n$' . $m$ represents the number of categories in the initial incremental task, while $n$ denotes the number of subsequent incremental tasks, with categories in these tasks evenly distributed. If $m = 0$, all categories in the dataset are evenly distributed across $n$ incremental tasks.
- pre-trained backbone: ViT-B/16-IN21K 
- log: 'LAMDA-PILOT-main/res'
- accuracy：CNN/NME
- code: `📁 LAMDA-PILOT-main`

### CIFAR-100

|             | B0 Inc5     | B0 Inc10    | B0 Inc20    | B50 Inc5    | B50 Inc10   |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |


### CUB-200

> fixed_memory: True

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240926214357.png" style="zoom: 60%;" /></div>

|             | B0 Inc5     | B0 Inc10    | B0 Inc20    | B100 Inc5   | B100 Inc10  |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|Ease |90.14 ± 0.05| 90.21 ± 0.01| 91.32 ± 0.01| 85.26 ± 0.02| 87.44 ± 0.07|
|CODA-Prompt |85.78 ± 0.09|84.93 ± 0.05|83.03 ± 0.08|79.59 ± 0.03|76.44 ± 0.14|
|DualPrompt |85.5 ± 0.03|84.84 ± 0.04|82.9 ± 0.08|77.01 ± 0.07|73.8 ± 0.1|
|L2P |83.74 ± 0.04|81.8 ± 0.05|79.33 ± 0.01|79.15 ± 0.18|74.65 ± 0.02|
|FOSTER-CNN |78.81 ± 0.53|81.18 ± 0.36|85.17 ± 0.64|82.65 ± 0.25|85.56 ± 0.01|
|FOSTER-NME |90.65 ± 0.05|91.02 ± 0.16|91.4 ± 0.34|88.92 ± 0.18|88.72 ± 0.02|
|DER-CNN |89.08 ± 0.48|88.71 ± 0.86|90.13 ± 0.05|87.14 ± 0.2|86.72 ± 0.06|
|DER-NME |89.89 ± 0.19|89.43 ± 0.71|90.99 ± 0.05|88.62 ± 0.11|88.34 ± 0.21|
|iCaRL-CNN |87.77 ± 0.25|88.2 ± 0.13|88.2 ± 0.36|85.93 ± 0.32|85.4 ± 0.42|
|iCaRL-NME |89.06 ± 0.04|89.94 ± 0.13|90.2 ± 0.21|87.91 ± 0.14|87.25 ± 0.28|
|SimpleCIL |89.92 ± 0.0|90.57 ± 0.01|90.96 ± 0.0|87.51 ± 0.04|87.43 ± 0.04|
|Finetune |77.36 ± 0.49|70.94 ± 0.74|60.21 ± 2.64|72.88 ± 0.42|62.32 ± 2.26|

### ImageNet-R

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240926214533.png" style="zoom: 60%;" /></div>

|             | B0 Inc5 | B0 Inc10 | B0 Inc20 | B100 Inc5 | B100 Inc10 |
| ----------- | ------- | -------- | -------- | --------- | ---------- |
|Ease |81.55 ± 0.04| 81.01 ± 0.07| 78.88 ± 0.01| 77.96 ± 0.01| 76.58 ± 0.01|
|CODA-Prompt |79.9 ± 0.06|78.72 ± 0.06|74.45 ± 0.13|73.1 ± 0.03|67.86 ± 0.02|
|DualPrompt |73.54 ± 0.02|71.16 ± 0.13|69.84 ± 0.06|65.04 ± 0.16|59.73 ± 0.13|
|L2P |76.88 ± 0.02|76.31 ± 0.04|73.69 ± 0.01|69.77 ± 0.05|64.88 ± 0.0|
|SimpleCIL |65.86 ± 0.02|67.07 ± 0.01|67.59 ± 0.01|63.53 ± 0.01|63.41 ± 0.01|
|Finetune |72.24 ± 0.42|68.32 ± 0.46|61.95 ± 1.74|70.73 ± 0.02|66.72 ± 0.28|

### ImageNet-A

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240926214602.png" style="zoom: 60%;" /></div>

|             | B0 Inc5 | B0 Inc10 | B0 Inc20 | B100 Inc5 | B100 Inc10 |
| ----------- | ------- | -------- | -------- | --------- | ---------- |
|Ease |67.84 ± 0.08| 62.67 ± 0.06| 58.13 ± 0.32| 63.55 ± 0.35| 62.18 ± 0.0|
|CODA-Prompt |60.99 ± 0.01|56.31 ± 0.17|47.62 ± 0.11|56.24 ± 0.26|53.2 ± 0.2|
|DualPrompt |54.16 ± 0.1|52.43 ± 0.03|48.59 ± 0.1|47.02 ± 0.11|45.05 ± 0.33|
|L2P |53.41 ± 0.12|52.68 ± 0.08|45.93 ± 0.18|49.32 ± 0.14|46.85 ± 0.08|
|SimpleCIL |58.38 ± 0.21|59.56 ± 0.16|60.35 ± 0.21|53.51 ± 0.16|53.43 ± 0.16|
|Finetune |38.03 ± 0.35|33.73 ± 1.2|15.26 ± 5.64|30.32 ± 0.59|24.65 ± 0.08|

### Omnibenchmark

|             | B0 Inc5     | B0 Inc10    | B0 Inc20    | B0 Inc30    | B150 Inc5   | B150 Inc10  |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |


### VTAB

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20240922193144.png" style="zoom: 60%;" /></div>

|             | B0 Inc5 | B0 Inc10 |
| ----------- | ------- | -------- |
|Ease |90.47 ± 0.0| 91.07 ± 0.0|
|CODA-Prompt |87.82 ± 0.0|83.36 ± 0.0|
|DualPrompt |89.96 ± 0.0|87.61 ± 0.0|
|L2P |82.11 ± 0.0|78.04 ± 0.0|
|SimpleCIL |90.94 ± 0.0|91.66 ± 0.0|
|Finetune |78.69 ± 0.0|59.2 ± 0.0|

## 👨‍🏫  TODO

| Title                                                        | Venue | Year | Type | Code                                                |
| ------------------------------------------------------------ | ----- | ---- | ---- | --------------------------------------------------- |
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](http://arxiv.org/abs/2404.00228) | CVPR  | 2024 | PTM  | [Official](https://github.com/liangyanshuo/InfLoRA) |
|                                                              |       |      |      |                                                     |

### ☄️Parameters

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

## 

## 🤗Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
