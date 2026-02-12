## Awesome-CIL

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" style="zoom: 60%;" /></div>
<p></p>
<div align=center><img src="https://visitor-badge.laobi.icu/badge?page_id=Geeks-Z.Awesome-CIL&left_color=green&right_color=red" /> <img src="https://img.shields.io/github/last-commit/Geeks-Z/Awesome-CIL" /> <img src="https://img.shields.io/github/license/Geeks-Z/Awesome-CIL" /></div>

## 🎉 Introduction

- **增量学习**：Continual Learning/Incremental Learning/Life-Long Learning

- **汇总**类增量学习（CIL，Class-Incremental Learning）的论文和代码，并对论文进行复现

- **论文阅读笔记**：[论文阅读](https://www.zhihu.com/people/icode1024/columns)

---

## 🚀 Survey

| Title                                                        | Venue | Year | Code                                                   |
| ------------------------------------------------------------ | ----- | ---- | ------------------------------------------------------ |
| [Class-Incremental Learning: A Survey](http://arxiv.org/abs/2302.03648) | TPAMI | 2024 | [Official](https://github.com/zhoudw-zdw/CIL_Surve)    |
| [Continual Learning with Pre-Trained Models: A Survey](http://arxiv.org/abs/2401.16386) | IJCAI | 2024 | [Official](https://github.com/sun-hailong/LAMDA-PILOT) |
| [PyCIL: A Python Toolbox for Class-Incremental Learning](https://arxiv.org/abs/2112.12533) |       |      | [Official](https://github.com/G-U-N/PyCIL)              |

---

## 🌟 Papers

| Title                                                        | Method      | Venue | Year | Type               | Code                                                    |
| ------------------------------------------------------------ | ----------- | ----- | ---- | ------------------ | ------------------------------------------------------- |
| [MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2412.09441) | MOS         | AAAI  | 2025 | PEFT Expansion     | [Official](https://github.com/sun-hailong/AAAI25-MOS)   |
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](http://arxiv.org/abs/2404.00228) | InfLoRA     | CVPR  | 2024 | PTM                | [Official](https://github.com/liangyanshuo/InfLoRA)     |
| [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2403.12030) | EASE        | CVPR  | 2024 | PEFT Expansion     | [Official](https://github.com/sun-hailong/CVPR24-Ease)  |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) | APER        | IJCV  | 2024 | PEFT Expansion     | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
| [A Unified Continual Learning Framework with General Parameter-Efficient Tuning](http://arxiv.org/abs/2303.10070) | LAE         | ICCV  | 2023 | PEFT Expansion     |                                                         |
| [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](http://arxiv.org/abs/2211.13218) | CODA-Prompt | CVPR  | 2023 | PEFT Expansion     | [Official](https://github.com/GT-RIPL/CODA-Prompt)      |
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | DualPrompt  | ECCV  | 2022 | PEFT Expansion     | [Official](https://github.com/google-research/l2p)      |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | L2P         | CVPR  | 2022 | PEFT Expansion     | [Official](https://github.com/google-research/l2p)      |
| [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) | FOSTER      | ECCV  | 2022 | Backbone Expansion | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      |
| [DER: Dynamically Expandable Representation for Class Incremental Learning](2021) | DER         | CVPR  | 2021 | Backbone Expansion | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      |
| [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725) | iCaRL       | CVPR  | 2017 | Memory             | [Official](https://github.com/srebuffi/iCaRL)           |

---

## 📚 Datasets

| Dataset            | training instances | testing instances | Classes | Link                                                         | Abstract                                 |
| ------------------ | ------------------ | ----------------- | ------- | ------------------------------------------------------------ | ---------------------------------------- |
| CIFAR100           | 50,000             | 10,000            | 100     |                                                              |                                          |
| CUB（CUB200-2011） | 9,430              | 2,358             | 200     |                                                              | 加州理工学院2010年提出的鸟类细粒度数据集 |
| ImageNet-R         | 24,000             | 6,000             | 200     |                                                              |                                          |
| ImageNet-A         | 5,981              | 1,519             | 200     |                                                              |                                          |
| ObjectNet          | 26,509             | 6,628             | 200     | [https://objectnet.dev/download.html](https://objectnet.dev/download.html) |                                          |
| Omnibenchmark      | 89,697             | 5,983             | 300     |                                                              |                                          |
| VTAB               | 1,796              | 8,619             | 50      |                                                              |                                          |

---

## 📊 Reproduced Results

### Details

- **Code:** [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)

  <div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20241226204218.png" style="zoom: 80%;" /></div>

- ### Dataset split

  - `B-$m$ Inc-$n$' ：$m$代表初始增量阶段类别数量，$n$ 代表后续每个增量阶段的类别数量；
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

> 实验结果：平均准确率（Accuracy）± 标准差（Std）/ 原文结果

#### CIFAR-100

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050946577.png" alt="image-20250705094642483" style="zoom:67%;" />

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B50 Inc5    | B100 Inc10   |
| --------- | ----------- | ------------ | ----------- | ------------ | ------------ |
|Inflora |89.55 ± 0.09|92.01 ± 0.13|93.22 ± 0.02|69.84 ± 0.15|75.12 ± 0.17|
|APER |90.22 ± 0.0|92.12 ± 0.02|92.51 ± 0.0|91.94 ± 0.01|91.95 ± 0.01|
|MOS |93.43 ± 0.01|94.76 ± 0.06|94.8 ± 0.0|94.18 ± 0.02|94.1 ± 0.01|
|Ease |91.64 ± 0.02| 92.56 ± 0.02| 93.09 ± 0.02| 89.26 ± 0.04| 90.41 ± 0.05|
|LAE |81.94 ± 0.0|90.54 ± 0.23|91.33 ± 0.0|88.12 ± 0.0|90.4 ± 0.0|
|CODA-Prompt |87.34 ± 0.0|91.31 ± 0.0|92.72 ± 0.0|82.79 ± 0.0|88.43 ± 0.0|
|DualPrompt |88.44 ± 0.04|90.32 ± 0.06|91.37 ± 0.01|80.83 ± 0.11|87.42 ± 0.03|
|L2P |87.57 ± 0.02|89.78 ± 0.01|90.76 ± 0.03|79.46 ± 0.09|87.62 ± 0.04|
|SimpleCIL |87.57 ± 0.0|87.13 ± 0.0|86.11 ± 0.0|83.79 ± 0.01|83.89 ± 0.01|
|Finetune |72.32 ± 0.28|76.94 ± 0.02|80.87 ± 0.31|80.29 ± 0.14|82.63 ± 0.12|
|FOSTER-CNN |94.02 ± 0.0|93.9 ± 0.05|93.61 ± 0.08|92.04 ± 0.02|92.26 ± 0.07|
|FOSTER-NME |94.19 ± 0.0|94.09 ± 0.01|93.62 ± 0.05|92.19 ± 0.09|92.26 ± 0.07|
|DER-CNN |88.67 ± 0.15|88.6 ± 0.02|88.68 ± 0.14|86.54 ± 0.03|86.86 ± 0.04|
|DER-NME |90.95 ± 0.14|91.05 ± 0.06|91.27 ± 0.13|88.92 ± 0.09|89.38 ± 0.0|
|iCaRL-CNN |84.6 ± 0.02|85.87 ± 0.16|86.91 ± 0.18|81.67 ± 0.03|83.54 ± 0.24|
|iCaRL-NME |89.52 ± 0.08|90.3 ± 0.05|90.64 ± 0.16|87.55 ± 0.0|88.98 ± 0.02|

#### CUB-200

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050947215.png" alt="image-20250705094735156" style="zoom:67%;" />

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B0 Inc40 | B100 Inc5    | B100 Inc10   | B100 Inc20 |
| --------- | ----------- | ------------ | ----------- | -------- | ------------ | ------------ | ---------- |
|Inflora |69.65 ± 0.32|74.61 ± 0.21|83.54 ± 0.17|86.31 ± 0.09|39.91 ± 0.5|55.33 ± 0.79|67.86 ± 0.08|
|MOS |93.78 ± 0.0|93.53 ± 0.0|93.24 ± 0.02|92.51 ± 0.01|91.53 ± 0.02|91.39 ± 0.03|91.46 ± 0.04|
|APER |92.52 ± 0.0|92.27 ± 0.0|92.06 ± 0.0|91.82 ± 0.0|89.64 ± 0.0|89.66 ± 0.0|89.73 ± 0.0|
|Ease |92.3 ± 0.0| 91.33 ± 0.01| 90.19 ± 0.01| 90.17 ± 0.02| 86.07 ± 0.06| 87.47 ± 0.06| 85.28 ± 0.01|
|LAE |61.07 ± 0.0|73.28 ± 0.0|81.15 ± 0.0|83.67 ± 0.0|68.29 ± 0.0|70.81 ± 0.0|74.66 ± 0.0|
|CODA-Prompt |71.59 ± 0.0|83.25 ± 0.0|84.65 ± 0.0|85.69 ± 0.05|69.63 ± 0.02|77.89 ± 0.02|82.39 ± 0.07|
|DualPrompt |80.82 ± 0.03|83.07 ± 0.06|84.81 ± 0.02|85.44 ± 0.07|71.06 ± 0.09|73.7 ± 0.08|76.75 ± 0.12|
|L2P |71.63 ± 0.04|79.27 ± 0.03|81.8 ± 0.03|83.78 ± 0.02|70.67 ± 0.03|74.62 ± 0.0|79.41 ± 0.01|
|SimpleCIL |92.4 ± 0.12|92.23 ± 0.0|91.85 ± 0.0|91.29 ± 0.0|88.98 ± 0.0|88.99 ± 0.0|89.05 ± 0.0|
|Finetune |31.13 ± 19.92|64.95 ± 0.35|70.35 ± 0.74|78.54 ± 0.05|46.33 ± 0.37|63.3 ± 0.44|73.32 ± 0.06|
|FOSTER-CNN |87.31 ± 0.33|85.82 ± 0.3|81.12 ± 0.6|78.48 ± 0.1|86.92 ± 0.13|85.85 ± 0.17|82.95 ± 0.05|
|FOSTER-NME |91.28 ± 0.27|91.69 ± 0.28|91.15 ± 0.14|90.72 ± 0.2|88.97 ± 0.01|89.06 ± 0.16|89.03 ± 0.19|
|DER-CNN |89.83 ± 0.08|89.58 ± 0.03|89.11 ± 0.14|89.2 ± 0.37|87.05 ± 0.06|87.07 ± 0.25|86.82 ± 0.08|
|DER-NME |90.59 ± 0.0|90.67 ± 0.29|89.93 ± 0.35|89.78 ± 0.03|87.83 ± 0.02|88.28 ± 0.04|88.36 ± 0.13|
|iCaRL-CNN |87.83 ± 0.0|88.12 ± 0.08|88.33 ± 0.09|87.51 ± 0.12|84.47 ± 0.05|85.44 ± 0.05|85.84 ± 0.09|
|iCaRL-NME |90.12 ± 0.0|90.19 ± 0.41|90.1 ± 0.04|88.99 ± 0.18|87.17 ± 0.21|87.6 ± 0.1|88.12 ± 0.01|


#### ImageNet-R

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050948418.png" alt="image-20250705094828355" style="zoom:67%;" />

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B0 Inc40 | B100 Inc5    | B100 Inc10   | B100 Inc20 |
| --------- | ----------- | ------------ | ----------- | -------- | ------------ | ------------ | ---------- |
|Inflora |72.95 ± 0.52|79.91 ± 0.27|83.06 ± 0.08|83.1 ± 0.2|51.12 ± 0.21|58.63 ± 0.17|65.64 ± 0.03|
|APER |71.19 ± 0.0|74.37 ± 0.08|75.17 ± 0.02|75.28 ± 0.0|73.99 ± 0.04|73.96 ± 0.04|74.03 ± 0.03|
|MOS |77.13 ± 0.04|81.24 ± 0.07|82.54 ± 0.07|82.72 ± 0.07|81.0 ± 0.02|81.07 ± 0.03|81.05 ± 0.0|
|Ease |67.81 ± 0.0| 78.94 ± 0.04| 80.95 ± 0.08| 81.53 ± 0.05| 71.4 ± 0.0| 76.59 ± 0.04| 77.98 ± 0.04|
|LAE |58.91 ± 0.06|70.41 ± 0.0|75.57 ± 0.05|76.57 ± 0.0|60.92 ± 0.0|67.13 ± 0.0|70.7 ± 0.0|
|CODA-Prompt |59.61 ± 0.0|72.48 ± 0.01|78.17 ± 0.03|80.46 ± 0.04|65.85 ± 0.05|71.61 ± 0.08|75.17 ± 0.04|
|DualPrompt |63.08 ± 0.12|69.9 ± 0.04|71.03 ± 0.05|73.58 ± 0.05|55.17 ± 0.14|59.69 ± 0.11|64.9 ± 0.06|
|L2P |70.48 ± 0.05|73.76 ± 0.04|76.23 ± 0.04|76.76 ± 0.06|59.58 ± 0.02|64.89 ± 0.05|69.72 ± 0.08|
|SimpleCIL |62.42 ± 0.04|61.85 ± 0.03|61.14 ± 0.03|59.72 ± 0.03|56.71 ± 0.01|56.7 ± 0.0|56.83 ± 0.01|
|Finetune |56.56 ± 0.22|63.8 ± 0.19|69.03 ± 0.02|72.19 ± 0.12|61.24 ± 2.76|67.13 ± 0.29|70.67 ± 0.1|


#### ImageNet-A

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050949218.png" alt="image-20250705094905159" style="zoom:67%;" />

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B0 Inc40 | B100 Inc5    | B100 Inc10   | B100 Inc20 |
| --------- | ----------- | ------------ | ----------- | -------- | ------------ | ------------ | ---------- |
|Inflora |40.51 ± 0.27|51.19 ± 0.59|59.17 ± 0.68|63.42 ± 0.41|33.75 ± 0.68|43.08 ± 0.48|51.09 ± 0.01|
|APER |61.24 ± 0.0|60.69 ± 0.0|60.02 ± 0.0|60.0 ± 0.0|61.36 ± 0.0|61.4 ± 0.0|61.48 ± 0.0|
|MOS |64.26 ± 0.0|64.29 ± 0.0|62.96 ± 2.95|65.42 ± 0.0|64.57 ± 0.07|64.89 ± 0.08|64.9 ± 0.04|
|Ease |60.18 ± 0.04| 57.69 ± 0.02| 62.6 ± 0.11| 67.73 ± 0.03| 62.14 ± 0.08| 62.35 ± 0.23| 63.15 ± 0.15|
|LAE |26.87 ± 0.0|38.31 ± 0.0|48.61 ± 0.0|54.76 ± 0.0|45.25 ± 0.0|48.73 ± 0.0|51.35 ± 0.0|
|CODA-Prompt |59.61 ± 0.0|72.48 ± 0.01|78.17 ± 0.03|80.46 ± 0.04|65.85 ± 0.05|71.61 ± 0.08|75.17 ± 0.04|
|DualPrompt |43.62 ± 0.03|48.48 ± 0.06|52.51 ± 0.06|54.02 ± 0.01|43.45 ± 0.01|44.57 ± 0.03|46.92 ± 0.07|
|L2P |37.86 ± 0.02|45.77 ± 0.03|52.42 ± 0.04|53.15 ± 0.01|44.57 ± 0.03|46.62 ± 0.0|49.07 ± 0.07|
|SimpleCIL |61.21 ± 0.0|60.64 ± 0.0|59.82 ± 0.0|58.3 ± 0.0|53.0 ± 0.0|53.06 ± 0.0|53.14 ± 0.0|
|Finetune |22.27 ± 0.17|18.74 ± 9.85|12.5 ± 0.15|38.37 ± 1.36|20.89 ± 0.41|25.19 ± 0.17|29.53 ± 0.02|




#### Omnibenchmark

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050949707.png" alt="image-20250705094934648" style="zoom:67%;" />

|           | B0 Inc5      | B0 Inc10     | B0 Inc20     | B0 Inc30     | B150 Inc5    | B100 Inc10   | B100 Inc30   |
| --------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|Inflora |67.92 ± 0.58|71.29 ± 0.1|76.25 ± 0.11|77.38 ± 0.08|52.76 ± 0.08|58.82 ± 0.15|66.33 ± 0.06|
|APER |80.36 ± 0.0|80.1 ± 0.0|80.31 ± 0.0|80.66 ± 0.06|77.41 ± 0.01|77.43 ± 0.01|77.58 ± 0.01|
|MOS |85.92 ± 0.0|85.79 ± 0.0|86.01 ± 0.06|85.88 ± 0.05|83.03 ± 0.02|83.25 ± 0.0|83.5 ± 0.03|
|Ease |73.89 ± 0.03| 74.31 ± 0.04| 75.43 ± 0.01| 74.81 ± 0.03| 68.86 ± 0.03| 68.05 ± 0.04| 71.69 ± 0.02|
|LAE |53.43 ± 0.0|68.9 ± 0.29|74.58 ± 0.0|76.17 ± 0.0|59.65 ± 0.0|63.41 ± 0.0|73.53 ± 0.0|
|CODA-Prompt |69.78 ± 0.0|74.41 ± 0.0|76.9 ± 0.0|77.79 ± 0.0|63.22 ± 0.0|65.65 ± 0.0|72.32 ± 0.0|
|DualPrompt |69.27 ± 0.05|73.61 ± 0.06|75.88 ± 0.04|74.97 ± 0.03|57.43 ± 0.0|60.18 ± 0.0|71.82 ± 0.07|
|L2P |69.09 ± 0.05|70.93 ± 0.02|74.08 ± 0.11|74.14 ± 0.03|58.29 ± 0.03|60.34 ± 0.09|70.32 ± 0.09|
|SimpleCIL |80.53 ± 0.01|80.26 ± 0.01|80.07 ± 0.0|79.35 ± 0.01|75.23 ± 0.0|75.26 ± 0.0|75.43 ± 0.01|
|Finetune |48.48 ± 0.17|56.93 ± 0.07|63.06 ± 0.09|65.86 ± 0.12|55.78 ± 0.08|63.15 ± 0.02|68.29 ± 0.09|
|FOSTER-CNN |81.15 ± 0.07|81.55 ± 0.05|80.23 ± 0.02|78.98 ± 0.1|79.9 ± 0.05|80.13 ± 0.1|78.67 ± 0.11|
|FOSTER-NME |82.01 ± 0.15|83.17 ± 0.12|83.23 ± 0.01|83.32 ± 0.13|80.57 ± 0.16|81.16 ± 0.07|81.61 ± 0.06|
|DER-CNN |75.73 ± 0.0|76.73 ± 0.01|77.18 ± 0.09|77.61 ± 0.11|76.09 ± 0.1|76.7 ± 0.14|76.56 ± 0.11|
|DER-NME |77.16 ± 0.0|78.96 ± 0.04|80.36 ± 0.03|80.92 ± 0.25|77.1 ± 0.18|77.72 ± 0.07|79.15 ± 0.02|
|iCaRL-CNN |71.61 ± 0.12|73.71 ± 0.32|74.93 ± 0.22|75.05 ± 0.3|70.77 ± 0.09|72.07 ± 0.05|74.27 ± 0.17|
|iCaRL-NME |77.1 ± 0.32|79.69 ± 0.11|81.27 ± 0.02|80.99 ± 0.08|75.79 ± 0.09|77.39 ± 0.01|79.2 ± 0.1|



#### VTAB

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050950395.png" alt="image-20250705095009313" style="zoom:67%;" />


|      | B0 Inc5 | B0 Inc10 |
| ---- | ------- | -------- |
|Inflora |82.11 ± 0.36|89.7 ± 0.22|
|APER |86.96 ± 0.0|86.25 ± 0.0|
|MOS |96.08 ± 0.22|95.92 ± 0.21|
|Ease |90.79 ± 0.0| 90.56 ± 0.16|
|LAE |69.31 ± 0.0|84.94 ± 0.0|
|CODA-Prompt |80.44± 0.0|87.24 ± 0.0|
|DualPrompt |87.46 ± 0.0|89.82 ± 0.06|
|L2P |77.47 ± 0.0|81.84 ± 0.19|
|SimpleCIL |91.35 ± 0.0|90.8 ± 0.0|
|Finetune |60.69 ± 2.02|80.8 ± 1.5|

---

### Different PTMs

| PTM             | Pre-Trained Dataset              | Finetuned Dataset   | timm                                                         | Description                                                  |
| --------------- | -------------------------------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ViT-B/16-IN1K   | ImageNet21K                      | ImageNet1K          | timm.create_model("vit_base_patch16_224",pretrained=True)    | Vision Transformer trained on ImageNet21K and fine-tuned on ImageNet1K. |
| ViT-B/16-IN21K  | ImageNet21K                      | ImageNet1K          | timm.create_model("vit_base_patch16_224_in21k",pretrained=True) | Vision Transformer trained on ImageNet21K without fine-tuning. |
| ViT-L/16-IN1K   | ImageNet21K                      | ImageNet1K          |                                                              | Large Vision Transformer trained on ImageNet21K and fine-tuned on ImageNet1K. |
| ViT-B/16-DINO   | ImageNet                         | ImageNet1K          |                                                              | Self-supervised Vision Transformer trained with DINO on ImageNet. |
| ViT-B/16-SAM    | SA-1B (Segment Anything Dataset) | COCO, ADE20K        |                                                              | Vision Transformer trained on a large-scale segmentation dataset. |
| ViT-B/16-MAE    | ImageNet21K                      | ImageNet1K          |                                                              | Vision Transformer trained with Masked Autoencoder on ImageNet21K. |
| ViT-B/16-CLIP   | OpenAI CLIP Dataset              | COCO, Flickr30K     |                                                              | Vision Transformer trained on a large corpus of text-image pairs by OpenAI. |
| ResNet18/50/152 | ImageNet1K                       | CIFAR-10, CIFAR-100 |                                                              | ResNet models trained on ImageNet1K.                         |

---

## 🤗 Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
