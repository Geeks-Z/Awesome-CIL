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

## 🌟 已复现方法

| Title                                                        | Method      | Venue | Year | Type               | Code                                                    |
| ------------------------------------------------------------ | ----------- | ----- | ---- | ------------------ | ------------------------------------------------------- |
| [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725) | iCaRL       | CVPR  | 2017 | Memory             | [Official](https://github.com/srebuffi/iCaRL)           |
| [Co-Transport for Class-Incremental Learning](https://arxiv.org/abs/2107.12654) | COIL        | MM    | 2021 | Classifier Transport |                                                        |
| [DER: Dynamically Expandable Representation for Class Incremental Learning](https://arxiv.org/abs/2103.16788) | DER         | CVPR  | 2021 | Backbone Expansion | [Official](https://github.com/G-U-N/PyCIL)              |
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | DualPrompt  | ECCV  | 2022 | Prompt             | [Official](https://github.com/google-research/l2p)      |
| [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) | FOSTER      | ECCV  | 2022 | Backbone Expansion | [Official](https://github.com/G-U-N/ECCV22-FOSTER)      |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | L2P         | CVPR  | 2022 | Prompt             | [Official](https://github.com/google-research/l2p)      |
| [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](http://arxiv.org/abs/2211.13218) | CODA-Prompt | CVPR  | 2023 | Prompt             | [Official](https://github.com/GT-RIPL/CODA-Prompt)      |
| [Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality](https://arxiv.org/abs/2310.07234) | HiDe-Prompt | NeurIPS | 2023 | Prompt             | [Official](https://github.com/thu-ml/HiDe-Prompt)       |
| [A Unified Continual Learning Framework with General Parameter-Efficient Tuning](http://arxiv.org/abs/2303.10070) | LAE         | ICCV  | 2023 | PEFT Expansion     | [Official](https://github.com/gqk/LAE)                  |
| [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2403.12030) | EASE        | CVPR  | 2024 | PEFT Expansion     | [Official](https://github.com/sun-hailong/CVPR24-Ease)  |
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](http://arxiv.org/abs/2404.00228) | InfLoRA     | CVPR  | 2024 | PEFT-LoRA          | [Official](https://github.com/liangyanshuo/InfLoRA)     |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) | APER / ADAM | IJCV  | 2024 | PEFT Adapter       | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) | SimpleCIL   | IJCV  | 2024 | PTM Prototype      | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
| [BiLoRA: Almost-Orthogonal Parameter Spaces for Continual Learning](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_BiLoRA_Almost-Orthogonal_Parameter_Spaces_for_Continual_Learning_CVPR_2025_paper.html) | BiLoRA      | CVPR  | 2025 | PEFT-LoRA          | [Official](https://github.com/yifeiacc/BiLoRA)          |
| [CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2025/html/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.html) | CL-LoRA     | CVPR  | 2025 | PEFT-LoRA          | [Official](https://github.com/JiangpengHe/CL-LoRA)      |
| [MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2412.09441) | MOS         | AAAI  | 2025 | PEFT Adapter       | [Official](https://github.com/sun-hailong/AAAI25-MOS)   |
| [SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning](https://openreview.net/forum?id=5U1rlpX68A) | SD-LoRA     | ICLR  | 2025 | PEFT-LoRA          | [Official](https://github.com/WuYichen-97/SD-Lora-CL)   |

---

## ☄️ 使用方法

### 🕹️ 克隆

克隆本 GitHub 仓库：

```
git clone https://github.com/Geeks-Z/Awesome-CIL
cd Awesome-CIL
```

### 🗂️ 依赖

1. [torch 2.0.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.15.2](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)
4. [tqdm](https://github.com/tqdm/tqdm)
5. [numpy](https://github.com/numpy/numpy)
6. [scipy](https://github.com/scipy/scipy)
7. [easydict](https://github.com/makinacorpus/easydict)
8. [open-clip 2.17.1](https://github.com/mlfoundations/open_clip/releases/tag/v2.17.1)

### 🔑 运行实验

1. 在 `configs/<method>/` 中选择或复制一个 json 配置文件，设置数据集路径、GPU、seed、初始类别数、增量类别数和方法超参数。

2. 运行：

   ```bash
   python main.py --config configs/ease/ease_cifar_B0_Inc10.json
   ```

3. 也可以直接使用 `scripts/` 中的批量脚本：

   ```bash
   bash scripts/train_ease.sh
   ```

4. 当前配置目录与 `model_name`

   | Config directory | model_name |
   | ---------------- | ---------- |
   | `configs/aper` | `adam_adapter` |
   | `configs/bilora` | `bilora` |
   | `configs/cllora` | `cllora` |
   | `configs/coda_prompt` | `coda_prompt` |
   | `configs/coil` | `coil` |
   | `configs/der` | `der` |
   | `configs/dualprompt` | `dualprompt` |
   | `configs/ease` | `ease` |
   | `configs/finetune` | `finetune` |
   | `configs/foster` | `foster` |
   | `configs/hidep` | `hidep` |
   | `configs/icarl` | `icarl` |
   | `configs/inflora` | `inflora` |
   | `configs/l2p` | `l2p` |
   | `configs/lae` | `lae` |
   | `configs/mos` | `mos` |
   | `configs/sdlora` | `sdlora` |
   | `configs/simplecil` | `simplecil` |

5. 常用 `hyper-parameters`

   - **model_name**: 模型名称，必须与 `utils/factory.py` 中注册的名称一致。
   - **dataset**: 数据集名称，例如 `cifar224`、`cub`、`imageneta`、`imagenetr`、`omnibenchmark`、`vtab`。
   - **data_path**: 数据集本地路径。
   - **device**: 使用的 GPU ID 列表，例如 `["0"]` 或 `["0", "1"]`。
   - **seed**: 类别顺序随机种子列表，例如 `[1993, 1994, 42]`。
   - **init_cls**: 初始增量阶段的类别数量。`0` 表示从第一个增量任务开始按 `increment` 划分。
   - **increment**: 后续每个增量阶段的类别数量。
   - **backbone_type**: 骨干网络名称。PTM-CIL 公平比较默认使用 ViT-B/16-IN21K，对应配置通常包含 `vit_base_patch16_224_in21k` 或 `pretrained_vit_b16_224_in21k`。
   - **fixed_memory**: 布尔参数。设为 true 时每类保留固定数量的 memory；设为 false 时每类 memory 数量动态分配。

- **memory_size**: 增量学习过程中的 exemplar 总数。如果 `fixed_memory` 为 false，假设当前阶段有 $K$ 个类别，模型将为每类保留 $\left[\frac{{memory-size}}{K}\right]$ 个 exemplar。**ZS-CLIP、SimpleCIL、ADAM、EASE、TUNA、CLG-CBM、MG_CLIP、ENGINE 和 BOFA 不需要 exemplar**，因此不会使用 exemplar 相关参数。
  - **memory_per_class**: 如果 `fixed memory` 为 true，模型将为每个类别保留固定数量的 `memory_per_class` exemplars。

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

## 📊 Reproduced Results

#### CIFAR-100

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050946577.png" alt="image-20250705094642483" style="zoom:67%;" />

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B50 Inc5    | B100 Inc10   |
| --------- | ----------- | ------------ | ----------- | ------------ | ------------ |
|Inflora |89.55|92.01|93.22|69.84|75.12|
|APER |90.22|92.12|92.51|91.94|91.95|
|MOS |93.43|94.76|94.8|94.18|94.1|
|Ease |91.64| 92.56| 93.09| 89.26| 90.41|
|LAE |81.94|90.54|91.33|88.12|90.4|
|CODA-Prompt |87.34|91.31|92.72|82.79|88.43|
|DualPrompt |88.44|90.32|91.37|80.83|87.42|
|L2P |87.57|89.78|90.76|79.46|87.62|
|SimpleCIL |87.57|87.13|86.11|83.79|83.89|
|Finetune |72.32|76.94|80.87|80.29|82.63|
|FOSTER-CNN |94.02|93.9|93.61|92.04|92.26|
|FOSTER-NME |94.19|94.09|93.62|92.19|92.26|
|DER-CNN |88.67|88.6|88.68|86.54|86.86|
|DER-NME |90.95|91.05|91.27|88.92|89.38|
|iCaRL-CNN |84.6|85.87|86.91|81.67|83.54|
|iCaRL-NME |89.52|90.3|90.64|87.55|88.98|

#### CUB-200

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050947215.png" alt="image-20250705094735156" style="zoom:67%;" />

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B0 Inc40 | B100 Inc5    | B100 Inc10   | B100 Inc20 |
| --------- | ----------- | ------------ | ----------- | -------- | ------------ | ------------ | ---------- |
|Inflora |69.65|74.61|83.54|86.31|39.91|55.33|67.86|
|MOS |93.78|93.53|93.24|92.51|91.53|91.39|91.46|
|APER |92.52|92.27|92.06|91.82|89.64|89.66|89.73|
|Ease |92.3| 91.33| 90.19| 90.17| 86.07| 87.47| 85.28|
|LAE |61.07|73.28|81.15|83.67|68.29|70.81|74.66|
|CODA-Prompt |71.59|83.25|84.65|85.69|69.63|77.89|82.39|
|DualPrompt |80.82|83.07|84.81|85.44|71.06|73.7|76.75|
|L2P |71.63|79.27|81.8|83.78|70.67|74.62|79.41|
|SimpleCIL |92.4|92.23|91.85|91.29|88.98|88.99|89.05|
|Finetune |31.13|64.95|70.35|78.54|46.33|63.3|73.32|
|FOSTER-CNN |87.31|85.82|81.12|78.48|86.92|85.85|82.95|
|FOSTER-NME |91.28|91.69|91.15|90.72|88.97|89.06|89.03|
|DER-CNN |89.83|89.58|89.11|89.2|87.05|87.07|86.82|
|DER-NME |90.59|90.67|89.93|89.78|87.83|88.28|88.36|
|iCaRL-CNN |87.83|88.12|88.33|87.51|84.47|85.44|85.84|
|iCaRL-NME |90.12|90.19|90.1|88.99|87.17|87.6|88.12|


#### ImageNet-R

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050948418.png" alt="image-20250705094828355" style="zoom:67%;" />

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B0 Inc40 | B100 Inc5    | B100 Inc10   | B100 Inc20 |
| --------- | ----------- | ------------ | ----------- | -------- | ------------ | ------------ | ---------- |
|Inflora |72.95|79.91|83.06|83.1|51.12|58.63|65.64|
|APER |71.19|74.37|75.17|75.28|73.99|73.96|74.03|
|MOS |77.13|81.24|82.54|82.72|81.0|81.07|81.05|
|Ease |67.81| 78.94| 80.95| 81.53| 71.4| 76.59| 77.98|
|LAE |58.91|70.41|75.57|76.57|60.92|67.13|70.7|
|CODA-Prompt |59.61|72.48|78.17|80.46|65.85|71.61|75.17|
|DualPrompt |63.08|69.9|71.03|73.58|55.17|59.69|64.9|
|L2P |70.48|73.76|76.23|76.76|59.58|64.89|69.72|
|SimpleCIL |62.42|61.85|61.14|59.72|56.71|56.7|56.83|
|Finetune |56.56|63.8|69.03|72.19|61.24|67.13|70.67|


#### ImageNet-A

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050949218.png" alt="image-20250705094905159" style="zoom:67%;" />

|           | B0 Inc5     | B0 Inc10     | B0 Inc20    | B0 Inc40 | B100 Inc5    | B100 Inc10   | B100 Inc20 |
| --------- | ----------- | ------------ | ----------- | -------- | ------------ | ------------ | ---------- |
|Inflora |40.51|51.19|59.17|63.42|33.75|43.08|51.09|
|APER |61.24|60.69|60.02|60.0|61.36|61.4|61.48|
|MOS |64.26|64.29|62.96|65.42|64.57|64.89|64.9|
|Ease |60.18| 57.69| 62.6| 67.73| 62.14| 62.35| 63.15|
|LAE |26.87|38.31|48.61|54.76|45.25|48.73|51.35|
|CODA-Prompt |59.61|72.48|78.17|80.46|65.85|71.61|75.17|
|DualPrompt |43.62|48.48|52.51|54.02|43.45|44.57|46.92|
|L2P |37.86|45.77|52.42|53.15|44.57|46.62|49.07|
|SimpleCIL |61.21|60.64|59.82|58.3|53.0|53.06|53.14|
|Finetune |22.27|18.74|12.5|38.37|20.89|25.19|29.53|




#### Omnibenchmark

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050949707.png" alt="image-20250705094934648" style="zoom:67%;" />

|           | B0 Inc5      | B0 Inc10     | B0 Inc20     | B0 Inc30     | B150 Inc5    | B100 Inc10   | B100 Inc30   |
| --------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|Inflora |67.92|71.29|76.25|77.38|52.76|58.82|66.33|
|APER |80.36|80.1|80.31|80.66|77.41|77.43|77.58|
|MOS |85.92|85.79|86.01|85.88|83.03|83.25|83.5|
|Ease |73.89| 74.31| 75.43| 74.81| 68.86| 68.05| 71.69|
|LAE |53.43|68.9|74.58|76.17|59.65|63.41|73.53|
|CODA-Prompt |69.78|74.41|76.9|77.79|63.22|65.65|72.32|
|DualPrompt |69.27|73.61|75.88|74.97|57.43|60.18|71.82|
|L2P |69.09|70.93|74.08|74.14|58.29|60.34|70.32|
|SimpleCIL |80.53|80.26|80.07|79.35|75.23|75.26|75.43|
|Finetune |48.48|56.93|63.06|65.86|55.78|63.15|68.29|
|FOSTER-CNN |81.15|81.55|80.23|78.98|79.9|80.13|78.67|
|FOSTER-NME |82.01|83.17|83.23|83.32|80.57|81.16|81.61|
|DER-CNN |75.73|76.73|77.18|77.61|76.09|76.7|76.56|
|DER-NME |77.16|78.96|80.36|80.92|77.1|77.72|79.15|
|iCaRL-CNN |71.61|73.71|74.93|75.05|70.77|72.07|74.27|
|iCaRL-NME |77.1|79.69|81.27|80.99|75.79|77.39|79.2|



#### VTAB

<img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/202507050950395.png" alt="image-20250705095009313" style="zoom:67%;" />


|      | B0 Inc5 | B0 Inc10 |
| ---- | ------- | -------- |
|Inflora |82.11|89.7|
|APER |86.96|86.25|
|MOS |96.08|95.92|
|Ease |90.79| 90.56|
|LAE |69.31|84.94|
|CODA-Prompt |80.44|87.24|
|DualPrompt |87.46|89.82|
|L2P |77.47|81.84|
|SimpleCIL |91.35|90.8|
|Finetune |60.69|80.8|

---

### 

|      |      |      |      |      |
| ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |
|      |      |      |      |      |

---

## 🤗 Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
