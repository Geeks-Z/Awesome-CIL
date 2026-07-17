# Awesome-CIL

<p align="center">
  <img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/logo.png" alt="Awesome-CIL" width="240" />
</p>

<p align="center">
  <a href="https://visitor-badge.laobi.icu/badge?page_id=hongwei-zhao.Awesome-CIL"><img src="https://visitor-badge.laobi.icu/badge?page_id=hongwei-zhao.Awesome-CIL&left_color=green&right_color=red" alt="Visitors" /></a>
  <a href="https://github.com/hongwei-zhao/Awesome-CIL/commits/develop"><img src="https://img.shields.io/github/last-commit/hongwei-zhao/Awesome-CIL/develop?label=last%20commit" alt="Last commit on develop" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/hongwei-zhao/Awesome-CIL?label=license" alt="MIT License" /></a>
</p>

Awesome-CIL collects resources, reference implementations, and reproducible results for class-incremental learning (CIL), also known as continual, incremental, or lifelong learning.

For paper reading notes, see [Paper Reading Notes](https://www.zhihu.com/people/icode1024/columns).

## 🎉Surveys

| Title | Venue | Year | Code |
| --- | --- | --- | --- |
| [Class-Incremental Learning: A Survey](http://arxiv.org/abs/2302.03648) | TPAMI | 2024 | [Official](https://github.com/zhoudw-zdw/CIL_Surve) |
| [Continual Learning with Pre-Trained Models: A Survey](http://arxiv.org/abs/2401.16386) | IJCAI | 2024 | [Official](https://github.com/sun-hailong/LAMDA-PILOT) |
| [PyCIL: A Python Toolbox for Class-Incremental Learning](https://arxiv.org/abs/2112.12533) | — | — | [Official](https://github.com/G-U-N/PyCIL) |

## 🚀 Reproduced Methods

| Title | Method | Venue | Year | Category | Reference implementation |
| --- | --- | --- | --- | --- | --- |
| [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725) | iCaRL | CVPR | 2017 | Memory | [Official](https://github.com/srebuffi/iCaRL) |
| [Co-Transport for Class-Incremental Learning](https://arxiv.org/abs/2107.12654) | COIL | MM | 2021 | Classifier transport | — |
| [DER: Dynamically Expandable Representation for Class Incremental Learning](https://arxiv.org/abs/2103.16788) | DER | CVPR | 2021 | Backbone expansion | [Official](https://github.com/G-U-N/PyCIL) |
| [DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning](https://arxiv.org/abs/2204.04799) | DualPrompt | ECCV | 2022 | Prompt | [Official](https://github.com/google-research/l2p) |
| [FOSTER: Feature Boosting and Compression for Class-Incremental Learning](https://arxiv.org/abs/2204.04662) | FOSTER | ECCV | 2022 | Backbone expansion | [Official](https://github.com/G-U-N/ECCV22-FOSTER) |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654) | L2P | CVPR | 2022 | Prompt | [Official](https://github.com/google-research/l2p) |
| [CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning](https://arxiv.org/abs/2211.13218) | CODA-Prompt | CVPR | 2023 | Prompt | [Official](https://github.com/GT-RIPL/CODA-Prompt) |
| [Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality](https://arxiv.org/abs/2310.07234) | HiDe-Prompt | NeurIPS | 2023 | Prompt | [Official](https://github.com/thu-ml/HiDe-Prompt) |
| [A Unified Continual Learning Framework with General Parameter-Efficient Tuning](http://arxiv.org/abs/2303.10070) | LAE | ICCV | 2023 | PEFT expansion | [Official](https://github.com/gqk/LAE) |
| [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2403.12030) | EASE | CVPR | 2024 | PEFT expansion | [Official](https://github.com/sun-hailong/CVPR24-Ease) |
| [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](http://arxiv.org/abs/2404.00228) | InfLoRA | CVPR | 2024 | PEFT / LoRA | [Official](https://github.com/liangyanshuo/InfLoRA) |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) | APER / ADAM | IJCV | 2024 | PEFT / adapter | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
| [Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](https://arxiv.org/pdf/2303.07338) | SimpleCIL | IJCV | 2024 | PTM prototype | [Official](https://github.com/zhoudw-zdw/RevisitingCIL) |
| [BiLoRA: Almost-Orthogonal Parameter Spaces for Continual Learning](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_BiLoRA_Almost-Orthogonal_Parameter_Spaces_for_Continual_Learning_CVPR_2025_paper.html) | BiLoRA | CVPR | 2025 | PEFT / LoRA | [Official](https://github.com/yifeiacc/BiLoRA) |
| [CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2025/html/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.html) | CL-LoRA | CVPR | 2025 | PEFT / LoRA | [Official](https://github.com/JiangpengHe/CL-LoRA) |
| [MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning](http://arxiv.org/abs/2412.09441) | MOS | AAAI | 2025 | PEFT / adapter | [Official](https://github.com/sun-hailong/AAAI25-MOS) |
| [SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning](https://openreview.net/forum?id=5U1rlpX68A) | SD-LoRA | ICLR | 2025 | PEFT / LoRA | [Official](https://github.com/WuYichen-97/SD-Lora-CL) |

## 🌟 Getting Started

### 🕹️ Clone

```bash
git clone https://github.com/hongwei-zhao/Awesome-CIL.git
cd Awesome-CIL
```

### 🗂️ Dependencies

- `torch==2.0.1+cu118`
- `torchvision==0.15.2+cu118`
- `timm==0.6.7`
- `numpy==1.26.3`
- `scipy==1.12.0`
- `scikit-learn==1.4.2`
- `Pillow==10.2.0`
- `PyYAML==6.0.1`
- `tqdm==4.66.2`

### 🔑 Run an Experiment

Select a JSON file in `configs/<method>/`, set the dataset path, GPU IDs, class order seed, initial classes, increment, and method-specific hyperparameters, then run:

```bash
export CIL_DATA_ROOT=/path/to/Dataset
python main.py --config configs/ease/ease_cifar_B0_Inc10.json
```

Batch launchers are available under `scripts/`. For example:

```bash
bash scripts/train_ease.sh
```

The configuration directory and registered `model_name` must agree:

| Configuration directory | `model_name` |
| --- | --- |
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

Common configuration fields:

- `model_name`: Must match a model registered in `utils/factory.py`.
- `dataset`: One of `cifar224`, `cub`, `imageneta`, `imagenetr`, `omnibenchmark`, or `vtab`.
- `data_path`: Local dataset root. `CIL_DATA_ROOT` provides the default dataset root used by the data loader.
- `device`: GPU ID list, for example `["0"]` or `["0", "1"]`.
- `seed`: Class-order seeds, for example `[1993, 1994, 42]`.
- `init_cls` and `increment`: Initial and subsequent class counts. `init_cls=0` starts directly with the incremental split.
- `backbone_type`: The PTM-CIL comparisons typically use ViT-B/16-IN21K (`vit_base_patch16_224_in21k` or `pretrained_vit_b16_224_in21k`).
- `fixed_memory`, `memory_size`, and `memory_per_class`: Exemplar-memory controls for methods that use rehearsal.

## 📚 Datasets

| Dataset | Training examples | Test examples | Classes |
| --- | ---: | ---: | ---: |
| CIFAR-100 | 50,000 | 10,000 | 100 |
| CUB-200-2011 | 9,430 | 2,358 | 200 |
| ImageNet-R | 24,000 | 6,000 | 200 |
| ImageNet-A | 5,981 | 1,519 | 200 |
| ObjectNet | 26,509 | 6,628 | 200 |
| OmniBenchmark | 89,697 | 5,983 | 300 |
| VTAB | 1,796 | 8,619 | 50 |

### ☄️ Pre-trained Models

- **ViT-B/16-IN1K** — Pre-trained on ImageNet-21K and fine-tuned on ImageNet-1K; create it with `timm.create_model("vit_base_patch16_224", pretrained=True)`.
- **ViT-B/16-IN21K** — Pre-trained on ImageNet-21K without ImageNet-1K fine-tuning; create it with `timm.create_model("vit_base_patch16_224_in21k", pretrained=True)`.
- **ViT-L/16-IN1K** — Large Vision Transformer pre-trained on ImageNet-21K and fine-tuned on ImageNet-1K.
- **ViT-B/16-DINO** — Self-supervised Vision Transformer trained with DINO on ImageNet.
- **ViT-B/16-SAM** — Vision Transformer trained on SA-1B and fine-tuned for segmentation tasks such as COCO and ADE20K.
- **ViT-B/16-MAE** — Vision Transformer trained with masked autoencoding on ImageNet-21K.
- **ViT-B/16-CLIP** — Vision Transformer trained with CLIP on large-scale image-text data.
- **ResNet-18 / ResNet-50 / ResNet-152** — ImageNet-1K pre-trained convolutional baselines.

## 📊 Reproduced Results

The [result workbook](CIL_Results.xlsx) is the single source of truth for reproduced metrics. It contains the following result sheets:

- **`seed1993`** — CIFAR-100, CUB-200, ImageNet-R, OmniBenchmark, and VTAB with class-order seed 1993.
- **`seed1994`** — The same benchmark suite with class-order seed 1994.
- **`seed42`** — The same benchmark suite with class-order seed 42.
- **`ImageNet-R` (INR)** — ImageNet-R task-count ablations for 5, 10, 20, and 40 tasks.

The [result-log mapping](result_log_mapping.csv) records the workbook cells, expected log names, parsed metrics, and verification status.

## License

This repository is distributed under the existing [MIT License](LICENSE). The original copyright notice is retained. Pre-trained weights, datasets, and incorporated third-party components remain subject to their respective licenses.

## 🤗 Acknowledgments

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [Awesome-Incremental-Learning](https://github.com/xialeiliu/Awesome-Incremental-Learning)
