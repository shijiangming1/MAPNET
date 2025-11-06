<div align="center">

# 🧠 MAPNet: Multi-Schema Proximity Network for Composed Image Retrieval

<a href="https://openaccess.thecvf.com/content/ICCV2025/html/Shi_Multi-Schema_Proximity_Network_for_Composed_Image_Retrieval_ICCV_2025_paper.html" target="_blank">
  <img src="https://img.shields.io/badge/ICCV-2025-blue.svg" alt="ICCV 2025">
</a>
<a href="https://pytorch.org/" target="_blank">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch">
</a>
<a href="https://github.com/your-username/MAPNet/blob/main/LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
</a>

</div>

---

## 👩‍💻 Authors & Affiliations

**Jiangming Shi**<sup>1,2</sup> · [📧](mailto:jiangming.shi@outlook.com)  
**Xiangbo Yin**<sup>1</sup> · [📧](mailto:yinxb@163.com)  
**Yeyun Chen**<sup>1,2</sup> · [📧](mailto:yeyunchen2022@stu.xmu.edu.cn)  
**Yachao Zhang**<sup>1</sup> · [📧](mailto:yachaozhang@xmu.edu.cn)  
**Zhizhong Zhang**<sup>3</sup> · [📧](mailto:zzzhang@cs.ecnu.edu.cn)  
**Yuan Xie**<sup>2,3</sup> · [📧](mailto:yxie@cs.ecnu.edu.cn)  
**Yanyun Qu**<sup>1*</sup> · [📧](mailto:yyqu@xmu.edu.cn) *(Corresponding Author)*  

<sup>1</sup> *Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, School of Informatics, Xiamen University*  
<sup>2</sup> *Shanghai Innovation Institute*  
<sup>3</sup> *East China Normal University*  

---

## 📖 Overview

**Composed Image Retrieval (CIR)** enables retrieving target images using a **composed query** (reference image + textual description), which better expresses user intent than single-modal queries.  

However, existing methods face two critical limitations:  
1. **Insufficient multi-schema interaction** — Fail to capture complex object-attribute relationships due to lack of fine-grained supervision.  
2. **Noisy negative interference** — Mislabeled matching pairs (noisy negatives) degrade training accuracy.  

We propose **MAPNet**, a novel architecture that overcomes these issues via two key innovations:  
- **Multi-Schema Interaction (MSI)** — Utilizes BLIP-2 Q-Former and optimal transport to dynamically model object–attribute relationships.  
- **Relaxed Proximity Loss (RPLoss)** — Reduces the influence of noisy negatives through similarity-based denoising and adaptive reweighting.

---

## 📊 Key Results

### CIRR Dataset (Test Set)
| Method | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Avg. |
|:-------|:---------:|:---------:|:----------:|:----------:|:----:|
| SPRC (SOTA) | 51.96% | 82.12% | 89.74% | 97.69% | 81.39 |
| **MAPNet (Ours)** | **54.65%** | **84.93%** | **91.44%** | **98.25%** | **83.04** |

### FashionIQ Dataset (Validation Set)
| Method | Dress (R@10/R@50) | Shirt (R@10/R@50) | Toptee (R@10/R@50) | Avg. (R@10/R@50) |
|:-------|:-----------------:|:------------------:|:-------------------:|:----------------:|
| SPRC (SOTA) | 47.80% / 72.70% | 55.84% / 74.37% | 58.89% / 78.99% | 54.17% / 75.35% |
| **MAPNet (Ours)** | **51.17% / 74.12%** | **56.37% / 75.17%** | **59.56% / 79.30%** | **55.70% / 76.20%** |

### LaSCo Dataset (Validation Set)
| Method | Recall@5 | Recall@10 | Recall@50 | Recall@500 |
|:-------|:---------:|:----------:|:-----------:|:------------:|
| SPRC (SOTA) | 22.21% | 30.43% | 55.10% | 88.17% |
| **MAPNet (Ours)** | **24.62%** | **32.27%** | **58.18%** | **90.09%** |

> 📈 *Full results and ablation studies are available in the paper.*

---

## 🛠️ Installation

### Prerequisites
- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- CUDA ≥ 11.7 *(Recommended: NVIDIA RTX H100 80GB)*  

### Step 1: Clone the Repo
```bash
git clone https://github.com/your-username/MAPNet.git
cd MAPNet
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
# Includes: torch, torchvision, transformers, pillow, scikit-learn, numpy, tqdm
```

### Step 3: Prepare Datasets
Download the three CIR benchmarks and organize the data directory as:
```plaintext
data/
├── CIRR/          # https://github.com/Cuberick-Orion/CIRR
│   ├── train/
│   ├── val/
│   └── test/
├── FashionIQ/     # https://github.com/XiaoxiaoGuo/Fashion-IQ
│   ├── dress/
│   ├── shirt/
│   └── toptee/
└── LaSCo/         # https://github.com/matanlevy/LaSCo
    ├── images/
    └── annotations/
```

---

## 🚀 Usage

### Training
```bash
# Train on CIRR
python train.py \
  --dataset cirr \
  --data_root ./data/CIRR \
  --lr 1e-5 \
  --batch_size 32 \
  --epochs 50 \
  --ckpt_save_dir ./weights

# Train on FashionIQ
python train.py \
  --dataset fashioniq \
  --data_root ./data/FashionIQ \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 50 \
  --ckpt_save_dir ./weights

# Train on LaSCo
python train.py \
  --dataset lasco \
  --data_root ./data/LaSCo \
  --lr 1e-5 \
  --batch_size 16 \
  --epochs 50 \
  --ckpt_save_dir ./weights
```

### Evaluation
```bash
# Evaluate on CIRR
python eval.py \
  --dataset cirr \
  --data_root ./data/CIRR \
  --ckpt_path ./weights/mapnet_cirr.pth \
  --metrics recall@1,recall@5,recall@10,recall@50

# Evaluate on FashionIQ
python eval.py \
  --dataset fashioniq \
  --data_root ./data/FashionIQ \
  --ckpt_path ./weights/mapnet_fashioniq.pth \
  --metrics recall@10,recall@50
```

---

## 🧩 Model Details

| Component | Configuration |
|------------|----------------|
| **Backbone** | Frozen BLIP-2 (ViT-L/14 image encoder, BERT-base text encoder) |
| **MSI Module** | 32 learnable queries + Sinkhorn-Knopp optimal transport (τ=0.1) |
| **RPLoss** | Confidence threshold γ=0.5, kernel bandwidth σ=0.6 |
| **Alignment Loss** | Squared Maximum Mean Discrepancy (MMD²) |

---

## 🙏 Acknowledgments

This work is supported by:
- National Natural Science Foundation of China (No. 62176224, 62176092, 62222602, 62306165)  
- Science and Technology on Sonar Laboratory (No. 2024-JCJQ-LB-32/07)  
- Railway Sciences (No. 2023Y1357)

---

## 📝 Citation

If you use this code or our results, please cite our paper:

```bibtex
@inproceedings{shi2025mapnet,
  title={Multi-Schema Proximity Network for Composed Image Retrieval},
  author={Shi, Jiangming and Yin, Xiangbo and Chen, Yeyun and Zhang, Yachao and Zhang, Zhizhong and Xie, Yuan and Qu, Yanyun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

---

## 📞 Contact

For questions or collaborations, please reach out to:

📧 **Jiangming Shi** — jiangming.shi@outlook.com  

---

<div align="center">

⭐ If you find this project helpful, please give it a star on [GitHub](https://github.com/your-username/MAPNet)!  
💬 *We welcome feedback, collaborations, and research discussions.*

</div>
