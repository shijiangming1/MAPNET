MAPNet: Multi-Schema Proximity Network for Composed Image Retrieval
<div align="center"><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Shi_Multi-Schema_Proximity_Network_for_Composed_Image_Retrieval_ICCV_2025_paper.html" target="_blank"><img src="https://img.shields.io/badge/ICCV-2025-blue.svg" alt="ICCV 2025"></a><a href="https://pytorch.org/" target="_blank"><img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch"></a><a href="https://github.com/your-username/MAPNet/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></div>
Official implementation of the ICCV 2025 paper "Multi-Schema Proximity Network for Composed Image Retrieval" .
📖 Overview
Composed Image Retrieval (CIR) enables retrieving target images using a composed query (reference image + textual description), which better expresses user intent than single-modal queries. However, existing methods face two critical limitations:
Insufficient multi-schema interaction: Fail to capture complex object-attribute relationships due to lack of fine-grained supervision.
Noisy negative interference: Mislabeled matching pairs (noisy negatives) degrade training accuracy.
We propose MAPNet to address these issues with two core components:
Multi-Schema Interaction (MSI): Uses BLIP-2 Q-Former and optimal transport to model dynamic object-attribute interactions .
Relaxed Proximity Loss (RPLoss): Reduces noisy negative impact via similarity-based denoising and reweighting .
📊 Key Results
MAPNet outperforms state-of-the-art methods on three standard CIR benchmarks:
CIRR Dataset (Test Set)
Method	Recall@1	Recall@5	Recall@10	Recall@50	Avg.
SPRC (SOTA)	51.96%	82.12%	89.74%	97.69%	81.39
MAPNet	54.65%	84.93%	91.44%	98.25%	83.04
FashionIQ Dataset (Validation Set)
Method	Dress (R@10/R@50)	Shirt (R@10/R@50)	Toptee (R@10/R@50)	Avg. (R@10/R@50)
SPRC (SOTA)	47.80%/72.70%	55.84%/74.37%	58.89%/78.99%	54.17%/75.35%
MAPNet	51.17%/74.12%	56.37%/75.17%	59.56%/79.30%	55.70%/76.20%
LaSCo Dataset (Validation Set)
Method	Recall@5	Recall@10	Recall@50	Recall@500
SPRC (SOTA)	22.21%	30.43%	55.10%	88.17%
MAPNet	24.62%	32.27%	58.18%	90.09%
Full results and ablation studies are available in the paper .
🛠️ Installation
Prerequisites
Python 3.8+
PyTorch 2.0+
CUDA 11.7+ (recommended: NVIDIA RTX H100 80GB)
Step 1: Clone the Repo
bash
git clone https://github.com/your-username/MAPNet.git
cd MAPNet
Step 2: Install Dependencies
bash
pip install -r requirements.txt
# Requirements include: torch, torchvision, transformers, pillow, scikit-learn, numpy, tqdm
Step 3: Prepare Datasets
Download three CIR benchmarks and organize the directory as follows:
plaintext
data/
├── CIRR/          # From https://github.com/Cuberick-Orion/CIRR
│   ├── train/
│   ├── val/
│   └── test/
├── FashionIQ/     # From https://github.com/XiaoxiaoGuo/Fashion-IQ
│   ├── dress/
│   ├── shirt/
│   └── toptee/
└── LaSCo/         # From https://github.com/matanlevy/LaSCo
    ├── images/
    └── annotations/
🚀 Usage
Training
Run training on a single GPU (adjust hyperparameters as needed):
bash
# Train on CIRR (lr=1e-5, batch_size=32)
python train.py \
  --dataset cirr \
  --data_root ./data/CIRR \
  --lr 1e-5 \
  --batch_size 32 \
  --epochs 50 \
  --ckpt_save_dir ./weights

# Train on FashionIQ (lr=2e-5, batch_size=32)
python train.py \
  --dataset fashioniq \
  --data_root ./data/FashionIQ \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 50 \
  --ckpt_save_dir ./weights

# Train on LaSCo (lr=1e-5, batch_size=16)
python train.py \
  --dataset lasco \
  --data_root ./data/LaSCo \
  --lr 1e-5 \
  --batch_size 16 \
  --epochs 50 \
  --ckpt_save_dir ./weights
Evaluation
Evaluate pre-trained models with Recall@K metrics:
bash
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
🧩 Model Details
Component	Configuration
Backbone	Frozen BLIP-2 (image encoder: ViT-L/14, text encoder: BERT-base) 
MSI Module	32 learnable queries + Sinkhorn-Knopp optimal transport (τ=0.1) 
RPLoss	Confidence threshold γ=0.5, kernel bandwidth σ=0.6 
Alignment Loss	Squared Maximum Mean Discrepancy (MMD²) 
🙏 Acknowledgments
This work is supported by:
National Natural Science Foundation of China (No. 62176224, 62176092, 62222602, 62306165) 
Science and Technology on Sonar Laboratory (No. 2024-JCJQ-LB-32/07) 
Railway Sciences (No. 2023Y1357) 
📝 Citation
If you use this code or our results, please cite our paper:
bibtex
@inproceedings{shi2025mapnet,
  title={Multi-Schema Proximity Network for Composed Image Retrieval},
  author={Shi, Jiangming and Yin, Xiangbo and Chen, Yeyun and Zhang, Yachao and Zhang, Zhizhong and Xie, Yuan and Qu, Yanyun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
📞 Contact
For questions or issues, please reach out to:
Jiangming Shi: jiangming.shi@outlook.com
Yanyun Qu (corresponding author): yyqu@xmu.edu.cn
