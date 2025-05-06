# Impact of Dataset Size and Distillation Techniques on Image Captioning Performance

This project investigates the impact of dataset size and dataset distillation strategies on image captioning models. We experiment with three pre-trained architectures—ResNet-50 + LSTM, CLIP + GPT-2, and GIT—to evaluate how well they perform with different data sizes and under two distillation approaches: **random sampling** and **gradient-based selection**.

## Motivation

Training deep learning models for image captioning typically requires massive datasets and compute resources. But do we always need that much data? This project explores if **smarter data selection** and **stronger model architectures** can help us train effective captioning models with far less data and cost.

## Dataset

We use a 5K subset of the [MSCOCO 2017 validation set](https://cocodataset.org/#download), where each image is annotated with 5 human-written captions. Images are resized to 224×224 pixels, and captions are tokenized and padded to a max length of 32 tokens.

## Experiments

We conducted three core experiments:

### 1. Comparing Distillation Methods
We trained models using full data but varied the distillation method:
- **Random Sampling**: Selects a random subset of images.
- **Gradient-Based Distillation**: Prioritizes samples with high gradient norms.

### 2. Varying Dataset Sizes
We scaled dataset sizes (25%, 50%, 75%, 100%) to test performance vs. compute trade-offs using both distillation methods.

### 3. Comparing Model Architectures
We trained and evaluated:
- `ResNet-50 + LSTM` (Baseline)
- `CLIP + GPT-2` (Linear projection + language decoder)
- `GIT` (Pretrained Vision-Language Transformer)

## Evaluation Metrics

We assess captioning quality using:
- **BLEU-1 to BLEU-4**
- **CIDEr**
- **Training Time (iterations/sec)**

## Key Findings

- **Model matters more than dataset size**: GIT outperforms GPT, which outperforms ResNet-50—even on smaller datasets.
- **Gradient-based distillation slightly improves results**, especially at lower data percentages.
- **75% dataset gives near-maximum performance**, suggesting potential data savings.
- **Automatic metrics can be misleading**: Models with high BLEU/CIDEr still generate poor real-world captions.

## Tech Stack

- Python, PyTorch, TensorFlow
- HuggingFace Transformers (`GPT-2`, `GIT`)
- CLIP (`ViT-B/32`)
- MSCOCO Dataset
- Jupyter Notebooks (Jetstream VM with NVIDIA A100 GPU)

## Setup Instructions

```bash
git clone https://github.com/your-username/image-captioning-distillation.git
cd image-captioning-distillation
pip install -r requirements.txt
