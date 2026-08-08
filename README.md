![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green)

# EdgeViT: Latency-Optimized Vision Transformer for Edge AI

An end-to-end deep learning pipeline that progressively develops image classification models on CIFAR-10, culminating in a Vision Transformer that is distilled, quantized, and benchmarked for edge inference.

The project moves through representation learning, transformer architectures, knowledge distillation, model compression, and deployment with ONNX Runtime and INT8 quantization — not just a single trained model, but a full ML systems workflow.

```
MLP → CNN → Transfer Learning (ResNet18 / VGG16) → LSTM/GRU + Attention
   → Autoencoder → DCGAN → Vision Transformer → Knowledge Distillation
   → ONNX Export → INT8 Quantization → Latency Benchmark → Gradio Deployment
```

## Why this pipeline is structured this way

Each stage's output feeds the next, so the project reads as one continuous flow rather than isolated exercises:

| Stage | Produces | Consumed by |
|---|---|---|
| CNN baseline | Trained CNN backbone | Feature extraction (Stage 2) |
| Feature extraction + temporal modeling | LSTM hidden states | Autoencoder (Stage 3) |
| Autoencoder + DCGAN | 5K synthetic CIFAR-10 images | ViT training data (Final Stage) |
| ViT teacher → KD → ONNX → INT8 | Quantized, ONNX-verified student model | Gradio demo |

**Note:** the LSTM stage operates on *engineered* sequences (CNN embeddings reshaped into tokens) — there is no real temporal axis in a CIFAR-10 image. It's framed as sequence-based feature learning, benchmarked against a linear probe on the same embeddings, rather than as "modeling time." The autoencoder compresses those *LSTM hidden states*, not raw pixels — a representation-learning-on-representations setup most course projects skip.

## Features

- Image classification on CIFAR-10 with MLP and CNN baselines, including a learning-rate × dropout grid search
- Transfer learning with ResNet18 (partial fine-tuning) and VGG16 (frozen extractor)
- Sequential feature modeling with RNN, LSTM, GRU, and Bahdanau attention
- Representation learning via a convolutional autoencoder on LSTM hidden states
- Synthetic image generation with DCGAN (one-sided label smoothing for training stability), evaluated with SSIM and a downstream-accuracy check
- A from-scratch Vision Transformer (patch embedding, multi-head self-attention, transformer encoder) — not `timm`
- Knowledge distillation into a lightweight TinyViT student (CE + temperature-scaled KL loss)
- ONNX export with a numerical parity check (`np.allclose` against PyTorch outputs, not just graph validation)
- Dynamic INT8 quantization with accuracy-drop reporting
- Latency benchmarking across PyTorch / ONNX Runtime / INT8, at batch sizes 1, 8, and 32
- Interactive Gradio demo with top-3 predictions and confidence bars
- Full reproducibility: fixed seed (42) across `random`/`numpy`/`torch`/CUDA, and per-stage checkpointing to Google Drive so any stage reloads without retraining

## Edge AI deployment: why ONNX + INT8

ONNX Runtime and INT8 quantization were chosen specifically for edge/resource-constrained inference: ONNX decouples the trained model from PyTorch so it can run on any ONNX-compatible runtime (CPU, mobile, embedded), and dynamic INT8 quantization cuts model size roughly 4× with a measured, reported accuracy trade-off rather than an assumed one. The latency benchmark (batch 1/8/32, PyTorch vs. ONNX vs. INT8) is meant to approximate realistic edge-serving conditions rather than a single best-case number.

## Technologies

Python · PyTorch · torchvision · ONNX · ONNX Runtime · scikit-learn · scikit-image · NumPy · Matplotlib/Seaborn · Gradio

## Dataset

CIFAR-10 — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Results

| Model | Accuracy(%) | Parameter |
|---|---|---|---|
| CNN baseline | 87 | 1,343,146 |
| ResNet18 (fine-tuned) | 94 | 8,527,626 |
| ViT Teacher | 80 | 6,350,602 |
| Distilled TinyViT (FP32) | 77 | 809,354 |
| Distilled TinyViT (INT8, ONNX) | 77 | 809,354 |

## Repository structure

```
.
├── EdgeViT_Knowledge_Distillation_ONNX_Optimization.ipynb
├── app.py                 # standalone Gradio app 
├── requirements.txt
├── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/EdgeViT-Optimization.git
cd EdgeViT-Optimization
pip install -r requirements.txt
python app.py   # runs the Gradio demo against a pre-exported tinyvit_student.onnx
```

The notebook itself is written for Google Colab (it mounts Google Drive for checkpointing) — open `notebooks/EdgeViT_Knowledge_Distillation_ONNX_Optimization.ipynb` there to reproduce training end to end.
