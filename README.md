# DINOv1: Self-Supervised Vision Transformers

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of DINO (Distillation with No labels), a self-supervised learning method for Vision Transformers. This repository provides a clean, modular implementation of the DINO framework for training visual representations without labels.

*Based on the original work from [group project](https://github.com/Coartix/DNN_Dino/) and reimplemented with improvements.*

## 🔥 Features

- **Self-supervised learning**: Train Vision Transformers without labeled data
- **Modular architecture**: Clean separation of models, training, and evaluation components
- **Configurable training**: YAML-based configuration system with Pydantic validation
- **Comprehensive evaluation**: k-NN classification and linear evaluation protocols
- **Modern PyTorch**: Built with latest PyTorch best practices

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Configuration](#configuration)
- [Results](#-results)
- [Advanced Usage](#-advanced-usage)
- [Contributing](#-contributing)
- [References](#-references)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)


## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- [Poetry](https://python-poetry.org/) (for dependency management)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-name>
   cd DINOv1
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

3. **Or install using pip:**
   ```bash
   pip install torch torchvision torchaudio
   pip install pyyaml pydantic tqdm matplotlib pillow
   ```

4. **Activate the environment:**
   ```bash
   poetry shell
   ```

## ⚡ Quick Start

### Training a DINO Model

```bash
# Train with default configuration
python train.py --config configs/dino.yml

# Train with custom settings
python train.py --config configs/dino.yml --epochs 100 --batch_size 64
```

### Evaluating a Trained Model

```bash
# Evaluate with k-NN classifier
python eval.py --model_path training_output/model_checkpoint.pth --config configs/dino.yml

# Evaluate with specific k value
python eval.py --model_path training_output/model_checkpoint.pth --k 20
```

## 📁 Project Structure

```
DINOv1/
├── configs/                # Configuration files
│   ├── config_models.py    # Pydantic model definitions
│   ├── dino.yml            # Main DINO configuration
│   └── dino_head.yml       # DINO head configuration
├── models/                 # Model implementations
│   ├── DINO.py             # Main DINO model
│   ├── DINO_head.py        # DINO projection head
│   └── DINO_loss.py        # DINO loss function
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── Trainer.py              # Training orchestration
├── utils.py                # Utility functions and classes
└── README.md               # This file
```

### Key Components

- **`DINO.py`**: Core DINO model implementation with student-teacher architecture
- **`DINO_head.py`**: Projection head for feature transformation
- **`DINO_loss.py`**: Self-distillation loss computation
- **`Trainer.py`**: Handles training loop, checkpointing, and logging
- **`utils.py`**: Data loading, augmentations, and helper functions

## 🎯 Training

### Configuration

Training parameters are specified in YAML configuration files. Key parameters include:

```yaml
# Example: configs/dino.yml
model:
  arch: "vit_small"
  patch_size: 16
  out_dim: 65536

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.0005
  warmup_teacher_temp: 0.04
  teacher_temp: 0.07

data:
  dataset: "ImageNet"
  data_path: "/path/to/dataset"
  num_workers: 8
```

### Training Commands

```bash
# Basic training
python train.py --config configs/dino.yml

# Training with custom parameters
python train.py --config configs/dino.yml \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 0.001

# Resume from checkpoint
python train.py --config configs/dino.yml \
    --resume training_output/checkpoint.pth
```

### Monitoring Training

Training progress is logged and can be monitored through:
- Console output with training metrics
- Saved checkpoints in `training_output/`
- Loss curves and learning rate schedules

## 📊 Evaluation

### k-Nearest Neighbors (k-NN) Classification

Evaluate learned representations using k-NN classification:

```bash
# Standard k-NN evaluation
python eval.py --model_path training_output/model.pth --k 20

# Multiple k values
python eval.py --model_path training_output/model.pth --k 1,5,10,20,50,100
```

### Linear Classification

For linear evaluation protocol:

```bash
# Linear evaluation (freeze features)
python eval.py --model_path training_output/model.pth --linear_eval
```

## ⚙️ Configuration

The project uses YAML configuration files validated with Pydantic models. This ensures type safety and prevents configuration errors.

### Creating Custom Configurations

1. Copy an existing config file:
   ```bash
   cp configs/dino.yml configs/my_config.yml
   ```

2. Modify parameters as needed

3. Validate configuration:
   ```python
   from configs.config_models import DinoConfig
   config = DinoConfig.from_yaml("configs/my_config.yml")
   ```

## 📈 Results

Expected performance on ImageNet-1K:

| Model | Architecture | k-NN Accuracy | Linear Accuracy |
|-------|-------------|---------------|-----------------|
| DINO  | ViT-S/16    | ~74.5%        | ~77.0%         |
| DINO  | ViT-S/8     | ~76.1%        | ~79.2%         |

*Results may vary depending on training configuration and hardware.*

## 🛠️ Advanced Usage

### Custom Data Loaders

Extend the data loading functionality by modifying `utils.py`:

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        # Your custom dataset implementation
        pass
```

### Custom Augmentations

Modify augmentation strategies in the configuration:

```yaml
augmentation:
  global_crops_scale: [0.4, 1.0]
  local_crops_scale: [0.05, 0.4]
  local_crops_number: 8
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 References

- **Original DINO Paper**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **Self-Supervised Learning**: [A Survey on Self-Supervised Learning](https://arxiv.org/abs/1902.06162)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original DINO implementation by Facebook AI Research
- Vision Transformer implementation by Google Research
- Community contributions and feedback
