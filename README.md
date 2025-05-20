# Evolutionary Training Manager for YOLO Models

An evolutionary algorithm-based training system for YOLO object detection models. This project automatically optimises both model architecture and training parameters to achieve the best trade-off between accuracy and inference speed.

## Features

- Evolutionary optimisation of YOLO model architecture and hyperparameters
- Hardware-aware training optimised for both NVIDIA (CUDA) and AMD (ROCm) GPUs
- Automatic resource management to prevent VRAM exhaustion
- Comprehensive checkpointing system for resuming training
- Population-based approach with configurable selection, crossover, and mutation
- Built on top of the TrainingAutomation framework for YOLO model training

## Requirements

- Python 3.8 or higher
- PyTorch 1.7 or higher
- GPU with CUDA or ROCm support (optional but recommended)

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:SethBennett2523/EvolutionaryTrainingManager.git
   cd EvolutionaryTrainingManager
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Update the configuration in `config.yaml` to match your environment.

## Usage

### Basic usage

```bash
python main.py --config config.yaml
```

### Additional options

- `--resume`: Resume training from the last checkpoint
- `--output-dir OUTPUT_DIR`: Specify output directory
- `--generations N`: Set maximum number of generations
- `--population N`: Set population size
- `--device {cuda,rocm,cpu,auto}`: Specify device to use
- `--debug`: Enable debug logging

## Configuration

The main configuration file (`config.yaml`) includes settings for:

- Evolution parameters (population size, mutation rate, etc.)
- Model parameters (base model type)
- Hardware settings (device, memory threshold)
- Hyperparameter ranges for evolution

## How It Works

1. The algorithm initialises a population of YOLO models with random hyperparameters and architecture settings
2. Each model is trained on the dataset and evaluated for accuracy (mAP) and speed (inference time)
