# Fine-Tuning Mistral 7B for Query Expansion

The report for the internship during which this work was done is: [Internship Report](https://github.com/Nanashi-bot/finetuning_mistral/blob/main/Internship%20report.pdf)

This repository contains code to fine-tune the Mistral 7B model using LoRA adapters and evaluate its performance on query expansion tasks. The goal is to take a short query and predict potential larger questions it could originate from.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Datasets Used](#datasets-used)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Overview

The fine-tuning process involves:
1. Loading the Mistral 7B model with 4-bit quantization for efficient training.
2. Using LoRA adapters to fine-tune the model on query-description pairs.
3. Training on datasets like TREC Web Track 2009, TREC Robust 2004, and TREC Gov Web 2002.
4. Evaluating the model by generating descriptions for unseen queries.

## Setup

1. Clone the repository:

```
git clone https://github.com/yourusername/finetuning_mistral.git cd finetuning_mistral
```

2. Create a virtual environment and install dependencies:
```
conda create -n mistral_finetune python=3.10
conda activate mistral_finetune
pip install torch transformers peft accelerate datasets ir_datasets
```

3. Ensure you have an active Hugging Face token for model access:

```
huggingface-cli login
```


## Training the Model

The `training.py` script fine-tunes the Mistral 7B model using LoRA adapters. It follows these steps:
1. Loads the base Mistral model with 4-bit quantization.
2. Prepares datasets by splitting into training, evaluation, and testing sets.
3. Applies LoRA adapters for efficient training.
4. Saves checkpoints for evaluation.

To train the model:
```
python training.py
```

This will:
- Load datasets (`clueweb09/catb/trec-web-2009`, `disks45/nocr/trec-robust-2004`, `gov/trec-web-2002`).
- Split data into training (280 samples), evaluation (35 samples), and testing (35 samples).
- Save fine-tuned models in the specified checkpoint directories.

## Testing the Model

The `testing_finetuned_model.py` script evaluates the fine-tuned model. It:
1. Loads the fine-tuned model from the checkpoint.
2. Generates expanded queries from unseen test samples.
3. Compares generated results with the original dataset descriptions.

To test the model:
```
python testing_finetuned_model.py
```


Example output:

```
QUERY: internet privacy

DESCRIPTION THAT MISTRAL GENERATES: How can individuals protect their personal information online?

DESCRIPTION FROM DATASET: What are effective methods for ensuring online privacy?
```


## Datasets Used

The following datasets were used for fine-tuning and evaluation:

1. **TREC Web Track 2009** (`clueweb09/catb/trec-web-2009`): 50 queries with descriptions.
2. **TREC Robust 2004** (`disks45/nocr/trec-robust-2004`): 250 queries with descriptions.
3. **TREC Gov Web 2002** (`gov/trec-web-2002`): 50 queries with descriptions.

Data is split as follows:
- **Training:** 280 samples
- **Evaluation:** 35 samples
- **Testing:** 35 samples

## Results

The fine-tuned Mistral 7B model generates meaningful query expansions, aligning closely with the dataset-provided descriptions. It demonstrates strong generalization across multiple datasets, showing promise for real-world query expansion tasks.

## Acknowledgments

Special thanks to:
- The Hugging Face team for the `transformers` and `peft` libraries.
- The creators of the IR datasets used for fine-tuning and evaluation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
