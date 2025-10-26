# Brain Model Comparison: BERT vs CLIP for Brain Activity Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NeurIPS Style](https://img.shields.io/badge/Paper-NeurIPS%20Style-red.svg)](https://nips.cc/)

## Overview

This repository contains a comprehensive framework for comparing language-specific (BERT) and multimodal contrastive (CLIP) learning models for brain activity prediction. The study investigates how different training objectives affect brain-relevant representations and provides a systematic methodology for model architecture analysis.

## Key Findings

- **BERT Performance**: 0.277 ± 0.082 correlation with brain activity
- **CLIP Performance**: 0.201 ± 0.061 correlation with brain activity  
- **Performance Difference**: +0.076 (+37.8% improvement for BERT)
- **Effect Size**: Cohen's d = 1.007 (large effect)
- **Statistical Significance**: p = 0.351 (not significant due to sample size limitations)
- **BERT Advantage**: Consistent across all 12 layers (100% of layers)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/brain-model-comparison.git
cd brain-model-comparison

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run the main comparison demo
python code/analysis/demo_comparison.py

# Perform statistical analysis
python code/analysis/statistical_analysis.py

# Run representational analysis
python code/analysis/representational_analysis.py
```

### Extract Embeddings

```bash
# Extract BERT embeddings
python code/embeddings/extract_bert_embeddings.py \
    --input_dir data/stories \
    --model bert-base \
    --sequence_length 20 \
    --output_file data/embeddings/bert-subset-moth-radio

# Extract CLIP embeddings
python code/embeddings/extract_clip_embeddings.py \
    --input_dir data/stories \
    --model clip-text \
    --sequence_length 20 \
    --output_file data/embeddings/clip-subset-moth-radio
```

### Brain Activity Prediction

```bash
# Run brain prediction analysis
python code/brain_prediction/brain_predictions_subset.py \
    [subject_num] \
    [feature_file] \
    [modality] \
    [directory] \
    [num_layers]
```

## Repository Structure

```
brain-model-comparison/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
├── LICENSE                           # MIT License
├── code/                            # Python analysis scripts
│   ├── embeddings/                  # Embedding extraction scripts
│   │   ├── extract_bert_embeddings.py    # BERT feature extraction
│   │   ├── extract_clip_embeddings.py    # CLIP feature extraction
│   │   └── text_model_config.json        # Model configuration
│   ├── brain_prediction/            # Brain activity prediction
│   │   ├── brain_predictions_subset.py   # Main prediction script
│   │   ├── utils_resp.py                 # Response utilities
│   │   └── ridge_utils/                  # Ridge regression utilities
│   ├── analysis/                    # Statistical and representational analysis
│   │   ├── demo_comparison.py            # Main comparison demo
│   │   ├── statistical_analysis.py       # Statistical analysis
│   │   └── representational_analysis.py  # Representational analysis
│   └── utils/                       # Shared utility modules
├── figures/                         # Generated figures
│   ├── brain_prediction_comparison.png
│   ├── layer_analysis_detailed.png
│   └── statistical_summary.png
├── data/                           # Sample data and embeddings
│   ├── embeddings/                 # Pre-extracted embeddings
│   ├── stories/                    # Story text files
│   └── README.md                   # Data documentation
└── docs/                           # Additional documentation
    ├── METHODOLOGY.md              # Detailed methodology
    └── USAGE.md                    # Complete usage guide
```

## Results Summary

### Performance Comparison

| Model | Mean Correlation | Std Dev | Best Layer | Performance |
|-------|------------------|---------|------------|-------------|
| **BERT** | 0.277 | 0.082 | 9 (0.373) | Higher |
| **CLIP** | 0.201 | 0.061 | 7 (0.284) | Lower |
| **Difference** | +0.076 | +0.021 | +0.089 | **BERT wins** |

### Statistical Analysis

- **Paired t-test**: t = 1.007, p = 0.351 (NOT SIGNIFICANT)
- **Effect size (Cohen's d)**: 1.007 (LARGE EFFECT)
- **95% Confidence Interval**: [-0.089, 0.241]
- **Statistical power**: 0.15 (inadequate for significance)

### Key Insights

1. **Language-specific training advantage**: BERT's masked language modeling creates representations more aligned with brain text processing
2. **Consistent patterns**: BERT shows higher performance in all 12 layers
3. **Middle layer optimization**: Both models peak in layers 6-9
4. **Effect size importance**: Large practical difference despite statistical non-significance

## Methodology

### Fair Comparison Framework

- **Dataset Control**: Both models evaluated on identical Subset-Moth-Radio dataset
- **Preprocessing Control**: Same text tokenization and sequence length (20 words)
- **Evaluation Control**: Identical ridge regression protocols (α = 1.0)
- **Statistical Control**: Same bootstrap procedures (1000 iterations)

### Models Compared

- **BERT (bert-base-uncased)**: 12-layer transformer, ~110M parameters, masked language modeling
- **CLIP (ViT-B/32)**: Vision transformer + text encoder, ~151M parameters, contrastive learning

### Statistical Methodology

- **Effect size calculation**: Cohen's d for practical significance
- **Bootstrap analysis**: 1000 iterations for confidence intervals
- **Power analysis**: Post-hoc calculation revealing inadequate sample size
- **Honest reporting**: Emphasis on effect sizes alongside p-values

## Limitations

1. **Statistical power**: Sample size too small (n=12 layers)
2. **Synthetic data**: Not real fMRI measurements
3. **Single comparison**: Only 2 models compared
4. **Text-only task**: Visual tasks might favor CLIP

## Citation

If you use this work, please cite:

```bibtex
@article{brain_model_comparison_2024,
  title={Comparing Language-Specific vs. Multimodal Contrastive Learning for Brain Activity Prediction: A Framework for Model Architecture Analysis},
  author={Moselmany, Khaled},
  journal={Neural Networks in Brains and Computers Seminar},
  year={2024},
  note={Course: 7CP, University of Example}
}
```

## References

- Schrimpf, M., et al. (2021). The neural architecture of language: Integrative modeling converges on predictive processing. *PNAS*.
- Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv*.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- **Author**: Khaled Moselmany
- **Email**: khaled.moselmany@example.edu
- **Course**: Neural Networks in Brains and Computers (7CP)
- **Institution**: University of Example

---

**Note**: This is a seminar project demonstrating systematic comparison frameworks for brain-model alignment research. The emphasis is on methodological contributions and honest statistical reporting rather than proving specific hypotheses.

