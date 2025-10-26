# Usage Guide

This document provides comprehensive instructions for using the brain model comparison framework.

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/brain-model-comparison.git
cd brain-model-comparison

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Analysis
```bash
# Run the main comparison demo
python code/analysis/demo_comparison.py

# This will generate:
# - brain_prediction_comparison.png
# - Console output with results
```

## Detailed Usage

### 1. Embedding Extraction

#### Extract BERT Embeddings
```bash
python code/embeddings/extract_bert_embeddings.py \
    --input_dir data/stories \
    --model bert-base \
    --sequence_length 20 \
    --output_file data/embeddings/bert-subset-moth-radio
```

#### Extract CLIP Embeddings
```bash
python code/embeddings/extract_clip_embeddings.py \
    --input_dir data/stories \
    --model clip-text \
    --sequence_length 20 \
    --output_file data/embeddings/clip-subset-moth-radio
```

#### Parameters
- `--input_dir`: Directory containing story text files
- `--model`: Model name (bert-base, clip-text)
- `--sequence_length`: Number of words per sequence (default: 20)
- `--output_file`: Output filename prefix

### 2. Brain Activity Prediction

#### Run Brain Prediction Analysis
```bash
python code/brain_prediction/brain_predictions_subset.py \
    [subject_num] \
    [feature_file] \
    [modality] \
    [directory] \
    [num_layers]
```

#### Example
```bash
python code/brain_prediction/brain_predictions_subset.py \
    1 \
    data/embeddings/bert-subset-moth-radio_bert-base_20.npy \
    text \
    data/ \
    12
```

#### Parameters
- `subject_num`: Subject number (1-8)
- `feature_file`: Path to embedding file (.npy)
- `modality`: Modality type (text, audio, etc.)
- `directory`: Data directory path
- `num_layers`: Number of layers to analyze (12 for BERT/CLIP)

### 3. Statistical Analysis

#### Run Statistical Analysis
```bash
python code/analysis/statistical_analysis.py
```

#### Output
- `statistical_summary.png`: Comprehensive statistical analysis
- `layer_analysis_detailed.png`: Detailed layer-wise analysis
- Console output with statistical results

#### Key Results
```
=== STATISTICAL RESULTS ===
Paired t-test: t = 1.007, p = 0.351
Effect size (Cohen's d): 1.007 (large)
BERT mean: 0.277 ± 0.082
CLIP mean: 0.201 ± 0.061
Difference: 0.076
```

### 4. Representational Analysis

#### Run Representational Analysis
```bash
python code/analysis/representational_analysis.py
```

#### Output
- `representational_spaces.png`: Representational space analysis
- `semantic_heatmaps.png`: Semantic similarity heatmaps
- Console output with analysis results

## Advanced Usage

### 1. Custom Analysis

#### Load Embeddings
```python
import numpy as np

# Load BERT embeddings
bert_embeddings = np.load('data/embeddings/bert-subset-moth-radio_bert-base_20.npy', allow_pickle=True)

# Load CLIP embeddings
clip_embeddings = np.load('data/embeddings/clip-subset-moth-radio_clip-text_20.npy', allow_pickle=True)

# Get story names
story_names = list(bert_embeddings.item().keys())
print(f"Available stories: {story_names}")
```

#### Extract Layer Embeddings
```python
def extract_layer_embeddings(embeddings, story_name, layer_idx):
    """Extract embeddings for a specific story and layer"""
    story_data = embeddings.item()[story_name]
    layer_embeddings = np.array(story_data[layer_idx])
    return layer_embeddings

# Example usage
bert_layer_0 = extract_layer_embeddings(bert_embeddings, 'alternateithicatom', 0)
clip_layer_0 = extract_layer_embeddings(clip_embeddings, 'alternateithicatom', 0)
```

#### Custom Brain Prediction
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np

def predict_brain_activity(embeddings, brain_data, alpha=1.0):
    """Predict brain activity using ridge regression"""
    model = Ridge(alpha=alpha)
    
    # Cross-validation
    scores = cross_val_score(model, embeddings, brain_data, cv=5, scoring='r2')
    
    # Fit on full data
    model.fit(embeddings, brain_data)
    predictions = model.predict(embeddings)
    
    # Correlation
    correlation = np.corrcoef(predictions, brain_data)[0, 1]
    
    return {
        'correlation': correlation,
        'cv_scores': scores,
        'cv_mean': scores.mean(),
        'cv_std': scores.std()
    }
```

### 2. Custom Visualizations

#### Create Custom Plots
```python
import matplotlib.pyplot as plt
import numpy as np

def plot_custom_comparison(bert_scores, clip_scores, save_path=None):
    """Create custom comparison plot"""
    plt.figure(figsize=(12, 8))
    layers = range(1, 13)
    
    plt.plot(layers, bert_scores, 'b-o', label='BERT', linewidth=2)
    plt.plot(layers, clip_scores, 'r-s', label='CLIP', linewidth=2)
    
    plt.xlabel('Layer')
    plt.ylabel('Correlation with Brain Activity')
    plt.title('Custom Layer-wise Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
```

### 3. Batch Processing

#### Process Multiple Models
```python
import os
import subprocess

def extract_all_embeddings():
    """Extract embeddings for all models"""
    models = ['bert-base', 'clip-text']
    
    for model in models:
        print(f"Extracting {model} embeddings...")
        
        if model == 'bert-base':
            script = 'code/embeddings/extract_bert_embeddings.py'
        else:
            script = 'code/embeddings/extract_clip_embeddings.py'
        
        cmd = [
            'python', script,
            '--input_dir', 'data/stories',
            '--model', model,
            '--sequence_length', '20',
            '--output_file', f'data/embeddings/{model}-subset-moth-radio'
        ]
        
        subprocess.run(cmd)

# Run extraction
extract_all_embeddings()
```

## Configuration

### 1. Model Configuration

#### Edit `text_model_config.json`
```json
{
    "bert-base": {
        "huggingface_hub": "bert-base-uncased",
        "num_layers": 12,
        "model_type": "encoder"
    },
    "clip-text": {
        "huggingface_hub": "openai/clip-vit-base-patch32",
        "num_layers": 12,
        "model_type": "encoder"
    }
}
```

### 2. Analysis Parameters

#### Modify Analysis Scripts
```python
# In statistical_analysis.py
N_BOOTSTRAP = 1000  # Number of bootstrap iterations
CONFIDENCE_LEVEL = 0.95  # Confidence level for CIs
ALPHA_LEVEL = 0.05  # Significance level

# In demo_comparison.py
SEQUENCE_LENGTH = 20  # Word sequence length
N_LAYERS = 12  # Number of layers to analyze
```

## Troubleshooting

### 1. Common Issues

#### CUDA Out of Memory
```python
# Use CPU instead of GPU
device = "cpu"  # Instead of "cuda:0"
```

#### Missing Dependencies
```bash
# Install missing packages
pip install torch transformers scikit-learn matplotlib seaborn numpy pandas scipy
```

#### File Not Found Errors
```bash
# Check file paths
ls -la data/embeddings/
ls -la data/stories/
```

### 2. Performance Optimization

#### Reduce Memory Usage
```python
# Process stories one at a time
for story in story_names:
    process_story(story)
    del story_data  # Free memory
```

#### Use Smaller Sequences
```bash
# Use shorter sequences
python code/embeddings/extract_bert_embeddings.py \
    --sequence_length 10  # Instead of 20
```

### 3. Debugging

#### Enable Debug Output
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Data Shapes
```python
# Verify embedding shapes
print(f"BERT shape: {bert_embeddings.shape}")
print(f"CLIP shape: {clip_embeddings.shape}")
```

## Examples

### 1. Complete Analysis Pipeline
```bash
# 1. Extract embeddings
python code/embeddings/extract_bert_embeddings.py \
    --input_dir data/stories \
    --model bert-base \
    --sequence_length 20 \
    --output_file data/embeddings/bert-subset-moth-radio

python code/embeddings/extract_clip_embeddings.py \
    --input_dir data/stories \
    --model clip-text \
    --sequence_length 20 \
    --output_file data/embeddings/clip-subset-moth-radio

# 2. Run analysis
python code/analysis/demo_comparison.py
python code/analysis/statistical_analysis.py
python code/analysis/representational_analysis.py

# 3. Check results
ls -la figures/
```

### 2. Custom Story Analysis
```python
# Analyze specific stories
story_names = ['alternateithicatom', 'avatar', 'howtodraw']

for story in story_names:
    print(f"Analyzing {story}...")
    
    # Extract embeddings for this story
    bert_story = extract_layer_embeddings(bert_embeddings, story, 0)
    clip_story = extract_layer_embeddings(clip_embeddings, story, 0)
    
    # Analyze
    similarity = compute_similarity(bert_story, clip_story)
    print(f"Similarity: {similarity:.3f}")
```

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the methodology documentation
3. Open an issue on GitHub
4. Contact the author

## Citation

When using this framework, please cite:
```bibtex
@article{brain_model_comparison_2024,
  title={Comparing Language-Specific vs. Multimodal Contrastive Learning for Brain Activity Prediction: A Framework for Model Architecture Analysis},
  author={Moselmany, Khaled},
  journal={Neural Networks in Brains and Computers Seminar},
  year={2024}
}
```

