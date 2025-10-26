# Data Directory

This directory contains the data files used in the brain model comparison study.

## Structure

```
data/
├── embeddings/                 # Pre-extracted model embeddings
│   ├── bert-subset-moth-radio_bert-base_20.npy
│   └── clip-subset-moth-radio_clip-text_20.npy
├── stories/                    # Story text files
│   ├── alternateithicatom.txt
│   ├── avatar.txt
│   ├── howtodraw.txt
│   ├── legacy.txt
│   ├── life.txt
│   ├── myfirstdaywiththeyankees.txt
│   ├── naked.txt
│   ├── odetostepfather.txt
│   ├── souls.txt
│   ├── undertheinfluence.txt
│   └── wheretheressmoke.txt
└── README.md                   # This file
```

## Embeddings

### BERT Embeddings (`bert-subset-moth-radio_bert-base_20.npy`)
- **Model**: bert-base-uncased
- **Sequence Length**: 20 words
- **Format**: NumPy array with nested dictionary structure
- **Structure**: `{story_name: {layer_idx: embeddings}}`
- **Layers**: 12 transformer layers (0-11) + token embeddings (-1)
- **Dataset**: Subset-Moth-Radio (11 stories)

### CLIP Embeddings (`clip-subset-moth-radio_clip-text_20.npy`)
- **Model**: openai/clip-vit-base-patch32 (text encoder)
- **Sequence Length**: 20 words
- **Format**: NumPy array with nested dictionary structure
- **Structure**: `{story_name: {layer_idx: embeddings}}`
- **Layers**: 12 text encoder layers (0-11) + token embeddings (-1)
- **Dataset**: Subset-Moth-Radio (11 stories)

## Stories

The `stories/` directory contains 11 text files from the Subset-Moth-Radio dataset. Each file contains one story with words separated by newlines.

### Story List
1. `alternateithicatom.txt` - Story about atomic theory
2. `avatar.txt` - Story about avatars
3. `howtodraw.txt` - Story about drawing
4. `legacy.txt` - Story about legacy
5. `life.txt` - Story about life
6. `myfirstdaywiththeyankees.txt` - Story about baseball
7. `naked.txt` - Story about being naked
8. `odetostepfather.txt` - Story about stepfather
9. `souls.txt` - Story about souls
10. `undertheinfluence.txt` - Story about influence
11. `wheretheressmoke.txt` - Story about smoke

## Data Format

### Embedding Structure
```python
import numpy as np

# Load embeddings
embeddings = np.load('bert-subset-moth-radio_bert-base_20.npy', allow_pickle=True)

# Access story data
story_name = 'alternateithicatom'
story_data = embeddings.item()[story_name]

# Access layer embeddings
layer_0_embeddings = story_data[0]  # First transformer layer
layer_11_embeddings = story_data[11]  # Last transformer layer
token_embeddings = story_data[-1]  # Token embeddings

# Shape: (n_sequences, embedding_dim)
print(f"Layer 0 shape: {layer_0_embeddings.shape}")
```

### Story Format
```python
# Read story file
with open('stories/alternateithicatom.txt', 'r') as f:
    words = f.read().strip().split('\n')

print(f"Number of words: {len(words)}")
print(f"First 10 words: {words[:10]}")
```

## Usage

### Loading Embeddings
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

### Extracting Layer Embeddings
```python
def extract_layer_embeddings(embeddings, story_name, layer_idx):
    """Extract embeddings for a specific story and layer"""
    story_data = embeddings.item()[story_name]
    layer_embeddings = np.array(story_data[layer_idx])
    return layer_embeddings

# Example usage
bert_layer_0 = extract_layer_embeddings(bert_embeddings, 'alternateithicatom', 0)
clip_layer_0 = extract_layer_embeddings(clip_embeddings, 'alternateithicatom', 0)

print(f"BERT layer 0 shape: {bert_layer_0.shape}")
print(f"CLIP layer 0 shape: {clip_layer_0.shape}")
```

## Regenerating Embeddings

If you need to regenerate the embeddings, use the extraction scripts:

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

## Notes

- Embeddings are extracted using 20-word sequences with the last word as the target
- All embeddings are normalized and ready for analysis
- The data is compatible with the brain prediction scripts in `code/brain_prediction/`
- For more details on the extraction process, see `code/embeddings/` scripts

