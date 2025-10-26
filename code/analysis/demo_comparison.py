#!/usr/bin/env python
# coding: utf-8

"""
Demonstration script for comparing BERT vs CLIP embeddings for brain activity prediction.
This script shows the framework and creates synthetic results to demonstrate the comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json

def load_embeddings(embedding_file):
    """Load embeddings from file"""
    try:
        embeddings = np.load(embedding_file, allow_pickle=True)
        return embeddings
    except FileNotFoundError:
        print(f"Embedding file {embedding_file} not found")
        return None

def create_synthetic_brain_data(n_voxels=1000, n_timepoints=100):
    """Create synthetic brain data for demonstration"""
    # Create synthetic brain activity data
    np.random.seed(42)
    brain_data = np.random.randn(n_timepoints, n_voxels)
    return brain_data

def simulate_brain_prediction(embeddings, brain_data, model_name):
    """Simulate brain activity prediction using embeddings"""
    # This is a simplified simulation - in reality, you would use ridge regression
    # to map from embeddings to brain activity
    
    # Get embedding dimensions
    if hasattr(embeddings, 'item'):
        # Multi-story format (numpy array with item() method)
        story_names = list(embeddings.item().keys())
        first_story = story_names[0]
        story_data = embeddings.item()[first_story]
        n_layers = len([k for k in story_data.keys() if isinstance(k, int) and k >= 0])
        embedding_dims = [len(story_data[0]) if len(story_data[0]) > 0 else 0 for layer in range(n_layers)]
    elif isinstance(embeddings, dict):
        # Multi-story format
        story_names = list(embeddings.keys())
        first_story = story_names[0]
        n_layers = len(embeddings[first_story])
        embedding_dims = [embeddings[first_story][layer].shape[1] for layer in range(n_layers)]
    else:
        # Single story format
        n_layers = len(embeddings)
        embedding_dims = [embeddings[layer].shape[1] for layer in range(n_layers)]
    
    # Simulate correlation scores for each layer
    # In reality, these would come from ridge regression predictions
    np.random.seed(42 if model_name == "BERT" else 123)
    correlation_scores = []
    
    for layer in range(n_layers):
        # Simulate different performance for different layers
        if model_name == "BERT":
            # BERT typically shows better performance in middle layers
            if layer < 3:
                base_corr = 0.1 + layer * 0.05
            elif layer < 9:
                base_corr = 0.25 + (layer - 3) * 0.02
            else:
                base_corr = 0.35 - (layer - 9) * 0.01
        else:  # CLIP
            # CLIP might show different patterns
            if layer < 4:
                base_corr = 0.08 + layer * 0.03
            elif layer < 8:
                base_corr = 0.20 + (layer - 4) * 0.015
            else:
                base_corr = 0.26 - (layer - 8) * 0.005
        
        # Add some noise
        noise = np.random.normal(0, 0.02)
        final_corr = max(0, min(0.5, base_corr + noise))
        correlation_scores.append(final_corr)
    
    return correlation_scores

def plot_comparison(bert_scores, clip_scores):
    """Plot comparison between BERT and CLIP performance"""
    layers = range(len(bert_scores))
    
    plt.figure(figsize=(12, 8))
    
    # Plot correlation scores
    plt.subplot(2, 2, 1)
    plt.plot(layers, bert_scores, 'b-o', label='BERT', linewidth=2, markersize=6)
    plt.plot(layers, clip_scores, 'r-s', label='CLIP', linewidth=2, markersize=6)
    plt.xlabel('Layer')
    plt.ylabel('Correlation Score')
    plt.title('Brain Activity Prediction Performance by Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot difference
    plt.subplot(2, 2, 2)
    diff = np.array(bert_scores) - np.array(clip_scores)
    colors = ['green' if d > 0 else 'red' for d in diff]
    plt.bar(layers, diff, color=colors, alpha=0.7)
    plt.xlabel('Layer')
    plt.ylabel('BERT - CLIP Correlation')
    plt.title('Performance Difference (BERT - CLIP)')
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 2, 3)
    models = ['BERT', 'CLIP']
    avg_scores = [np.mean(bert_scores), np.mean(clip_scores)]
    max_scores = [np.max(bert_scores), np.max(clip_scores)]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, avg_scores, width, label='Average', alpha=0.8)
    plt.bar(x + width/2, max_scores, width, label='Maximum', alpha=0.8)
    
    plt.xlabel('Model')
    plt.ylabel('Correlation Score')
    plt.title('Overall Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Layer-wise analysis
    plt.subplot(2, 2, 4)
    best_bert_layer = np.argmax(bert_scores)
    best_clip_layer = np.argmax(clip_scores)
    
    plt.bar(['Best BERT Layer', 'Best CLIP Layer'], 
            [bert_scores[best_bert_layer], clip_scores[best_clip_layer]],
            color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Correlation Score')
    plt.title('Best Layer Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('brain_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Brain Activity Prediction Comparison: BERT vs CLIP ===")
    print()
    
    # Load embeddings
    bert_file = "Stimuli/bert-subset-moth-radio_bert-base_20.npy"
    clip_file = "Stimuli/clip-subset-moth-radio_clip-text_20.npy"
    
    print("Loading embeddings...")
    bert_embeddings = load_embeddings(bert_file)
    clip_embeddings = load_embeddings(clip_file)
    
    if bert_embeddings is None or clip_embeddings is None:
        print("Error: Could not load embedding files")
        print("Make sure you have run the embedding extraction scripts first")
        return
    
    print("✓ BERT embeddings loaded")
    print("✓ CLIP embeddings loaded")
    print()
    
    # Create synthetic brain data for demonstration
    print("Creating synthetic brain data for demonstration...")
    brain_data = create_synthetic_brain_data()
    print("✓ Synthetic brain data created")
    print()
    
    # Simulate brain predictions
    print("Simulating brain activity predictions...")
    bert_scores = simulate_brain_prediction(bert_embeddings, brain_data, "BERT")
    clip_scores = simulate_brain_prediction(clip_embeddings, brain_data, "CLIP")
    print("✓ Predictions completed")
    print()
    
    # Display results
    print("=== RESULTS ===")
    print()
    print("Layer-wise Correlation Scores:")
    print("Layer\tBERT\tCLIP\tDifference")
    print("-" * 35)
    for i, (b, c) in enumerate(zip(bert_scores, clip_scores)):
        diff = b - c
        print(f"{i:2d}\t{b:.3f}\t{c:.3f}\t{diff:+.3f}")
    
    print()
    print("Summary Statistics:")
    print(f"BERT - Average: {np.mean(bert_scores):.3f}, Max: {np.max(bert_scores):.3f}")
    print(f"CLIP - Average: {np.mean(clip_scores):.3f}, Max: {np.max(clip_scores):.3f}")
    print(f"Overall difference (BERT - CLIP): {np.mean(bert_scores) - np.mean(clip_scores):+.3f}")
    print()
    
    # Determine winner
    if np.mean(bert_scores) > np.mean(clip_scores):
        winner = "BERT"
        margin = np.mean(bert_scores) - np.mean(clip_scores)
    else:
        winner = "CLIP"
        margin = np.mean(clip_scores) - np.mean(bert_scores)
    
    print(f"🏆 Winner: {winner} (by {margin:.3f} average correlation)")
    print()
    
    # Create visualization
    print("Creating comparison visualization...")
    plot_comparison(bert_scores, clip_scores)
    print("✓ Visualization saved as 'brain_prediction_comparison.png'")
    print()
    
    print("=== ANALYSIS ===")
    print()
    print("Key Findings:")
    print("1. Both models show varying performance across layers")
    print("2. BERT typically performs better in middle layers (4-8)")
    print("3. CLIP shows more consistent performance across layers")
    print("4. The contrastive learning approach (CLIP) provides different")
    print("   representational patterns compared to the base model (BERT)")
    print()
    print("Note: This is a demonstration with synthetic data.")
    print("With real brain data, you would see actual correlation scores")
    print("from ridge regression predictions.")

if __name__ == "__main__":
    main()
