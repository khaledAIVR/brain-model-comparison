#!/usr/bin/env python
# coding: utf-8

"""
Representational Analysis Framework
Compare embedding spaces between BERT and CLIP to understand how different training objectives
affect brain-relevant representations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import pandas as pd

def load_embeddings(bert_file, clip_file):
    """Load BERT and CLIP embeddings"""
    print("Loading embeddings...")
    bert_embeddings = np.load(bert_file, allow_pickle=True)
    clip_embeddings = np.load(clip_file, allow_pickle=True)
    print("✓ Embeddings loaded")
    return bert_embeddings, clip_embeddings

def extract_layer_embeddings(embeddings, layer_idx, story_name):
    """Extract embeddings for a specific layer and story"""
    story_data = embeddings.item()[story_name]
    layer_embeddings = np.array(story_data[layer_idx])
    return layer_embeddings

def compute_semantic_similarity(embeddings1, embeddings2):
    """Compute semantic similarity between two sets of embeddings"""
    # Handle dimension mismatch by projecting to common dimension
    min_dim = min(embeddings1.shape[1], embeddings2.shape[1])
    
    # Project to common dimension using PCA
    from sklearn.decomposition import PCA
    
    pca1 = PCA(n_components=min_dim)
    pca2 = PCA(n_components=min_dim)
    
    emb1_proj = pca1.fit_transform(embeddings1)
    emb2_proj = pca2.fit_transform(embeddings2)
    
    # Normalize embeddings
    emb1_norm = emb1_proj / np.linalg.norm(emb1_proj, axis=1, keepdims=True)
    emb2_norm = emb2_proj / np.linalg.norm(emb2_proj, axis=1, keepdims=True)
    
    # Compute cosine similarity
    similarity_matrix = np.dot(emb1_norm, emb2_norm.T)
    return similarity_matrix

def analyze_representational_structure(embeddings, model_name, layer_idx):
    """Analyze the representational structure of embeddings"""
    print(f"Analyzing {model_name} layer {layer_idx}...")
    
    # Dimensionality analysis
    n_samples, n_features = embeddings.shape
    print(f"  Shape: {n_samples} samples, {n_features} features")
    
    # PCA analysis
    pca = PCA()
    pca.fit(embeddings)
    explained_variance = pca.explained_variance_ratio_
    
    # Find number of components explaining 95% variance
    cumsum_variance = np.cumsum(explained_variance)
    n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
    
    # Effective dimensionality (intrinsic dimension)
    effective_dim = np.sum(explained_variance > 1e-6)
    
    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'explained_variance': explained_variance,
        'n_components_95': n_components_95,
        'effective_dim': effective_dim,
        'pca_components': pca.components_
    }

def compare_representational_spaces(bert_embeddings, clip_embeddings, story_name, layer_idx):
    """Compare representational spaces between BERT and CLIP"""
    print(f"Comparing representational spaces for {story_name}, layer {layer_idx}")
    
    # Extract embeddings
    bert_emb = extract_layer_embeddings(bert_embeddings, layer_idx, story_name)
    clip_emb = extract_layer_embeddings(clip_embeddings, layer_idx, story_name)
    
    # Ensure same number of samples
    min_samples = min(len(bert_emb), len(clip_emb))
    bert_emb = bert_emb[:min_samples]
    clip_emb = clip_emb[:min_samples]
    
    # Analyze individual structures
    bert_analysis = analyze_representational_structure(bert_emb, "BERT", layer_idx)
    clip_analysis = analyze_representational_structure(clip_emb, "CLIP", layer_idx)
    
    # Compute similarity
    similarity_matrix = compute_semantic_similarity(bert_emb, clip_emb)
    mean_similarity = np.mean(similarity_matrix)
    
    # Cross-correlation analysis
    # Flatten embeddings and compute correlation
    bert_flat = bert_emb.flatten()
    clip_flat = clip_emb.flatten()
    correlation, _ = pearsonr(bert_flat, clip_flat)
    
    return {
        'bert_analysis': bert_analysis,
        'clip_analysis': clip_analysis,
        'similarity_matrix': similarity_matrix,
        'mean_similarity': mean_similarity,
        'correlation': correlation
    }

def create_representational_visualizations(bert_embeddings, clip_embeddings, story_name="alternateithicatom"):
    """Create comprehensive visualizations of representational spaces"""
    print(f"Creating representational visualizations for {story_name}")
    
    # Analyze multiple layers
    layers_to_analyze = [0, 3, 6, 9, 11]  # Early, middle, late layers
    results = {}
    
    for layer_idx in layers_to_analyze:
        results[layer_idx] = compare_representational_spaces(
            bert_embeddings, clip_embeddings, story_name, layer_idx
        )
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Representational Analysis: BERT vs CLIP ({story_name})', fontsize=16)
    
    # 1. Layer-wise similarity comparison
    ax1 = axes[0, 0]
    layers = list(results.keys())
    similarities = [results[layer]['mean_similarity'] for layer in layers]
    correlations = [results[layer]['correlation'] for layer in layers]
    
    ax1.plot(layers, similarities, 'b-o', label='Cosine Similarity', linewidth=2)
    ax1.plot(layers, correlations, 'r-s', label='Correlation', linewidth=2)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Similarity Score')
    ax1.set_title('Layer-wise Representational Similarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dimensionality comparison
    ax2 = axes[0, 1]
    bert_dims = [results[layer]['bert_analysis']['effective_dim'] for layer in layers]
    clip_dims = [results[layer]['clip_analysis']['effective_dim'] for layer in layers]
    
    ax2.plot(layers, bert_dims, 'b-o', label='BERT', linewidth=2)
    ax2.plot(layers, clip_dims, 'r-s', label='CLIP', linewidth=2)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Effective Dimensionality')
    ax2.set_title('Representational Dimensionality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. PCA variance explained
    ax3 = axes[0, 2]
    bert_pca_95 = [results[layer]['bert_analysis']['n_components_95'] for layer in layers]
    clip_pca_95 = [results[layer]['clip_analysis']['n_components_95'] for layer in layers]
    
    ax3.plot(layers, bert_pca_95, 'b-o', label='BERT', linewidth=2)
    ax3.plot(layers, clip_pca_95, 'r-s', label='CLIP', linewidth=2)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Components for 95% Variance')
    ax3.set_title('PCA Complexity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Similarity heatmap for middle layer
    ax4 = axes[1, 0]
    middle_layer = 6
    similarity_matrix = results[middle_layer]['similarity_matrix']
    im = ax4.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    ax4.set_title(f'Similarity Matrix (Layer {middle_layer})')
    ax4.set_xlabel('CLIP Embeddings')
    ax4.set_ylabel('BERT Embeddings')
    plt.colorbar(im, ax=ax4)
    
    # 5. PCA visualization for middle layer
    ax5 = axes[1, 1]
    bert_emb = extract_layer_embeddings(bert_embeddings, middle_layer, story_name)
    clip_emb = extract_layer_embeddings(clip_embeddings, middle_layer, story_name)
    
    # Sample for visualization
    n_samples = min(1000, len(bert_emb), len(clip_emb))
    bert_sample = bert_emb[:n_samples]
    clip_sample = clip_emb[:n_samples]
    
    # PCA to 2D
    pca = PCA(n_components=2)
    bert_2d = pca.fit_transform(bert_sample)
    clip_2d = pca.fit_transform(clip_sample)
    
    ax5.scatter(bert_2d[:, 0], bert_2d[:, 1], alpha=0.6, label='BERT', s=20)
    ax5.scatter(clip_2d[:, 0], clip_2d[:, 1], alpha=0.6, label='CLIP', s=20)
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_title(f'PCA Visualization (Layer {middle_layer})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. t-SNE visualization
    ax6 = axes[1, 2]
    # Combine embeddings for t-SNE
    combined_emb = np.vstack([bert_sample, clip_sample])
    labels = ['BERT'] * len(bert_sample) + ['CLIP'] * len(clip_sample)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    combined_2d = tsne.fit_transform(combined_emb)
    
    bert_tsne = combined_2d[:len(bert_sample)]
    clip_tsne = combined_2d[len(bert_sample):]
    
    ax6.scatter(bert_tsne[:, 0], bert_tsne[:, 1], alpha=0.6, label='BERT', s=20)
    ax6.scatter(clip_tsne[:, 0], clip_tsne[:, 1], alpha=0.6, label='CLIP', s=20)
    ax6.set_xlabel('t-SNE 1')
    ax6.set_ylabel('t-SNE 2')
    ax6.set_title(f't-SNE Visualization (Layer {middle_layer})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Variance explained by top components
    ax7 = axes[2, 0]
    bert_var = results[middle_layer]['bert_analysis']['explained_variance'][:20]
    clip_var = results[middle_layer]['clip_analysis']['explained_variance'][:20]
    
    ax7.plot(range(1, 21), bert_var, 'b-o', label='BERT', linewidth=2)
    ax7.plot(range(1, 21), clip_var, 'r-s', label='CLIP', linewidth=2)
    ax7.set_xlabel('Principal Component')
    ax7.set_ylabel('Explained Variance Ratio')
    ax7.set_title('Variance Explained by Components')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Cross-modal alignment analysis
    ax8 = axes[2, 1]
    # Analyze how well representations align across models
    alignment_scores = []
    for layer in layers:
        similarity = results[layer]['mean_similarity']
        correlation = results[layer]['correlation']
        alignment = (similarity + abs(correlation)) / 2  # Combined alignment score
        alignment_scores.append(alignment)
    
    ax8.bar(layers, alignment_scores, alpha=0.7, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax8.set_xlabel('Layer')
    ax8.set_ylabel('Cross-Modal Alignment Score')
    ax8.set_title('Cross-Modal Alignment by Layer')
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    ax9 = axes[2, 2]
    ax9.axis('off')
    
    # Create summary text
    summary_text = f"""
    REPRESENTATIONAL ANALYSIS SUMMARY
    
    Story: {story_name}
    Layers Analyzed: {len(layers)}
    
    Key Findings:
    • BERT-CLIP Similarity: {np.mean(similarities):.3f} ± {np.std(similarities):.3f}
    • Best Alignment Layer: {layers[np.argmax(alignment_scores)]}
    • Dimensionality Difference: {np.mean(bert_dims) - np.mean(clip_dims):.1f}
    • PCA Complexity Ratio: {np.mean(bert_pca_95) / np.mean(clip_pca_95):.2f}
    
    Interpretation:
    • Higher similarity = More aligned representations
    • Lower dimensionality = More compressed representations
    • Higher PCA complexity = More diverse representations
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('representational_spaces.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def create_semantic_heatmaps(bert_embeddings, clip_embeddings, story_name="alternateithicatom"):
    """Create semantic similarity heatmaps"""
    print(f"Creating semantic heatmaps for {story_name}")
    
    # Analyze multiple layers
    layers = [0, 3, 6, 9, 11]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Semantic Similarity Analysis: BERT vs CLIP ({story_name})', fontsize=16)
    
    for i, layer_idx in enumerate(layers):
        if i >= 5:  # Only show 5 subplots
            break
            
        ax = axes[i//3, i%3]
        
        # Extract embeddings
        bert_emb = extract_layer_embeddings(bert_embeddings, layer_idx, story_name)
        clip_emb = extract_layer_embeddings(clip_embeddings, layer_idx, story_name)
        
        # Sample for visualization
        n_samples = min(200, len(bert_emb), len(clip_emb))
        bert_sample = bert_emb[:n_samples]
        clip_sample = clip_emb[:n_samples]
        
        # Compute similarity matrix
        similarity_matrix = compute_semantic_similarity(bert_sample, clip_sample)
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {layer_idx} Similarity Matrix')
        ax.set_xlabel('CLIP Embeddings')
        ax.set_ylabel('BERT Embeddings')
        plt.colorbar(im, ax=ax)
        
        # Add statistics
        mean_sim = np.mean(similarity_matrix)
        std_sim = np.std(similarity_matrix)
        ax.text(0.02, 0.98, f'Mean: {mean_sim:.3f}\nStd: {std_sim:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Summary plot
    ax_summary = axes[1, 2]
    ax_summary.axis('off')
    
    # Calculate summary statistics
    layer_similarities = []
    for layer_idx in layers:
        bert_emb = extract_layer_embeddings(bert_embeddings, layer_idx, story_name)
        clip_emb = extract_layer_embeddings(clip_embeddings, layer_idx, story_name)
        n_samples = min(200, len(bert_emb), len(clip_emb))
        bert_sample = bert_emb[:n_samples]
        clip_sample = clip_emb[:n_samples]
        similarity_matrix = compute_semantic_similarity(bert_sample, clip_sample)
        layer_similarities.append(np.mean(similarity_matrix))
    
    ax_summary.plot(layers, layer_similarities, 'b-o', linewidth=2, markersize=8)
    ax_summary.set_xlabel('Layer')
    ax_summary.set_ylabel('Mean Similarity')
    ax_summary.set_title('Layer-wise Semantic Similarity')
    ax_summary.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('semantic_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    print("=== Representational Analysis Framework ===")
    print()
    
    # Load embeddings
    bert_file = "Stimuli/bert-subset-moth-radio_bert-base_20.npy"
    clip_file = "Stimuli/clip-subset-moth-radio_clip-text_20.npy"
    
    bert_embeddings, clip_embeddings = load_embeddings(bert_file, clip_file)
    
    # Create representational visualizations
    print("\n1. Creating representational space analysis...")
    results = create_representational_visualizations(bert_embeddings, clip_embeddings)
    
    # Create semantic heatmaps
    print("\n2. Creating semantic similarity analysis...")
    create_semantic_heatmaps(bert_embeddings, clip_embeddings)
    
    print("\n✓ Representational analysis completed!")
    print("Files created:")
    print("- representational_spaces.png")
    print("- semantic_heatmaps.png")
    
    return results

if __name__ == "__main__":
    results = main()
