import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import os

def plot_class_distribution(y, output_path=None):
    plt.figure(figsize=(10, 6))
    if isinstance(y, pd.Series):
        sns.countplot(x=y)
    else:
        sns.countplot(x=pd.Series(y))
    plt.title('Target Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_correlation_heatmap(df, output_path=None):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

def apply_and_plot_pca(X_scaled, y, output_path=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Target Class')
    plt.title('PCA - 2D Projection')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    if output_path:
        plt.savefig(output_path)
    plt.show()
    return X_pca, pca

def plot_model_comparison(results_dict, output_path=None):
    models = list(results_dict.keys())
    accuracies = [metrics['accuracy'] for metrics in results_dict.values()]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=accuracies, palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()