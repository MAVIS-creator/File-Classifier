"""
File Type Classifier Using Hash Features
Main module for training and evaluating Naive Bayes classifier
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from data_handler import prepare_data
import argparse


def train_classifier(X_train, y_train):
    """
    Train Naive Bayes classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained classifier model
    """
    print("Training Naive Bayes Classifier...")
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    print("Training completed!\n")
    return classifier


def evaluate_classifier(classifier, X_test, y_test):
    """
    Evaluate classifier performance
    
    Args:
        classifier: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating classifier...")
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_test': y_test
    }
    
    return metrics


def generate_accuracy_report(metrics, X_train, X_test):
    """
    Generate comprehensive accuracy report
    
    Args:
        metrics: Dictionary of evaluation metrics
        X_train: Training features
        X_test: Test features
    """
    print("\n" + "="*70)
    print("FILE TYPE CLASSIFIER - ACCURACY REPORT")
    print("Using MD5 Hash-Related Metadata Features")
    print("Algorithm: Naive Bayes Classifier")
    print("="*70)
    
    print(f"\nDataset Information:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    print(f"  - Number of features: {X_train.shape[1]}")
    
    print(f"\n{'-'*70}")
    print("PERFORMANCE METRICS")
    print(f"{'-'*70}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    print(f"\n{'-'*70}")
    print("CONFUSION MATRIX")
    print(f"{'-'*70}")
    cm = metrics['confusion_matrix']
    print(f"                Predicted")
    print(f"              Benign  Malicious")
    print(f"Actual Benign    {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"       Malicious {cm[1][0]:3d}      {cm[1][1]:3d}")
    
    print(f"\n{'-'*70}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'-'*70}")
    print(classification_report(
        metrics['y_test'], 
        metrics['y_pred'],
        target_names=['Benign', 'Malicious']
    ))
    
    print("="*70)
    
    # Visualize confusion matrix
    visualize_results(metrics)


def visualize_results(metrics):
    """
    Create visualizations of classification results
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix Heatmap
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_xlabel('Predicted', fontsize=12)
    
    # Performance Metrics Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = axes[1].bar(metric_names, metric_values, color=colors, alpha=0.7)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = 'classification_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{output_path}'")
    # Only show plots if explicitly requested to avoid blocking CI/headless runs
    if os.environ.get("SHOW_PLOTS", "0").lower() in {"1", "true", "yes"}:
        plt.show()
    else:
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="File Type Classifier Using Hash Features")
    parser.add_argument("--benign-dir", type=str, default=None, help="Path to directory of benign files")
    parser.add_argument("--malicious-dir", type=str, default=None, help="Path to directory of malicious files")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split proportion (default 0.3)")
    parser.add_argument("--no-generate", action="store_true", help="Do not auto-generate synthetic samples if dirs missing")
    parser.add_argument("--scan", nargs="*", help="One or more file paths to classify after training")
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively")
    parser.add_argument("--report", type=str, default=None, help="Optional path to save textual report")
    return parser.parse_args()


def classify_files(file_paths, model):
    from hash_features import extract_hash_features
    rows = []
    for fp in file_paths:
        if not os.path.isfile(fp):
            print(f"[WARN] Skipping missing file: {fp}")
            continue
        feats = extract_hash_features(fp)
        if not feats:
            print(f"[WARN] No features extracted: {fp}")
            continue
        import pandas as pd
        df = pd.DataFrame([feats]).fillna(0)
        pred = model.predict(df)[0]
        label = "Malicious" if pred == 1 else "Benign"
        print(f"[SCAN] {fp} -> {label}")
        rows.append({"file": fp, "prediction": label})
    return rows


def main():
    args = parse_args()

    if args.show_plots:
        os.environ["SHOW_PLOTS"] = "1"

    print("\n" + "="*70)
    print("FILE TYPE CLASSIFIER USING HASH FEATURES")
    print("="*70 + "\n")

    print("Preparing dataset...")
    X_train, X_test, y_train, y_test = prepare_data(
        test_size=args.test_size,
        benign_dir=args.benign_dir,
        malicious_dir=args.malicious_dir,
        generate_if_missing=not args.no_generate
    )
    print(f"✓ Dataset prepared: {len(X_train)} training, {len(X_test)} testing samples\n")

    classifier = train_classifier(X_train, y_train)
    metrics = evaluate_classifier(classifier, X_test, y_test)
    generate_accuracy_report(metrics, X_train, X_test)

    if args.scan:
        print("\nInitiating scan of user-provided file(s)...")
        scan_results = classify_files(args.scan, classifier)
        if scan_results:
            print("\nScan Summary:")
            for r in scan_results:
                print(f"  {r['file']}: {r['prediction']}")
    
    if args.report:
        try:
            with open(args.report, "w", encoding="utf-8") as f:
                f.write("FILE TYPE CLASSIFIER REPORT\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1: {metrics['f1_score']:.4f}\n")
            print(f"\n✓ Text report saved to {args.report}")
        except Exception as e:
            print(f"Failed to save report: {e}")

    print("\n✓ Classification run complete.")


if __name__ == "__main__":
    main()
