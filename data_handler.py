"""
Data Handler Module
Manages file labeling and dataset preparation
"""

import os
import pandas as pd
from typing import List, Tuple
from hash_features import extract_hash_features


def create_sample_files(sample_dir: str = "sample_files"):
    """
    Create sample benign and malicious file directories for demonstration
    
    Args:
        sample_dir: Base directory for sample files
    """
    benign_dir = os.path.join(sample_dir, "benign")
    malicious_dir = os.path.join(sample_dir, "malicious")
    
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malicious_dir, exist_ok=True)
    
    # Create sample benign files (text, documents)
    benign_samples = [
        ("document1.txt", b"This is a normal text document with regular content."),
        ("document2.txt", b"Another benign file with legitimate data and information."),
        ("config.txt", b"Configuration file with settings: port=8080, host=localhost"),
        ("readme.txt", b"README file explaining project details and usage instructions."),
        ("data.txt", b"Sample data file containing CSV-like information\nname,age,city"),
        ("log.txt", b"Application log file with timestamps and normal events."),
        ("notes.txt", b"Personal notes and reminders for daily tasks."),
        ("report.txt", b"Monthly report with statistics and analysis data."),
    ]
    
    # Create sample malicious files (simulated with suspicious patterns)
    malicious_samples = [
        ("malware1.exe", b"\x4D\x5A" + b"\x90" * 100 + b"malicious payload here"),
        ("virus.dll", b"\x4D\x5A\x50\x45" + b"\xFF" * 150 + b"infected code"),
        ("trojan.bin", b"\xDE\xAD\xBE\xEF" * 50 + b"unauthorized access code"),
        ("backdoor.exe", b"\x4D\x5A" + b"\xCC" * 120 + b"remote shell code"),
        ("keylogger.dll", b"\x50\x45\x00\x00" + b"\xAA" * 100 + b"keystroke capture"),
        ("ransomware.bin", b"\xFF\xFE" * 75 + b"file encryption routine"),
        ("rootkit.sys", b"\x4B\x45\x52\x4E" + b"\x00" * 130 + b"kernel manipulation"),
        ("spyware.exe", b"\x4D\x5A\x90\x00" + b"\xBB" * 110 + b"data exfiltration"),
    ]
    
    # Write benign files
    for filename, content in benign_samples:
        with open(os.path.join(benign_dir, filename), "wb") as f:
            f.write(content)
    
    # Write malicious files
    for filename, content in malicious_samples:
        with open(os.path.join(malicious_dir, filename), "wb") as f:
            f.write(content)
    
    return benign_dir, malicious_dir


def load_dataset(benign_dir: str, malicious_dir: str) -> Tuple[pd.DataFrame, List[int]]:
    """
    Load files from directories and create labeled dataset
    
    Args:
        benign_dir: Directory containing benign files
        malicious_dir: Directory containing malicious files
        
    Returns:
        Tuple of (features DataFrame, labels list)
    """
    file_features = []
    labels = []
    
    # Process benign files (label = 0)
    if os.path.exists(benign_dir):
        for filename in os.listdir(benign_dir):
            file_path = os.path.join(benign_dir, filename)
            if os.path.isfile(file_path):
                features = extract_hash_features(file_path)
                if features:
                    file_features.append(features)
                    labels.append(0)  # 0 = benign
    
    # Process malicious files (label = 1)
    if os.path.exists(malicious_dir):
        for filename in os.listdir(malicious_dir):
            file_path = os.path.join(malicious_dir, filename)
            if os.path.isfile(file_path):
                features = extract_hash_features(file_path)
                if features:
                    file_features.append(features)
                    labels.append(1)  # 1 = malicious
    
    # Convert to DataFrame
    df = pd.DataFrame(file_features)
    
    # Fill any missing values with 0
    df = df.fillna(0)
    
    return df, labels


def prepare_data(test_size: float = 0.3, random_state: int = 42):
    """
    Prepare complete dataset with train/test split
    
    Args:
        test_size: Proportion of dataset for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Create sample files
    benign_dir, malicious_dir = create_sample_files()
    
    # Load dataset
    X, y = load_dataset(benign_dir, malicious_dir)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
