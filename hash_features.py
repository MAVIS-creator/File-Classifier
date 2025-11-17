"""
Hash Feature Extractor Module
Generates MD5 hash-related metadata features from files
"""

import hashlib
import os
from typing import Dict, List


def calculate_md5(file_path: str) -> str:
    """
    Calculate MD5 hash of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash as hexadecimal string
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating MD5 for {file_path}: {e}")
        return ""


def extract_hash_features(file_path: str) -> Dict[str, float]:
    """
    Extract MD5 hash-related features from a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary of hash-based features
    """
    features = {}
    
    try:
        # Get MD5 hash
        md5_hash = calculate_md5(file_path)
        
        if not md5_hash:
            return features
        
        # Feature 1: File size
        features['file_size'] = os.path.getsize(file_path)
        
        # Feature 2-17: Convert first 16 bytes of MD5 hash to numeric features
        for i in range(min(16, len(md5_hash) // 2)):
            byte_val = int(md5_hash[i*2:i*2+2], 16)
            features[f'hash_byte_{i}'] = byte_val
        
        # Feature 18: Hash entropy (measure of randomness)
        hash_bytes = bytes.fromhex(md5_hash)
        byte_counts = [0] * 256
        for byte in hash_bytes:
            byte_counts[byte] += 1
        
        entropy = 0
        total = len(hash_bytes)
        for count in byte_counts:
            if count > 0:
                probability = count / total
                entropy -= probability * (probability if probability == 0 else 
                                         __import__('math').log2(probability))
        features['hash_entropy'] = entropy
        
        # Feature 19: Average byte value in hash
        features['hash_avg_byte'] = sum(hash_bytes) / len(hash_bytes)
        
        # Feature 20: Standard deviation of hash bytes
        avg = features['hash_avg_byte']
        variance = sum((b - avg) ** 2 for b in hash_bytes) / len(hash_bytes)
        features['hash_std'] = variance ** 0.5
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
    
    return features


def extract_features_batch(file_paths: List[str]) -> List[Dict[str, float]]:
    """
    Extract hash features from multiple files
    
    Args:
        file_paths: List of file paths
        
    Returns:
        List of feature dictionaries
    """
    return [extract_hash_features(fp) for fp in file_paths]
