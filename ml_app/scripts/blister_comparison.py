import numpy as np
from scipy.stats import chisquare

def calculate_size(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def categorize_size(size):
    if size < 100:
        return 'small'
    elif size < 500:
        return 'medium'
    else:
        return 'large'

def generate_histogram(detections):
    sizes = [calculate_size(bbox[:4]) for bbox in detections]
    categories = [categorize_size(size) for size in sizes]
    return {
        'small': categories.count('small'),
        'medium': categories.count('medium'),
        'large': categories.count('large')
    }

import numpy as np
from scipy.stats import chisquare

def compare_histograms(hist1, hist2):
    categories = ['small', 'medium', 'large']
    observed = np.array([hist1[cat] for cat in categories])
    expected = np.array([hist2[cat] for cat in categories])
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    observed = observed + epsilon
    expected = expected + epsilon
    
    # Normalize the arrays
    observed = observed / np.sum(observed)
    expected = expected / np.sum(expected)
    
    try:
        chi2, _ = chisquare(observed, expected)
        size_similarity = 1 / (1 + chi2)
    except ValueError:
        # If chisquare fails, fall back to a simple difference measure
        size_similarity = 1 - np.sum(np.abs(observed - expected)) / 2
    
    return size_similarity

def compare_images(detected_bboxes, standard_bboxes):
    detected_hist = generate_histogram(detected_bboxes)
    standard_hist = generate_histogram(standard_bboxes)
    
    print(f"Detected histogram: {detected_hist}")
    print(f"Standard histogram: {standard_hist}")
    
    total_detected = sum(detected_hist.values())
    total_standard = sum(standard_hist.values())
    
    size_similarity = compare_histograms(detected_hist, standard_hist)
    count_similarity = min(total_detected, total_standard) / max(total_detected, total_standard)
    
    print(f"Size similarity: {size_similarity}")
    print(f"Count similarity: {count_similarity}")
    
    # Combine size and count similarities, handling potential NaN
    if np.isnan(size_similarity):
        overall_similarity = count_similarity
    else:
        overall_similarity = (size_similarity + count_similarity) / 2
    
    return overall_similarity