import cv2
import numpy as np

def compute_internal_metrics(image: np.ndarray, label_map: np.ndarray) -> dict:
    """
    Compute Internal clustering metrics: F, F', and Q.
    These metrics evaluate segmentation quality without ground truth.
    
    Args:
        image: RGB image array of shape (M, N, 3).
        label_map: Integer array of shape (M, N) with cluster labels.
        
    Returns:
        Dictionary containing F, F', and Q metric values.
    """
    M, N = label_map.shape
    
    # Find all connected regions for each cluster label
    unique_labels = np.unique(label_map)
    regions = []
    
    for label in unique_labels:
        # Create binary mask for the current cluster
        mask = (label_map == label).astype(np.uint8)
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Skip background component (i=0) of the connectedComponents output
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area == 0:
                continue
                
            region_mask = (labels_im == i)
            region_pixels = image[region_mask]
            
            # e_i^2 is the sum of squared color errors (Euclidean distance to mean color)
            mean_color = np.mean(region_pixels, axis=0)
            e_i_2 = np.sum(np.linalg.norm(region_pixels - mean_color, axis=1) ** 2)
            
            regions.append({
                'area': area,
                'e_i_2': e_i_2
            })
            
    R = len(regions)
    if R == 0:
        return {'F': 0.0, "F'": 0.0, 'Q': 0.0}
        
    # Calculate R(A): number of regions of area A
    area_counts = {}
    for r in regions:
        a = r['area']
        area_counts[a] = area_counts.get(a, 0) + 1
        
    # Calculate the sum for F and F'
    sum_e_div_sqrt_A = sum(r['e_i_2'] / np.sqrt(r['area']) for r in regions)
    
    # Metric F
    F = (1.0 / (1000.0 * M * N)) * np.sqrt(R) * sum_e_div_sqrt_A
    
    # Metric F'
    # R(A)^(1 + 1/A)
    sum_R_A_term = sum((count ** (1.0 + 1.0 / A)) for A, count in area_counts.items())
    F_prime = (1.0 / (10000.0 * M * N)) * np.sqrt(sum_R_A_term) * sum_e_div_sqrt_A
    
    # Metric Q
    sum_Q = 0.0
    for r in regions:
        A = r['area']
        e_i_2 = r['e_i_2']
        R_A = area_counts[A]
        sum_Q += (e_i_2 / (1.0 + np.log(A))) + ((R_A / A) ** 2)
        
    Q = (1.0 / (10000.0 * M * N)) * np.sqrt(R) * sum_Q
    
    return {
        'F': F,
        "F'": F_prime,
        'Q': Q
    }


def compute_external_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    """
    Compute Externals clustering metrics: Accuracy, Precision, Recall, F1 Score, Specificity.
    Requires ground truth masks.
    
    Args:
        pred_mask: Boolean or binary array representing predicted tumor mask.
        true_mask: Boolean or binary array representing ground truth tumor mask.
        
    Returns:
        Dictionary containing Accuracy, Precision, Recall, F1 Score, and Specificity.
    """
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    
    TP = np.sum(pred & true)
    TN = np.sum(~pred & ~true)
    FP = np.sum(pred & ~true)
    FN = np.sum(~pred & true)
    
    accuracy = (TN + TP) / (TN + TP + FN + FP) if (TN + TP + FN + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'Specificity': specificity
    }
