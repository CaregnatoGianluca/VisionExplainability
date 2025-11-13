import numpy as np
from PIL import Image
import torch
import warnings
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, chisquare

# Function to calculate distributions from heatmaps (with resize if necessary)
def saliency_to_distribution(saliency_map, target_size=None, eps=1e-12):
  
    # Convert to np.array
    if isinstance(saliency_map, Image.Image):
        # PIL image
        array = np.array(saliency_map, dtype=np.float64)
    elif isinstance(saliency_map, torch.Tensor):
        # PyTorch tensor
        array = saliency_map.detach().cpu().numpy().astype(np.float64)
    elif isinstance(saliency_map, np.ndarray):
        array = saliency_map.astype(np.float64)
    else:
        raise TypeError("The input must be a PIL Image, np.ndarray, or torch.Tensor")
    
    # We make sure it's 2D
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Input map must be 2D or convertible to 2D. Input shape: {saliency_map.shape}")

    # Resize
    if target_size is not None:
        # target_size is (H, W), PIL.resize needs (W, H)
        pil_target_size = (target_size[1], target_size[0]) 
        if array.shape != target_size:
            # Conversion to PIL image for a more consistent resize
            pil_img = Image.fromarray(array)
            # Bilinear interpolation 
            pil_img = pil_img.resize(pil_target_size, Image.BILINEAR) 
            array = np.array(pil_img, dtype=np.float64)
    
    # Computing distribution
    # No negative values
    array = np.clip(array, a_min=0, a_max=None)
    
    dist = (array + eps).ravel()
    
    # Normalization
    dist_sum = dist.sum()
    if dist_sum < 1e-9: 
        warnings.warn("Saliency map near 0. Returning a uniform distribution.")
        dist = np.ones_like(dist) / len(dist) 
    else:
        dist = dist / dist_sum
            
    return dist


def calc_jss_chi2_pcc_scores(predicted_heatmap: np.ndarray, 
                             ground_truth_heatmap: np.ndarray):
    """Calculate JSS, Chi2, PCC scores between two heatmaps.
    
    Args:
        predicted_heatmap (PIL.Image): Predicted heatmap.
        ground_truth_heatmap (PIL.Image): Ground truth heatmap.
        target_size (tuple, optional): Target size (H, W) to resize heatmaps before calculation.
    
    Returns:
        dict: Dictionary containing JSS, Chi2, and PCC scores.
    """
    gt_target_size = ground_truth_heatmap.shape  # (H, W)

    Q = saliency_to_distribution(ground_truth_heatmap, target_size=None) 
    P = saliency_to_distribution(predicted_heatmap, target_size=gt_target_size)

    return {"JSS": 1 - jensenshannon(P, Q), "Chi2": chisquare(P, Q), "PCC": pearsonr(P, Q)[0]}
            