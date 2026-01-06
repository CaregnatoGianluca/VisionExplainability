import numpy as np
from PIL import Image
import torch
import warnings
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, chisquare
import cv2



# Function to calculate distributions from heatmaps (with resize if necessary)
def saliency_to_distribution(saliency_map, target_size=None, eps=1e-12):
    '''
    Convert a saliency map (PIL Image, numpy array, or torch tensor) to a probability distribution.
    Optionally resize the saliency map to a target size before conversion.

    Args:
        saliency_map (PIL.Image, np.ndarray, or torch.Tensor): Input saliency map.
        target_size (tuple, optional): Target size (H, W) to resize the saliency map.
        eps (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Probability distribution of the saliency map.
    '''

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

    if len(predicted_heatmap.shape) == 3 and predicted_heatmap.shape[2] == 3:
        #convert predicted heatmap to grayscale one channel
        predicted_heatmap = Image.fromarray(predicted_heatmap).convert("L")

    if len(ground_truth_heatmap.shape) == 3 and ground_truth_heatmap.shape[2] == 3:
        #convert predicted heatmap to grayscale one channel
        ground_truth_heatmap = Image.fromarray(ground_truth_heatmap).convert("L")

    Q = saliency_to_distribution(ground_truth_heatmap, target_size=gt_target_size) 
    P = saliency_to_distribution(predicted_heatmap, target_size=gt_target_size)

    eps = 1e-10

    if(np.any(P + Q == 0)):
        warnings.warn("Zero division encountered in Chi2 calculation. Adjusting values to avoid division by zero.")
        P = P + eps
        Q = Q + eps

    chi2_distance = 1 - np.sum(((P - Q) ** 2) / (P + Q))

    return {"JSS": 1 - jensenshannon(P, Q), "Chi2": chi2_distance, "PCC": pearsonr(P, Q)[0]}
            