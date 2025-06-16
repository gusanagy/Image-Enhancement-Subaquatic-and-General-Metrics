""" Adaptado de https://github.com/zzc-1998/NLIEE"""
import torch
import torch.nn.functional as F
import numpy as np

def nliee_score(low_light_img, enhanced_img):
    """
    Calculate NLIEE score for low-light image enhancement evaluation
    Compatible with PyTorch tensors and numpy arrays
    
    Args:
        low_light_img: Original low-light image 
                      - PyTorch tensor: (B, C, H, W) or (C, H, W) 
                      - Numpy array: (H, W, C) or (H, W)
                      - Values in range [0, 1] or [0, 255]
        enhanced_img: Enhanced image (same format as low_light_img)
    
    Returns:
        float or tensor: Quality score (0-100, higher is better)
                        Returns tensor if input is tensor, float if numpy
    
    Example:
        # PyTorch tensors
        score = nliee_score(low_tensor, enhanced_tensor)
        
        # Numpy arrays  
        score = nliee_score(low_np, enhanced_np)
    """
    
    # Handle PyTorch tensors
    is_tensor = torch.is_tensor(low_light_img)
    device = None
    
    if is_tensor:
        device = low_light_img.device
        # Convert to numpy for processing
        low_np = low_light_img.detach().cpu().numpy()
        enh_np = enhanced_img.detach().cpu().numpy()
        
        # Handle batch dimension
        if low_np.ndim == 4:  # (B, C, H, W)
            # Process batch and return tensor of scores
            scores = []
            for i in range(low_np.shape[0]):
                low_single = np.transpose(low_np[i], (1, 2, 0))  # (H, W, C)
                enh_single = np.transpose(enh_np[i], (1, 2, 0))  # (H, W, C)
                score = _compute_nliee_score(low_single, enh_single)
                scores.append(score)
            return torch.tensor(scores, device=device)
        
        elif low_np.ndim == 3:  # (C, H, W)
            low_np = np.transpose(low_np, (1, 2, 0))  # (H, W, C)
            enh_np = np.transpose(enh_np, (1, 2, 0))  # (H, W, C)
    
    else:
        # Handle numpy arrays directly
        low_np = low_light_img
        enh_np = enhanced_img
    
    score = _compute_nliee_score(low_np, enh_np)
    
    return torch.tensor(score, device=device) if is_tensor else score


def _compute_nliee_score(low_light_img, enhanced_img):
    """Internal function to compute NLIEE score"""
    
    # Normalize to [0, 1] range
    if low_light_img.max() > 1.0:
        low_light_img = low_light_img.astype(np.float32) / 255.0
    if enhanced_img.max() > 1.0:
        enhanced_img = enhanced_img.astype(np.float32) / 255.0
    
    # Convert to grayscale if needed
    if len(low_light_img.shape) == 3:
        gray_low = np.mean(low_light_img, axis=2)
        gray_enh = np.mean(enhanced_img, axis=2)
    else:
        gray_low = low_light_img.astype(np.float32)
        gray_enh = enhanced_img.astype(np.float32)
    
    # 1. Brightness Assessment (25% weight)
    brightness_low = np.mean(gray_low)
    brightness_enh = np.mean(gray_enh)
    brightness_improvement = brightness_enh - brightness_low
    
    # Optimal improvement should be 0.2-0.4
    brightness_score = max(0.0, 1.0 - abs(brightness_improvement - 0.3) / 0.5)
    
    # 2. Contrast Assessment (30% weight)
    contrast_low = np.std(gray_low)
    contrast_enh = np.std(gray_enh)
    
    if contrast_low > 1e-6:
        contrast_ratio = contrast_enh / contrast_low
        # Good enhancement: 1.2x to 3x contrast improvement
        if contrast_ratio >= 1.2:
            contrast_score = min(1.0, (contrast_ratio - 1.0) / 2.0)
        else:
            contrast_score = max(0.0, (contrast_ratio - 0.8) / 0.4)
    else:
        contrast_score = 0.5
    
    # 3. Detail Preservation (25% weight)
    # Use gradient magnitude as detail measure
    gy_low, gx_low = np.gradient(gray_low)
    gy_enh, gx_enh = np.gradient(gray_enh)
    
    grad_mag_low = np.sqrt(gx_low**2 + gy_low**2)
    grad_mag_enh = np.sqrt(gx_enh**2 + gy_enh**2)
    
    detail_low = np.mean(grad_mag_low)
    detail_enh = np.mean(grad_mag_enh)
    
    if detail_low > 1e-6:
        detail_ratio = detail_enh / detail_low
        detail_score = min(1.0, detail_ratio)  # Should preserve or enhance details
    else:
        detail_score = 1.0 if detail_enh > 1e-6 else 0.5
    
    # 4. Naturalness Assessment (20% weight)
    # Check for over-enhancement
    
    # Histogram spread - good enhancement should spread the histogram
    hist_low, _ = np.histogram(gray_low.flatten(), bins=64, range=(0, 1))
    hist_enh, _ = np.histogram(gray_enh.flatten(), bins=64, range=(0, 1))
    
    # Normalize histograms
    hist_low = hist_low / (np.sum(hist_low) + 1e-10)
    hist_enh = hist_enh / (np.sum(hist_enh) + 1e-10)
    
    # Calculate entropy (higher is better for enhancement)
    entropy_low = -np.sum(hist_low * np.log(hist_low + 1e-10))
    entropy_enh = -np.sum(hist_enh * np.log(hist_enh + 1e-10))
    
    max_entropy = np.log(64)
    entropy_improvement = (entropy_enh - entropy_low) / max_entropy
    naturalness_score = max(0.0, min(1.0, 0.5 + entropy_improvement))
    
    # Avoid over-saturation penalty
    if np.mean(gray_enh) > 0.9:  # Too bright
        naturalness_score *= 0.7
    
    # Combine all scores
    final_score = (
        0.25 * brightness_score +
        0.30 * contrast_score + 
        0.25 * detail_score +
        0.20 * naturalness_score
    )
    
    return final_score * 100


# Example usage for PyTorch training
if __name__ == "__main__":
    # Test with numpy arrays
    low_np = np.random.rand(64, 64, 3) * 0.3
    enh_np = np.clip(low_np * 2.5, 0, 1)
    score_np = nliee_score(low_np, enh_np)
    print(f"Numpy score: {score_np:.2f}")
    
    # Test with PyTorch tensors
    low_tensor = torch.rand(2, 3, 64, 64) * 0.3  # Batch of 2
    enh_tensor = torch.clamp(low_tensor * 2.5, 0, 1)
    scores_tensor = nliee_score(low_tensor, enh_tensor)
    print(f"PyTorch batch scores: {scores_tensor}")
    
    # Single tensor
    low_single = torch.rand(3, 64, 64) * 0.3
    enh_single = torch.clamp(low_single * 2.5, 0, 1)
    score_single = nliee_score(low_single, enh_single)
    print(f"Single tensor score: {score_single:.2f}")