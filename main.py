import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
from pathlib import Path
import time
from datetime import datetime

def load_sam_model(use_fast_config=True):
    """Load SAM model for segmentation with performance optimization"""
    print("Loading SAM model...")
    
    # Download SAM checkpoint if not exists
    model_path = "models/sam_vit_h_4b8939.pth"
    if not os.path.exists(model_path):
        print(f"SAM model not found at {model_path}")
        print("Please download SAM model checkpoint:")
        print("python download_sam.py")
        return None
    
    # Load SAM model
    sam = sam_model_registry["vit_h"](checkpoint=model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    
    # Create mask generator with optimized settings
    if use_fast_config:
        # Faster configuration for development/testing
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,  # Reduced from default 32
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,  # Disabled cropping for speed
            min_mask_region_area=500,  # Larger minimum area
        )
        print(f"SAM model loaded (FAST mode) on {device}")
    else:
        # High quality configuration
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        print(f"SAM model loaded (HIGH QUALITY mode) on {device}")
    
    return mask_generator

def create_product_mask(product_image):
    """Create alpha mask for products without transparency - preserves original product appearance"""
    if len(product_image.shape) == 3 and product_image.shape[2] == 4:
        # Already has alpha channel
        return product_image[:, :, 3] / 255.0
    
    # For products without alpha channel, create a smart mask
    # This preserves the original product but removes obvious background
    
    gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Method 1: Edge detection to find product boundaries
    edges = cv2.Canny(gray, 30, 100)
    
    # Method 2: Adaptive thresholding to separate foreground/background
    # Use multiple threshold values and combine
    masks = []
    
    # Try different threshold values
    for thresh in [200, 220, 240]:
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        masks.append(mask)
    
    # Combine masks - use intersection for conservative approach
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_and(combined_mask, mask)
    
    # Include edges to ensure product boundaries are preserved
    combined_mask = cv2.bitwise_or(combined_mask, edges)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes in the mask to ensure complete product coverage
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (should be the main product)
        largest_contour = max(contours, key=cv2.contourArea)
        mask_filled = np.zeros_like(combined_mask)
        cv2.fillPoly(mask_filled, [largest_contour], 255)
        
        # Only use filled mask if it's reasonable size (not too small or too large)
        contour_area = cv2.contourArea(largest_contour)
        image_area = h * w
        area_ratio = contour_area / image_area
        
        if 0.1 < area_ratio < 0.9:  # Product should be 10-90% of image
            combined_mask = mask_filled
    
    # Smooth the mask edges for better blending
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    
    # Convert to float and ensure reasonable transparency
    alpha = combined_mask / 255.0
    
    # Ensure we don't make the product completely transparent
    alpha = np.clip(alpha, 0.1, 1.0)  # Minimum 10% opacity
    
    return alpha

def segment_walls(image_path, mask_generator):
    """Segment walls from room image using SAM with timing"""
    print(f"Segmenting walls from {image_path}...")
    start_time = time.time()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB for SAM
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    masks = mask_generator.generate(image_rgb)
    
    end_time = time.time()
    print(f"Generated {len(masks)} masks in {end_time - start_time:.2f} seconds")
    
    # Find wall mask (largest mask in center area)
    if not masks:
        raise ValueError("No masks generated")
    
    # Filter for larger, more central masks (likely walls)
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    wall_candidates = []
    for mask_data in masks:
        mask = mask_data['segmentation']
        area = mask_data['area']
        
        # Calculate mask center of mass
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Distance from image center
            center_dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            # Score: larger area + closer to center = higher score
            score = area / (1 + center_dist * 0.001)
            
            wall_candidates.append({
                'mask': mask,
                'area': area,
                'center_dist': center_dist,
                'score': score
            })
    
    if not wall_candidates:
        # Fallback to largest mask
        largest_mask = max(masks, key=lambda x: x['area'])
        wall_mask = largest_mask['segmentation']
    else:
        # Get highest scoring mask (likely the main wall)
        best_wall = max(wall_candidates, key=lambda x: x['score'])
        wall_mask = best_wall['mask']
    
    return image, wall_mask, masks

def place_product_on_wall(image, wall_mask, product_path):
    """Enhanced product placement with better transparency handling"""
    print(f"Placing product from {product_path} on wall...")
    
    # Load product image
    product = cv2.imread(product_path, cv2.IMREAD_UNCHANGED)
    if product is None:
        raise ValueError(f"Could not load product image from {product_path}")
    
    # Create alpha mask for the product
    if len(product.shape) == 3 and product.shape[2] == 4:
        # Has alpha channel
        alpha = product[:, :, 3] / 255.0
        product_rgb = product[:, :, :3]
    else:
        # Create mask for non-transparent products
        alpha = create_product_mask(product)
        product_rgb = product
    
    # Find wall contours for placement area
    wall_mask_uint8 = wall_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(wall_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No wall contours found")
    
    # Get largest contour (main wall area)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Define target placement area with better sizing
    product_aspect_ratio = product.shape[1] / product.shape[0]
    wall_aspect_ratio = w / h
    
    # Adaptive scaling based on product type
    if product_aspect_ratio > 2.0:  # Wide products (like TVs)
        scale_factor = 0.25
    elif product_aspect_ratio < 0.8:  # Tall products
        scale_factor = 0.20
    else:  # Square-ish products (paintings)
        scale_factor = 0.30
    
    target_w = int(w * scale_factor)
    target_h = int(target_w / product_aspect_ratio)
    
    # Center the product
    target_x = x + (w - target_w) // 2
    target_y = y + (h - target_h) // 2
    
    # Resize product and alpha mask
    product_resized = cv2.resize(product_rgb, (target_w, target_h))
    alpha_resized = cv2.resize(alpha, (target_w, target_h))
    
    # Blend product onto image
    result = image.copy()
    
    # Blend the product directly without shadows
    for c in range(3):
        result[target_y:target_y+target_h, target_x:target_x+target_w, c] = (
            alpha_resized * product_resized[:, :, c] + 
            (1 - alpha_resized) * result[target_y:target_y+target_h, target_x:target_x+target_w, c]
        )
    
    return result

def visualize_results(original, wall_mask, result, output_dir, product_name="product"):
    """Visualize and save results with timestamped filenames"""
    print("Visualizing results...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save wall mask
    mask_viz = np.zeros_like(original)
    mask_viz[wall_mask] = [0, 255, 0]  # Green for wall
    cv2.imwrite(f"{output_dir}/wall_mask_{product_name}_{timestamp}.png", mask_viz)
    
    # Save final result
    cv2.imwrite(f"{output_dir}/result_{product_name}_{timestamp}.png", result)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(wall_mask, cmap='gray')
    axes[1].set_title("Wall Segmentation", fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"AR Preview - {product_name.title()}", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_{product_name}_{timestamp}.png", 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved with timestamp: {timestamp}")
    return timestamp

def main():
    print("🚀 AR Preview Pipeline Starting...")
    
    # Configuration
    room_image_path = "assets/room_wall.png"
    output_dir = "output"
    use_fast_mode = True  # Set to False for higher quality
    
    # Process both products
    products = [
        ("assets/prod_1_tv.png", "tv"),
        ("assets/prod_2_painting.png", "painting")
    ]
    
    try:
        # Step 1: Load SAM model
        print(f"🔧 Using {'FAST' if use_fast_mode else 'HIGH QUALITY'} mode")
        mask_generator = load_sam_model(use_fast_config=use_fast_mode)
        if mask_generator is None:
            return
        
        results = {}
        
        for product_path, product_name in products:
            if not os.path.exists(product_path):
                print(f"⚠️  Product not found: {product_path}")
                continue
                
            print(f"\n📸 Processing {product_name}...")
            
            # Step 2: Segment walls (reuse for efficiency)
            if 'wall_data' not in results:
                original_image, wall_mask, all_masks = segment_walls(room_image_path, mask_generator)
                results['wall_data'] = (original_image, wall_mask, all_masks)
            else:
                original_image, wall_mask, all_masks = results['wall_data']
                print(f"Reusing wall segmentation (found {len(all_masks)} masks)")
            
            # Step 3: Place product on wall
            result_image = place_product_on_wall(original_image, wall_mask, product_path)
            
            # Step 4: Visualize and save results
            timestamp = visualize_results(original_image, wall_mask, result_image, output_dir, product_name)
            results[product_name] = {
                'result': result_image,
                'timestamp': timestamp
            }
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {output_dir}/")
        print(f"🎯 Processed {len([p for p, _ in products if os.path.exists(p)])} products")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
