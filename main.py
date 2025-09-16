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
        # Faster configuration for development/testing with lower memory usage
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=12,  # Reduced further from 16
            pred_iou_thresh=0.90,  # Higher threshold for fewer masks
            stability_score_thresh=0.96,  # Higher stability requirement
            crop_n_layers=0,  # Disabled cropping for speed and memory
            min_mask_region_area=1000,  # Larger minimum area to reduce masks
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
    """Create an alpha mask for products without transparency using smart background detection"""
    print("Creating product mask using background detection...")
    
    h, w = product_image.shape[:2]
    
    # Convert to different color spaces for better background detection
    hsv = cv2.cvtColor(product_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(product_image, cv2.COLOR_BGR2LAB)
    
    # Create multiple masks based on edge detection and color consistency
    gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
    
    # Edge-based mask: find strong edges (likely product boundaries)
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    
    # Distance transform from edges
    dist_transform = cv2.distanceTransform(255 - edges_dilated, cv2.DIST_L2, 5)
    
    # Normalize and create mask
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold to create binary mask
    _, mask_from_edges = cv2.threshold(dist_normalized, 20, 255, cv2.THRESH_BINARY)
    
    # Corner-based background detection (assume corners are background)
    corner_samples = [
        product_image[0:10, 0:10],      # Top-left
        product_image[0:10, -10:],      # Top-right  
        product_image[-10:, 0:10],      # Bottom-left
        product_image[-10:, -10:]       # Bottom-right
    ]
    
    # Get average background color
    bg_colors = []
    for corner in corner_samples:
        bg_colors.append(np.mean(corner.reshape(-1, 3), axis=0))
    bg_color = np.mean(bg_colors, axis=0)
    
    # Create mask based on color similarity to background
    color_diff = np.sqrt(np.sum((product_image - bg_color) ** 2, axis=2))
    color_threshold = np.std(color_diff) * 0.8  # Adaptive threshold
    mask_from_color = (color_diff > color_threshold).astype(np.uint8) * 255
    
    # Combine masks
    combined_mask = cv2.bitwise_and(mask_from_edges, mask_from_color)
    
    # Clean up mask with morphological operations
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes using contour detection
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Fill the largest contour
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
    """Segment walls from room image using SAM with timing and memory optimization"""
    print(f"Segmenting walls from {image_path}...")
    start_time = time.time()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize large images to reduce memory usage
    original_shape = image.shape
    max_dimension = 1024
    h, w = image.shape[:2]
    
    if max(h, w) > max_dimension:
        if h > w:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
        else:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
        
        image = cv2.resize(image, (new_w, new_h))
        print(f"Resized from {original_shape[:2]} to {image.shape[:2]} for memory optimization")
    
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
    
    # Create organized output directories for Task 1 (Deterministic Pipeline)
    task1_base = os.path.join(output_dir, "task1_deterministic")
    results_dir = os.path.join(task1_base, "results")
    comparisons_dir = os.path.join(task1_base, "comparisons") 
    masks_dir = os.path.join(task1_base, "masks")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save wall mask
    mask_viz = np.zeros_like(original)
    mask_viz[wall_mask] = [0, 255, 0]  # Green for wall
    cv2.imwrite(os.path.join(masks_dir, f"wall_mask_{product_name}_{timestamp}.png"), mask_viz)
    
    # Save final result
    cv2.imwrite(os.path.join(results_dir, f"result_{product_name}_{timestamp}.png"), result)
    
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
    plt.savefig(os.path.join(comparisons_dir, f"comparison_{product_name}_{timestamp}.png"), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved with timestamp: {timestamp}")
    print(f"  📁 Results: {results_dir}/")
    print(f"  📊 Comparisons: {comparisons_dir}/")
    print(f"  🎭 Masks: {masks_dir}/")
    return timestamp


def main():
    print("🚀 AR Preview Pipeline Starting...")
    
    # Configuration - Following AI Assignment requirements: single room image input
    room_image_path = "assets/room_wall_3.png"  # Main room image as per assignment (room_wall_3 for testing)
    output_dir = "output"
    
    # Performance mode: Set to False for higher quality but slower processing
    # Fast mode: ~50s processing, High quality: ~350s processing with more detailed masks
    use_fast_mode = True  # Change to False for HIGH QUALITY mode with more detailed segmentation
    
    # Product images to process
    products = [
        ("assets/prod_1_tv.png", "tv"),
        ("assets/prod_2_painting.png", "painting")
    ]
    
    try:
        # Verify room image exists
        if not os.path.exists(room_image_path):
            print(f"❌ Room image not found: {room_image_path}")
            print("Make sure room_wall.png exists in the assets folder")
            return
        
        # Verify products exist
        for product_path, product_name in products:
            if not os.path.exists(product_path):
                print(f"❌ Product not found: {product_path}")
                return
        
        print(f"🏠 Using room image: {room_image_path}")
        print(f"🖼️  Processing {len(products)} products: {[p[1] for p in products]}")
        
        # Load SAM model with performance configuration
        print(f"🔧 Using {'FAST' if use_fast_mode else 'HIGH QUALITY'} mode")
        mask_generator = load_sam_model(use_fast_config=use_fast_mode)
        if mask_generator is None:
            return
        
        results = {}
        
        # Segment walls once for this room (reuse for multiple products)
        print(f"\n🔍 Segmenting walls from {room_image_path}...")
        original_image, wall_mask, all_masks = segment_walls(room_image_path, mask_generator)
        
        # Process each product with the same wall segmentation
        for product_path, product_name in products:
            print(f"\n📸 Processing {product_name} placement...")
            
            # Place product on wall
            result_image = place_product_on_wall(original_image, wall_mask, product_path)
            
            # Visualize and save results
            timestamp = visualize_results(original_image, wall_mask, result_image, output_dir, product_name)
            
            results[product_name] = {
                'result': result_image,
                'timestamp': timestamp,
                'product_path': product_path
            }
        
        # Print completion summary
        print(f"\n✅ AR Preview Pipeline completed successfully!")
        print(f"📁 Results saved to: {output_dir}/")
        print(f"📊 PROCESSING SUMMARY:")
        print(f"🏠 Room: {room_image_path}")
        print(f"🔍 Wall masks detected: {len(all_masks)}")
        print(f"⚙️  Performance mode: {'FAST' if use_fast_mode else 'HIGH QUALITY'}")
        
        for product_name, result_data in results.items():
            timestamp = result_data['timestamp']
            product_path = result_data['product_path']
            print(f"📸 {product_name.title()} ({product_path}) → result_{product_name}_{timestamp}.png")
        
        print(f"\n🎯 Total products processed: {len(results)}")
        print(f"✅ Pipeline completed as per AI Assignment requirements")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()