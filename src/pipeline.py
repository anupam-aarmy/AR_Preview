#!/usr/bin/env python3
"""
Enhanced AR Preview Pipeline with improved product placement
"""
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
from pathlib import Path
import argparse
from datetime import datetime

class ARPreviewPipeline:
    def __init__(self, model_path="models/sam_vit_h_4b8939.pth"):
        self.model_path = model_path
        self.mask_generator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_sam_model(self):
        """Load SAM model for segmentation"""
        print("Loading SAM model...")
        
        if not os.path.exists(self.model_path):
            print(f"SAM model not found at {self.model_path}")
            print("Run: python download_sam.py")
            return False
        
        # Load SAM model
        sam = sam_model_registry["vit_h"](checkpoint=self.model_path)
        sam.to(device=self.device)
        
        # Create mask generator with better parameters
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        print(f"SAM model loaded successfully on {self.device}")
        return True
    
    def find_wall_mask(self, masks):
        """Find the best wall mask from all generated masks"""
        # Filter masks by area and position
        image_center_masks = []
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            
            # Get mask centroid
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Prefer larger masks that are more central
                image_center_masks.append({
                    'mask': mask,
                    'area': area,
                    'centroid': (cx, cy),
                    'data': mask_data
                })
        
        if not image_center_masks:
            # Fallback to largest mask
            return max(masks, key=lambda x: x['area'])['segmentation']
        
        # Sort by area and take one of the larger, more central masks
        image_center_masks.sort(key=lambda x: x['area'], reverse=True)
        
        # Return the mask of a large, central region (likely wall)
        return image_center_masks[0]['mask']
    
    def get_optimal_placement_area(self, wall_mask, product_aspect_ratio):
        """Find optimal area on wall for product placement"""
        # Find contours
        wall_mask_uint8 = wall_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(wall_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No wall contours found")
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate optimal product size based on wall dimensions
        wall_aspect_ratio = w / h
        
        if product_aspect_ratio > wall_aspect_ratio:
            # Product is wider relative to wall - constrain by width
            target_w = int(w * 0.35)  # 35% of wall width
            target_h = int(target_w / product_aspect_ratio)
        else:
            # Product is taller relative to wall - constrain by height
            target_h = int(h * 0.35)  # 35% of wall height
            target_w = int(target_h * product_aspect_ratio)
        
        # Center the product on the wall
        target_x = x + (w - target_w) // 2
        target_y = y + (h - target_h) // 2
        
        return target_x, target_y, target_w, target_h
    
    def blend_product_advanced(self, image, product, x, y, w, h):
        """Advanced blending with better edge handling"""
        result = image.copy()
        
        # Resize product
        if product.shape[2] == 4:  # Has alpha channel
            product_resized = cv2.resize(product, (w, h))
            alpha = product_resized[:, :, 3] / 255.0
            product_rgb = product_resized[:, :, :3]
        else:
            product_resized = cv2.resize(product, (w, h))
            # Create soft alpha for non-transparent products
            alpha = np.ones((h, w)) * 0.95  # Slight transparency for realism
            product_rgb = product_resized
        
        # Add slight shadow effect
        shadow_offset = 5
        shadow_alpha = alpha * 0.3
        
        # Apply shadow (if within bounds)
        if y + shadow_offset + h < image.shape[0] and x + shadow_offset + w < image.shape[1]:
            shadow_region = result[y + shadow_offset:y + shadow_offset + h, 
                                 x + shadow_offset:x + shadow_offset + w]
            for c in range(3):
                shadow_region[:, :, c] = shadow_region[:, :, c] * (1 - shadow_alpha)
        
        # Blend product
        region = result[y:y+h, x:x+w]
        for c in range(3):
            region[:, :, c] = (alpha * product_rgb[:, :, c] + 
                             (1 - alpha) * region[:, :, c])
        
        return result
    
    def process_image(self, room_image_path, product_image_path, output_dir="output"):
        """Process a single image through the pipeline"""
        print(f"Processing: {room_image_path} + {product_image_path}")
        
        # Load images
        image = cv2.imread(room_image_path)
        product = cv2.imread(product_image_path, cv2.IMREAD_UNCHANGED)
        
        if image is None or product is None:
            raise ValueError("Could not load images")
        
        # Convert to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        print("Generating segmentation masks...")
        masks = self.mask_generator.generate(image_rgb)
        print(f"Generated {len(masks)} masks")
        
        # Find wall mask
        wall_mask = self.find_wall_mask(masks)
        
        # Calculate product placement
        product_aspect_ratio = product.shape[1] / product.shape[0]
        x, y, w, h = self.get_optimal_placement_area(wall_mask, product_aspect_ratio)
        
        # Blend product
        result = self.blend_product_advanced(image, product, x, y, w, h)
        
        # Save results
        self.save_results(image, wall_mask, result, output_dir, 
                         Path(product_image_path).stem)
        
        return result
    
    def save_results(self, original, wall_mask, result, output_dir, product_name):
        """Save visualization results in organized structure"""
        # Create organized output directories for Task 1 (Deterministic Pipeline)
        task1_base = os.path.join(output_dir, "task1_deterministic")
        results_dir = os.path.join(task1_base, "results")
        comparisons_dir = os.path.join(task1_base, "comparisons")
        masks_dir = os.path.join(task1_base, "masks")
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(comparisons_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual results
        mask_viz = np.zeros_like(original)
        mask_viz[wall_mask] = [0, 255, 0]
        cv2.imwrite(os.path.join(masks_dir, f"wall_mask_{product_name}_{timestamp}.png"), mask_viz)
        cv2.imwrite(os.path.join(results_dir, f"result_{product_name}_{timestamp}.png"), result)
        
        # Create comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Room", fontsize=14)
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
        
        print(f"Results saved: {results_dir}/result_{product_name}_{timestamp}.png")

def main():
    parser = argparse.ArgumentParser(description="AR Preview Pipeline")
    parser.add_argument("--room", default="assets/room_wall.png", 
                       help="Path to room image")
    parser.add_argument("--product", 
                       help="Path to product image (if not specified, processes all)")
    parser.add_argument("--output", default="output", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ARPreviewPipeline()
    
    if not pipeline.load_sam_model():
        return
    
    try:
        if args.product:
            # Process single product
            pipeline.process_image(args.room, args.product, args.output)
        else:
            # Process all products
            product_files = ["assets/prod_1_paintings.png", "assets/prod_2_tv.png"]
            for product_file in product_files:
                if os.path.exists(product_file):
                    pipeline.process_image(args.room, product_file, args.output)
        
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()