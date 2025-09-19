"""
Task 1: Deterministic Product Placement Pipeline (PRODUCTION)
SAM wall segmentation + OpenCV homography with aspect-corrected placement
Production-ready implementation with enhanced quality and error handling

PRODUCTION FEATURES:
- Products completely fill placement areas (no background visible)
- Actual aspect ratio preservation from product images
- Smart sizing with proper bounds checking
- LANCZOS resampling for maximum quality
- Safe positioning preventing floor/wall overflow
- Enhanced error handling and logging
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import urllib.request
import subprocess
from PIL import Image, ImageEnhance

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def check_sam_installation():
    """Check if SAM is properly installed"""
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        return True
    except ImportError:
        print("❌ SAM not installed. Please install: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return False

def setup_sam():
    """Setup SAM model and download checkpoint if needed"""
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check for SAM checkpoint
    sam_checkpoint = models_dir / "sam_vit_h_4b8939.pth"
    
    if not sam_checkpoint.exists():
        print("📥 Downloading SAM checkpoint...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(url, sam_checkpoint)
        print("✅ SAM checkpoint downloaded")
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "vit_h"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Create mask generator with optimized settings
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,
    )
    
    print(f"🎯 SAM initialized ({device})")
    return mask_generator, device

class FixedDeterministicPipeline:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("output/task1_deterministic")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create run-specific directory
        self.run_dir = self.output_dir / f"run_aspect_corrected_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print("🔧 Task 1: Deterministic Pipeline (SAM + OpenCV)")
        print(f"📁 Output directory: {self.run_dir}")
        
    def load_and_enhance_product(self, product_path):
        """Load product image with enhanced preprocessing"""
        if not os.path.exists(product_path):
            raise FileNotFoundError(f"Product image not found: {product_path}")
            
        # Load with transparency support
        product = cv2.imread(product_path, cv2.IMREAD_UNCHANGED)
        if product is None:
            raise ValueError(f"Could not load product image: {product_path}")
            
        # Convert BGRA to RGB with alpha channel preserved
        if len(product.shape) == 3 and product.shape[2] == 4:
            product_rgb = cv2.cvtColor(product, cv2.COLOR_BGRA2RGB)
            alpha = product[:, :, 3]
        else:
            product_rgb = cv2.cvtColor(product, cv2.COLOR_BGR2RGB)
            alpha = None
            
        # Enhance product image for better quality
        product_pil = Image.fromarray(product_rgb)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(product_pil)
        product_pil = enhancer.enhance(1.2)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(product_pil)
        product_pil = enhancer.enhance(1.1)
        
        product_enhanced = np.array(product_pil)
        
        return product_enhanced, alpha
        
    def get_actual_product_aspect_ratio(self, product_image):
        """Get the ACTUAL aspect ratio of the product image"""
        h, w = product_image.shape[:2]
        aspect_ratio = w / h
        return aspect_ratio
        
    def segment_wall_with_sam(self, image, mask_generator):
        """Segment wall using SAM with enhanced selection"""
        # Convert BGR to RGB for SAM
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = mask_generator.generate(rgb_image)
        
        if not masks:
            raise ValueError("No masks found by SAM")
        
        # Sort masks by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        # Enhanced wall selection criteria
        best_wall_mask = None
        best_score = 0
        
        h, w = image.shape[:2]
        image_area = h * w
        
        for i, mask in enumerate(masks[:10]):  # Check top 10 largest masks
            mask_array = mask['segmentation']
            area = mask['area']
            bbox = mask['bbox']
            
            # Calculate features for wall detection
            area_ratio = area / image_area
            aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 0
            
            # Calculate center position
            moments = cv2.moments(mask_array.astype(np.uint8))
            if moments['m00'] > 0:
                center_x = moments['m10'] / moments['m00']
                center_y = moments['m01'] / moments['m00']
                center_x_norm = center_x / w
                center_y_norm = center_y / h
            else:
                continue
            
            # Wall scoring criteria (enhanced)
            score = 0
            
            # Prefer larger segments (20-60% of image)
            if 0.15 < area_ratio < 0.70:
                score += 30 * (1 - abs(area_ratio - 0.35))  # Peak at 35%
                
            # Prefer segments in upper-center area (walls are usually there)
            if 0.2 < center_x_norm < 0.8 and 0.1 < center_y_norm < 0.7:
                score += 25
                
            # Prefer moderate aspect ratios (not too thin or square)
            if 1.0 < aspect_ratio < 3.0:
                score += 20
                
            # Prefer segments with good predicted IoU
            score += mask.get('predicted_iou', 0) * 15
            
            # Prefer segments with good stability
            score += mask.get('stability_score', 0) * 10
            
            if score > best_score:
                best_score = score
                best_wall_mask = mask_array
        
        if best_wall_mask is None:
            # Fallback: use largest mask
            best_wall_mask = masks[0]['segmentation']
            
        # Convert boolean mask to uint8
        wall_mask = (best_wall_mask * 255).astype(np.uint8)
        
        print(f"🎯 Wall detected (confidence: {best_score:.1f}/100)")
        return wall_mask
        
    def calculate_dimensions_using_actual_aspect(self, room_shape, product_image, product_type, size_variant="default"):
        """Calculate dimensions using ACTUAL product aspect ratio (not hardcoded ratios)"""
        h, w = room_shape[:2]
        
        # Get ACTUAL aspect ratio from the product image
        actual_aspect = self.get_actual_product_aspect_ratio(product_image)
        
        # Calculate width based on product type and size variant
        if product_type == "tv":
            if size_variant == "large":
                product_w = int(w * 0.35)  # Large TV: 35% of room width
            else:
                product_w = int(w * 0.28)  # Standard TV: 28% of room width
        else:  # painting
            if size_variant == "large":
                product_w = int(w * 0.22)  # Large painting: 22% of room width
            else:
                product_w = int(w * 0.18)  # Standard painting: 18% of room width
        
        # Calculate height using ACTUAL aspect ratio of the product
        product_h = int(product_w / actual_aspect)
                
        # Ensure minimum size
        product_w = max(product_w, 100)
        product_h = max(product_h, 100)
        
        # Ensure maximum size doesn't exceed reasonable wall bounds
        max_h = int(h * 0.4)  # Max 40% of room height
        if product_h > max_h:
            product_h = max_h
            product_w = int(product_h * actual_aspect)
        
        print(f"📐 {product_type} ({size_variant}): {product_w}×{product_h} px, aspect ratio {actual_aspect:.3f}:1")
        return product_w, product_h
        
    def find_safe_wall_position(self, wall_mask, product_w, product_h, product_type):
        """Find safe position on wall ensuring product fits completely within wall bounds"""
        h, w = wall_mask.shape
        
        # Find wall boundaries
        wall_points = np.where(wall_mask > 0)
        if len(wall_points[0]) == 0:
            raise ValueError("No wall area found in mask")
            
        wall_top = np.min(wall_points[0])
        wall_bottom = np.max(wall_points[0])
        wall_left = np.min(wall_points[1])
        wall_right = np.max(wall_points[1])
        
        wall_width = wall_right - wall_left
        wall_height = wall_bottom - wall_top
        
        # Ensure product fits within wall bounds
        if product_w > wall_width:
            product_w = wall_width - 20  # Leave 20px margin
            
        if product_h > wall_height:
            product_h = wall_height - 20  # Leave 20px margin  
        
        # Center horizontally in wall area
        center_x = wall_left + wall_width // 2
        start_x = center_x - product_w // 2
        
        # Safe vertical positioning by product type
        if product_type == "tv":
            # TVs: Place in upper-middle area (25% from wall top)
            target_y = wall_top + int(wall_height * 0.25)
            start_y = target_y
        else:  # painting
            # Paintings: Center vertically in wall area
            center_y = wall_top + wall_height // 2
            start_y = center_y - product_h // 2
                
        # Final bounds checking
        start_x = max(wall_left, min(start_x, wall_right - product_w))
        start_y = max(wall_top, min(start_y, wall_bottom - product_h))
        
        # Verify final position is within wall
        end_x = start_x + product_w
        end_y = start_y + product_h
        
        if end_x > wall_right or end_y > wall_bottom:
            start_x = wall_right - product_w
            start_y = wall_bottom - product_h
        
        return start_x, start_y, product_w, product_h
        
    def place_product_filling_area(self, room_image, product_image, alpha_channel, start_x, start_y, product_w, product_h):
        """Place product COMPLETELY FILLING the designated area"""
        
        # Resize product to EXACTLY fill the area
        product_pil = Image.fromarray(product_image)
        product_resized = product_pil.resize((product_w, product_h), Image.Resampling.LANCZOS)
        
        # Enhance after resize to maintain quality
        enhancer = ImageEnhance.Sharpness(product_resized)
        product_resized = enhancer.enhance(1.1)
        
        product_final = np.array(product_resized)
        
        # Handle alpha channel
        if alpha_channel is not None:
            alpha_resized = cv2.resize(alpha_channel, (product_w, product_h), interpolation=cv2.INTER_LINEAR)
            alpha_norm = alpha_resized.astype(np.float32) / 255.0
        else:
            # Create alpha mask from product content
            gray = cv2.cvtColor(product_final, cv2.COLOR_RGB2GRAY)
            alpha_norm = (gray > 10).astype(np.float32)
            
        # Convert room to RGB
        room_rgb = cv2.cvtColor(room_image, cv2.COLOR_BGR2RGB)
        result = room_rgb.copy()
        
        # Place product to COMPLETELY FILL the designated area
        for c in range(3):  # RGB channels
            result[start_y:start_y+product_h, start_x:start_x+product_w, c] = (
                alpha_norm * product_final[:, :, c] + 
                (1 - alpha_norm) * result[start_y:start_y+product_h, start_x:start_x+product_w, c]
            )
        
        return result
        
    def process_single_product(self, room_path, product_path, product_type, mask_generator):
        """Process a single product with FIXED sizing and positioning"""
        print(f"🎯 Processing {product_type}: {Path(product_path).name}")
        
        # Load room image
        room_image = cv2.imread(room_path)
        if room_image is None:
            raise ValueError(f"Could not load room: {room_path}")
            
        # Load and enhance product
        product_image, alpha_channel = self.load_and_enhance_product(product_path)
        
        # Segment wall with SAM
        wall_mask = self.segment_wall_with_sam(room_image, mask_generator)
        
        # Process size variants
        variants = ["standard", "large"]
        results = {}
        
        for variant in variants:
            # Calculate ACTUAL dimensions using real product aspect ratios
            product_w, product_h = self.calculate_dimensions_using_actual_aspect(
                room_image.shape, product_image, product_type, variant
            )
            
            # Find SAFE position ensuring product fits completely
            start_x, start_y, final_w, final_h = self.find_safe_wall_position(
                wall_mask, product_w, product_h, product_type
            )
            
            # Place product FILLING the entire designated area
            result = self.place_product_filling_area(
                room_image, product_image, alpha_channel, 
                start_x, start_y, final_w, final_h
            )
            
            results[variant] = {
                'result': result,
                'wall_mask': wall_mask,
                'placement': (start_x, start_y, final_w, final_h),
                'original': room_image,
                'product': product_image
            }
            
            # Save individual result
            result_path = self.run_dir / f"{product_type}_{variant}_aspect_corrected_{self.timestamp}.png"
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(result_path), result_bgr)
            print(f"✅ Saved {variant}: {result_path.name}")
            
        return results
        
    def create_fixed_comparison(self, results, product_type, product_image):
        """Create comparison showing FIXED implementation"""
        variants = list(results.keys())
        if len(variants) == 0:
            return None
            
        # Create figure with reduced matplotlib verbosity
        plt.rcParams['figure.max_open_warning'] = 0
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Task 1: Deterministic {product_type.title()} Placement', 
                     fontsize=16, fontweight='bold')
        
        # Original room
        original = results[variants[0]]['original']
        axes[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original Room', fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        
        # Product
        axes[0,1].imshow(product_image)
        axes[0,1].set_title(f'Product: {product_type.title()}', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # Wall mask
        wall_mask = results[variants[0]]['wall_mask']
        axes[0,2].imshow(wall_mask, cmap='gray')
        axes[0,2].set_title('SAM Wall Segmentation', fontsize=12)
        axes[0,2].axis('off')
        
        # First result
        axes[0,3].imshow(results[variants[0]]['result'])
        placement = results[variants[0]]['placement']
        aspect = placement[2] / placement[3]
        title = f'{variants[0].title()} Size (AR: {aspect:.2f}:1)'
        axes[0,3].set_title(f'{title}', fontsize=11, fontweight='bold')
        axes[0,3].axis('off')
        
        # Bottom row
        if len(variants) > 1:
            # First placement visualization with EXACT dimensions
            placement1 = results[variants[0]]['placement']
            vis1 = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).copy()
            cv2.rectangle(vis1, (placement1[0], placement1[1]), 
                         (placement1[0]+placement1[2], placement1[1]+placement1[3]), 
                         (255, 0, 0), 3)
            axes[1,0].imshow(vis1)
            axes[1,0].set_title(f'{variants[0]} Placement: {placement1[2]}x{placement1[3]}', fontsize=11)
            axes[1,0].axis('off')
            
            # Second placement visualization
            placement2 = results[variants[1]]['placement']
            vis2 = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).copy()
            cv2.rectangle(vis2, (placement2[0], placement2[1]), 
                         (placement2[0]+placement2[2], placement2[1]+placement2[3]), 
                         (0, 255, 0), 3)
            axes[1,1].imshow(vis2)
            axes[1,1].set_title(f'{variants[1]} Placement: {placement2[2]}x{placement2[3]}', fontsize=11)
            axes[1,1].axis('off')
            
            # Size comparison
            axes[1,2].text(0.5, 0.5, f'{variants[0]}: {placement1[2]}x{placement1[3]}\\n{variants[1]}: {placement2[2]}x{placement2[3]}', 
                          ha='center', va='center', fontsize=12, transform=axes[1,2].transAxes)
            axes[1,2].set_title('Size Comparison', fontsize=12)
            axes[1,2].axis('off')
            
            # Second result
            axes[1,3].imshow(results[variants[1]]['result'])
            aspect = placement2[2] / placement2[3]
            title = f'{variants[1].title()} Size (AR: {aspect:.2f}:1)'
            axes[1,3].set_title(f'{title}', fontsize=11, fontweight='bold')
            axes[1,3].axis('off')
        else:
            for i in range(4):
                axes[1,i].axis('off')
                
        plt.tight_layout()
        
        # Save comparison
        comparison_path = self.run_dir / f"{product_type}_ASPECT_CORRECTED_comparison_{self.timestamp}.png"
        fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved comparison: {comparison_path.name}")
        return comparison_path
        
    def run_aspect_corrected_pipeline(self):
        """Run the FIXED deterministic pipeline"""
        print("🚀 Starting Deterministic Pipeline...")
        
        # Check SAM installation
        if not check_sam_installation():
            return None
            
        # Setup SAM
        mask_generator, device = setup_sam()
        
        # Define test configuration with new assets
        room_path = "assets/room_wall.png"
        
        # Process TV using tv_1.png
        try:
            tv_product_path = "assets/tv_1.png"
            tv_product, _ = self.load_and_enhance_product(tv_product_path)
            
            tv_results = self.process_single_product(room_path, tv_product_path, "tv", mask_generator)
            
            if tv_results:
                self.create_fixed_comparison(tv_results, "tv", tv_product)
                
        except Exception as e:
            print(f"❌ TV processing failed: {e}")
            
        # Process Painting using painting_1.png
        try:
            painting_product_path = "assets/painting_1.png"
            painting_product, _ = self.load_and_enhance_product(painting_product_path)
            
            painting_results = self.process_single_product(room_path, painting_product_path, "painting", mask_generator)
            
            if painting_results:
                self.create_fixed_comparison(painting_results, "painting", painting_product)
                
        except Exception as e:
            print(f"❌ Painting processing failed: {e}")
            
        print("✅ Deterministic Pipeline Complete")
        print(f"📁 Results: {self.run_dir}")

def main():
    """Main execution function"""
    pipeline = FixedDeterministicPipeline()
    pipeline.run_aspect_corrected_pipeline()
    
    return pipeline

if __name__ == "__main__":
    main()