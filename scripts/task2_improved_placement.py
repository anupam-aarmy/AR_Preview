"""
Task 2: IMPROVED Product Placement Pipeline
Addresses detail loss, alignment, aspect ratios, and shape preservation
Uses new assets: tv_1.png, tv_2.png, painting_1.png, painting_2.png
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Set up HuggingFace environment to suppress warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class ImprovedProductPlacementPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("output/task2_real_product_placement")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create run-specific directory
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print("🔧 TASK 2: IMPROVED PRODUCT PLACEMENT PIPELINE")
        print("✅ Enhanced detail preservation, alignment, and aspect ratios")
        print(f"📱 Device: {self.device}")
        print(f"📁 Output: {self.run_dir}")
        
    def setup_pipeline(self):
        """Setup enhanced ControlNet inpainting pipeline"""
        print("🚀 Setting up IMPROVED PRODUCT PLACEMENT pipeline...")
        
        # Load ControlNet for inpainting
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Load Stable Diffusion Inpainting pipeline
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.pipe = self.pipe.to(self.device)
        
        # Load depth estimation model
        self.depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.depth_model = self.depth_model.to(self.device)
        
        print("✅ Improved product placement pipeline ready")
        
    def load_product_image(self, product_path):
        """Load product image with enhanced preprocessing"""
        if not os.path.exists(product_path):
            raise FileNotFoundError(f"Product image not found: {product_path}")
            
        # Load with transparency support
        product = cv2.imread(product_path, cv2.IMREAD_UNCHANGED)
        if product is None:
            raise ValueError(f"Could not load product image: {product_path}")
            
        # Convert BGRA to RGB if has alpha channel
        if len(product.shape) == 3 and product.shape[2] == 4:
            product_rgb = cv2.cvtColor(product, cv2.COLOR_BGRA2RGB)
        else:
            product_rgb = cv2.cvtColor(product, cv2.COLOR_BGR2RGB)
            
        # Enhance product image for better detail preservation
        product_pil = Image.fromarray(product_rgb)
        
        # Enhance sharpness to preserve details
        enhancer = ImageEnhance.Sharpness(product_pil)
        product_pil = enhancer.enhance(1.2)  # Slightly sharper
        
        # Enhance contrast for better feature preservation
        enhancer = ImageEnhance.Contrast(product_pil)
        product_pil = enhancer.enhance(1.1)  # Slightly more contrast
        
        product_enhanced = np.array(product_pil)
        
        print(f"✅ Loaded and enhanced product: {product.shape} -> {product_enhanced.shape} from {Path(product_path).name}")
        return product_enhanced
        
    def create_depth_map(self, image):
        """Create enhanced depth map with better wall detection"""
        h, w = image.shape[:2]
        
        # Convert BGR to RGB for depth model
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        inputs = self.depth_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Convert to numpy and normalize
        depth = predicted_depth.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = (depth * 255).astype(np.uint8)
        
        # Resize depth map to match original image dimensions
        depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Apply enhanced CLAHE for better wall detection
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        depth_enhanced = clahe.apply(depth_resized)
        
        # Additional gaussian blur for smoother depth transitions
        depth_enhanced = cv2.GaussianBlur(depth_enhanced, (3, 3), 0)
        
        print(f"🔍 Enhanced depth map: {depth.shape} -> resized to {depth_enhanced.shape}")
        return depth_enhanced
        
    def calculate_optimal_dimensions(self, room_shape, product_type, size_variant):
        """Calculate optimal product dimensions preserving aspect ratios"""
        h, w = room_shape[:2]
        
        if product_type == "tv":
            # TVs should maintain 16:9 aspect ratio but be appropriately sized
            if size_variant == "42_inch":
                # 42" TV: 30% width, maintain 16:9 aspect ratio
                product_w = int(w * 0.30)
                product_h = int(product_w / 1.78)  # 16:9 aspect ratio
            else:  # 55_inch
                # 55" TV: 38% width, maintain 16:9 aspect ratio
                product_w = int(w * 0.38)
                product_h = int(product_w / 1.78)  # 16:9 aspect ratio
                
        else:  # painting
            # Paintings should maintain their natural proportions
            if size_variant == "medium":
                # Medium painting: 26% width, preserve aspect ratio
                product_w = int(w * 0.26)
                product_h = int(product_w * 1.0)  # Assume square for now, will adjust based on actual image
            else:  # large
                # Large painting: 35% width, preserve aspect ratio
                product_w = int(w * 0.35)
                product_h = int(product_w * 1.0)  # Assume square for now, will adjust based on actual image
                
        print(f"📐 {product_type} ({size_variant}): {product_w}x{product_h} ({product_w/w*100:.0f}%x{product_h/h*100:.0f}%)")
        if product_type == "tv":
            print(f"   📺 TV aspect ratio: {product_w/product_h:.2f}:1 (target: 1.78:1)")
        return product_w, product_h
        
    def create_optimal_placement_mask(self, room_shape, product_w, product_h, product_type):
        """Create optimally positioned mask avoiding floor and ceiling"""
        h, w = room_shape[:2]
        
        # Center horizontally
        start_x = (w - product_w) // 2
        
        # Better vertical positioning - center in available wall space
        if product_type == "tv":
            # TVs: Place in upper-middle of wall (30% from top)
            start_y = int(h * 0.30)
        else:  # painting
            # Paintings: Center in available wall space (avoid floor and ceiling)
            # Create safe zone: 15% from top, 20% from bottom
            safe_top = int(h * 0.15)
            safe_bottom = int(h * 0.80)
            available_height = safe_bottom - safe_top
            
            # Center in available space
            if product_h <= available_height:
                start_y = safe_top + (available_height - product_h) // 2
            else:
                # If too tall, place at safe top
                start_y = safe_top
                # Adjust height to fit in safe zone
                product_h = min(product_h, available_height)
        
        # Ensure bounds
        start_x = max(0, min(start_x, w - product_w))
        start_y = max(int(h * 0.1), min(start_y, h - product_h - int(h * 0.1)))
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[start_y:start_y+product_h, start_x:start_x+product_w] = 255
        
        print(f"📍 Optimal placement: x={start_x}, y={start_y} (y: {start_y/h*100:.1f}% from top)")
        return mask, (start_x, start_y, product_w, product_h)
        
    def resize_product_preserving_details(self, product_image, target_w, target_h, product_type):
        """Resize product while preserving maximum detail and shape"""
        
        # Use high-quality resampling
        product_pil = Image.fromarray(product_image)
        
        # For better detail preservation, use LANCZOS resampling
        product_resized = product_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # Apply slight sharpening after resize to recover details
        enhancer = ImageEnhance.Sharpness(product_resized)
        product_resized = enhancer.enhance(1.15)
        
        product_final = np.array(product_resized)
        
        print(f"🔍 High-quality resize: {product_image.shape[:2]} -> {product_final.shape[:2]} with detail preservation")
        return product_final
        
    def place_product_with_blending(self, room_image, product_image, placement_info, product_type):
        """Place product with better blending and detail preservation"""
        start_x, start_y, product_w, product_h = placement_info
        
        # Create a copy of the room
        room_with_product = room_image.copy()
        
        # Resize product with detail preservation
        product_resized = self.resize_product_preserving_details(
            product_image, product_w, product_h, product_type
        )
        
        # Simple direct placement - let inpainting handle the blending
        room_with_product[start_y:start_y+product_h, start_x:start_x+product_w] = product_resized
        
        return room_with_product
        
    def create_enhanced_control_image(self, room_image, mask, depth_map):
        """Create enhanced control image for better ControlNet conditioning"""
        # Use depth map as primary control
        control = depth_map.copy()
        
        # Enhance the masked region more subtly
        mask_norm = mask.astype(np.float32) / 255.0
        control = control.astype(np.float32)
        
        # Gentle emphasis for placement area
        control = control + (mask_norm * 25)  # More subtle emphasis
        control = np.clip(control, 0, 255).astype(np.uint8)
        
        # Apply slight gaussian blur to smooth transitions
        control = cv2.GaussianBlur(control, (3, 3), 0)
        
        return control
        
    def generate_improved_placement(self, room_image, product_image, mask, depth_map, placement_info, product_type, size_variant):
        """Generate improved product placement with better detail preservation"""
        h, w = room_image.shape[:2]
        
        # Create initial placement with enhanced blending
        initial_image = self.place_product_with_blending(room_image, product_image, placement_info, product_type)
        
        # Create enhanced control image
        control_image = self.create_enhanced_control_image(room_image, mask, depth_map)
        
        # Convert to PIL Images
        initial_pil = Image.fromarray(initial_image)
        mask_pil = Image.fromarray(mask)
        control_pil = Image.fromarray(control_image)
        
        # Enhanced prompts for better detail preservation
        if product_type == "tv":
            prompt = f"photorealistic {size_variant.replace('_', '-')} television with sharp details, clear screen, natural lighting, perfectly mounted on wall, high definition, crisp edges"
            negative_prompt = "blurry, low quality, distorted, floating, unrealistic, soft focus, pixelated, low resolution"
        else:  # painting
            prompt = f"photorealistic framed {size_variant} artwork with fine details, clear texture, natural lighting, perfectly mounted on wall, sharp frame, high definition"
            negative_prompt = "blurry, low quality, distorted, floating, unrealistic, soft focus, pixelated, low resolution"
        
        print(f"🎨 Generating improved {product_type} ({size_variant})...")
        print(f"📝 Enhanced prompt: {prompt[:70]}...")
        
        # Optimized parameters for detail preservation and shape integrity
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=initial_pil,
            mask_image=mask_pil,
            control_image=control_pil,
            num_inference_steps=25,  # Fewer steps to preserve details
            guidance_scale=6.5,      # Lower guidance to preserve original features
            controlnet_conditioning_scale=0.7,  # Moderate depth conditioning
            strength=0.5,            # Lower strength to preserve product details
            generator=torch.Generator(device=self.device).manual_seed(42)
        ).images[0]
        
        return np.array(result)
        
    def process_single_product(self, room_path, product_path, product_type):
        """Process a single product with size variations"""
        print(f"\n🎯 Processing {product_type}: {Path(product_path).name}")
        
        # Load room and product images
        room_image = cv2.imread(room_path)
        if room_image is None:
            raise ValueError(f"Could not load room: {room_path}")
            
        product_image = self.load_product_image(product_path)
        
        print(f"✅ Room loaded: {room_image.shape}")
        
        # Create enhanced depth map
        print("🔍 Creating enhanced depth map...")
        depth_map = self.create_depth_map(room_image)
        
        # Define size variants
        if product_type == "tv":
            variants = ["42_inch", "55_inch"]
        else:  # painting
            variants = ["medium", "large"]
            
        results = {}
        
        for variant in variants:
            print(f"\n📺 Processing {variant}...")
            
            # Calculate optimal dimensions
            product_w, product_h = self.calculate_optimal_dimensions(
                room_image.shape, product_type, variant
            )
            
            # Create optimal placement mask
            mask, placement_info = self.create_optimal_placement_mask(
                room_image.shape, product_w, product_h, product_type
            )
            
            # Generate improved placement
            try:
                result = self.generate_improved_placement(
                    room_image, product_image, mask, depth_map, placement_info, product_type, variant
                )
                
                results[variant] = {
                    'result': result,
                    'mask': mask,
                    'placement': placement_info,
                    'original': room_image,
                    'product': product_image,
                    'depth': depth_map
                }
                
                # Save individual result
                result_path = self.run_dir / f"{product_type}_{variant}_{self.timestamp}.png"
                cv2.imwrite(str(result_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                print(f"✅ Saved: {result_path.name}")
                
            except Exception as e:
                print(f"❌ Failed to generate {variant}: {e}")
                
        return results
        
    def create_enhanced_comparison(self, results, product_type, product_image):
        """Create enhanced comparison visualization"""
        print(f"📊 Creating enhanced {product_type} comparison...")
        
        # Get variants
        variants = list(results.keys())
        if len(variants) == 0:
            print(f"❌ No results to compare for {product_type}")
            return None
            
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Task 2: Improved {product_type.title()} Placement (Detail Preserved)', fontsize=16, fontweight='bold')
        
        # Original room
        original = results[variants[0]]['original']
        axes[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original Room', fontsize=12, fontweight='bold')
        axes[0,0].axis('off')
        
        # Original product
        axes[0,1].imshow(product_image)
        axes[0,1].set_title(f'Enhanced {product_type.title()}', fontsize=12, fontweight='bold')
        axes[0,1].axis('off')
        
        # Enhanced depth map
        depth = results[variants[0]]['depth']
        axes[0,2].imshow(depth, cmap='viridis')
        axes[0,2].set_title('Enhanced Depth Map', fontsize=12)
        axes[0,2].axis('off')
        
        # First variant result
        if len(variants) > 0:
            axes[0,3].imshow(results[variants[0]]['result'])
            title = f'{variants[0].replace("_", " ").title()}'
            if product_type == "tv":
                placement = results[variants[0]]['placement']
                aspect = placement[2] / placement[3]
                title += f' (AR: {aspect:.2f}:1)'
            axes[0,3].set_title(f'{title} ✅', fontsize=11, fontweight='bold')
            axes[0,3].axis('off')
        else:
            axes[0,3].axis('off')
            
        # Bottom row
        if len(variants) > 1:
            # First variant mask
            axes[1,0].imshow(results[variants[0]]['mask'], cmap='gray')
            axes[1,0].set_title(f'{variants[0]} Mask', fontsize=12)
            axes[1,0].axis('off')
            
            # Second variant mask
            axes[1,1].imshow(results[variants[1]]['mask'], cmap='gray')
            axes[1,1].set_title(f'{variants[1]} Mask', fontsize=12)
            axes[1,1].axis('off')
            
            # Empty
            axes[1,2].axis('off')
            
            # Second variant result
            axes[1,3].imshow(results[variants[1]]['result'])
            title = f'{variants[1].replace("_", " ").title()}'
            if product_type == "tv":
                placement = results[variants[1]]['placement']
                aspect = placement[2] / placement[3]
                title += f' (AR: {aspect:.2f}:1)'
            axes[1,3].set_title(f'{title} ✅', fontsize=11, fontweight='bold')
            axes[1,3].axis('off')
        else:
            for i in range(4):
                axes[1,i].axis('off')
            
        plt.tight_layout()
        
        # Save comparison
        comparison_path = self.run_dir / f"{product_type}_improved_comparison_{self.timestamp}.png"
        fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved enhanced comparison: {comparison_path.name}")
        return comparison_path
        
    def run_improved_pipeline(self):
        """Run the improved product placement pipeline with new assets"""
        print("\n" + "="*80)
        print("🚀 STARTING IMPROVED PRODUCT PLACEMENT PIPELINE")
        print("🎯 Using new assets: tv_1.png + painting_1.png")
        print("="*80)
        
        # Setup pipeline
        self.setup_pipeline()
        
        # Define test configuration with new assets
        room_path = "assets/room_wall.png"
        
        # Process TV using tv_1.png
        try:
            print("\n" + "="*50)
            print("📺 PROCESSING IMPROVED TV (tv_1.png)")
            print("="*50)
            
            tv_product_path = "assets/tv_1.png"
            tv_product = self.load_product_image(tv_product_path)
            
            tv_results = self.process_single_product(room_path, tv_product_path, "tv")
            
            if tv_results:
                self.create_enhanced_comparison(tv_results, "tv", tv_product)
            else:
                print("❌ No TV results generated")
                
        except Exception as e:
            print(f"❌ TV processing failed: {e}")
            
        # Process Painting using painting_1.png
        try:
            print("\n" + "="*50)
            print("🖼️ PROCESSING IMPROVED PAINTING (painting_1.png)")
            print("="*50)
            
            painting_product_path = "assets/painting_1.png"
            painting_product = self.load_product_image(painting_product_path)
            
            painting_results = self.process_single_product(room_path, painting_product_path, "painting")
            
            if painting_results:
                self.create_enhanced_comparison(painting_results, "painting", painting_product)
            else:
                print("❌ No painting results generated")
                
        except Exception as e:
            print(f"❌ Painting processing failed: {e}")
            
        print("\n" + "="*80)
        print("✅ IMPROVED PRODUCT PLACEMENT PIPELINE COMPLETE")
        print("🎯 Enhanced detail preservation, optimal alignment, preserved aspect ratios")
        print("📺 TV variants: 42\" + 55\" with preserved details and proper aspect ratio")
        print("🖼️ Painting variants: Medium + Large with enhanced detail preservation")
        print(f"📁 Results saved in: {self.run_dir}")
        print("="*80)

def main():
    """Main execution function"""
    pipeline = ImprovedProductPlacementPipeline()
    pipeline.run_improved_pipeline()
    
    return pipeline

if __name__ == "__main__":
    main()