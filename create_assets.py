#!/usr/bin/env python3
"""
Create clean transparent PNG assets for testing
"""
import cv2
import numpy as np
from pathlib import Path

def create_tv_asset():
    """Create a clean TV asset with proper transparency"""
    # Create a simple TV shape
    width, height = 400, 240  # 16:10 aspect ratio like many TVs
    
    # Create RGBA image (with alpha channel)
    tv_image = np.zeros((height, width, 4), dtype=np.uint8)
    
    # TV screen (dark blue/purple gradient)
    screen_color1 = [60, 40, 80, 255]   # Dark purple
    screen_color2 = [120, 80, 160, 255] # Lighter purple
    
    # Create gradient for screen
    for y in range(height):
        ratio = y / height
        for x in range(width):
            for c in range(3):
                tv_image[y, x, c] = int(screen_color1[c] * (1 - ratio) + screen_color2[c] * ratio)
            tv_image[y, x, 3] = 255  # Full opacity
    
    # Add TV frame (black border)
    frame_thickness = 8
    tv_image[:frame_thickness, :] = [0, 0, 0, 255]  # Top
    tv_image[-frame_thickness:, :] = [0, 0, 0, 255]  # Bottom
    tv_image[:, :frame_thickness] = [0, 0, 0, 255]  # Left
    tv_image[:, -frame_thickness:] = [0, 0, 0, 255]  # Right
    
    # Add a small logo/brand area
    logo_y, logo_x = height - 25, width - 60
    tv_image[logo_y:logo_y+15, logo_x:logo_x+50] = [200, 200, 200, 255]
    
    return tv_image

def create_paintings_asset():
    """Create abstract paintings with proper transparency"""
    # Create 4 panels arrangement
    panel_w, panel_h = 80, 120
    gap = 10
    total_w = 4 * panel_w + 3 * gap
    total_h = panel_h
    
    # Create RGBA image
    paintings = np.zeros((total_h, total_w, 4), dtype=np.uint8)
    
    # Define colors for abstract art
    colors = [
        [100, 200, 150, 255],  # Teal
        [200, 100, 150, 255],  # Pink
        [150, 180, 200, 255],  # Light blue
        [180, 150, 100, 255],  # Beige
    ]
    
    # Create 4 panels
    for i in range(4):
        x_start = i * (panel_w + gap)
        x_end = x_start + panel_w
        
        # Fill panel with base color
        base_color = colors[i]
        paintings[:, x_start:x_end] = base_color
        
        # Add some abstract shapes
        # Circles
        center_x = x_start + panel_w // 2
        center_y = panel_h // 2
        cv2.circle(paintings, (center_x, center_y), 20, 
                  [base_color[0] + 30, base_color[1] - 30, base_color[2] + 20, 255], -1)
        
        # Random brush strokes
        for j in range(5):
            y1, y2 = np.random.randint(10, panel_h-10, 2)
            x1 = x_start + np.random.randint(10, panel_w-10)
            x2 = x_start + np.random.randint(10, panel_w-10)
            stroke_color = [c + np.random.randint(-40, 40) for c in base_color[:3]] + [255]
            stroke_color = [max(0, min(255, c)) for c in stroke_color]
            cv2.line(paintings, (x1, y1), (x2, y2), stroke_color, 3)
        
        # Add frame (dark brown)
        frame_color = [40, 30, 20, 255]
        frame_thick = 3
        # Top and bottom frame
        paintings[:frame_thick, x_start:x_end] = frame_color
        paintings[-frame_thick:, x_start:x_end] = frame_color
        # Left and right frame
        paintings[:, x_start:x_start+frame_thick] = frame_color
        paintings[:, x_end-frame_thick:x_end] = frame_color
    
    return paintings

def main():
    print("🎨 Creating clean transparent PNG assets...")
    
    # Create assets directory if it doesn't exist
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Backup original assets
    backup_dir = assets_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    
    # Backup existing files
    for asset_file in ["prod_1_paintings.png", "prod_2_tv.png"]:
        if (assets_dir / asset_file).exists():
            import shutil
            shutil.copy2(assets_dir / asset_file, backup_dir / f"original_{asset_file}")
            print(f"📦 Backed up original {asset_file}")
    
    # Create new TV asset
    print("📺 Creating TV asset...")
    tv_asset = create_tv_asset()
    cv2.imwrite(str(assets_dir / "prod_2_tv.png"), tv_asset)
    
    # Create new paintings asset
    print("🖼️ Creating paintings asset...")
    paintings_asset = create_paintings_asset()
    cv2.imwrite(str(assets_dir / "prod_1_paintings.png"), paintings_asset)
    
    print("✅ Clean transparent PNG assets created!")
    print(f"📁 New assets saved in: {assets_dir}")
    print(f"💾 Original assets backed up in: {backup_dir}")
    
    # Verify the new assets
    for asset_name, asset_path in [("TV", "prod_2_tv.png"), ("Paintings", "prod_1_paintings.png")]:
        img = cv2.imread(str(assets_dir / asset_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            print(f"✓ {asset_name}: {img.shape} - {'Has alpha channel' if img.shape[2] == 4 else 'No alpha channel'}")
        else:
            print(f"✗ Failed to load {asset_name}")

if __name__ == "__main__":
    main()