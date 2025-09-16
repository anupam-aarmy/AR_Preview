# AI Candidate Test Tasks – Module 2 (Single Wall Fitting)

This document outlines two technical tasks designed to evaluate an AI candidate's skills relevant to Module 2 (Single Wall Fitting with TVs, frames, and paintings). The tasks focus on wall segmentation, product placement, and enhancement of Stable Diffusion models for realistic AR try-on experiences.

## Task 1 – Wall Segmentation & Product Placement (AI Fallback)

Objective: Test ability to segment walls from an image and realistically place an object (TV/painting).

Description:

\- Input: A room photo (with visible wall).  
\- Output:  
1\. Detect and mask the wall region using an AI vision model (SAM / U-Net / custom segmentation).  
2\. Place a given TV/painting frame image (PNG with transparency) onto the wall with correct scaling & alignment.  
3\. Ensure the final result blends well (no overlap on furniture, lighting adjusted if possible).

Evaluation Criteria:

\- Accuracy of wall segmentation.  
\- Proper scaling & perspective of TV/painting on wall.  
\- Visual realism (natural placement, not copy-pasted).  
\- Code quality (clean Python or PyTorch/TensorFlow pipeline).

## Task 2 – Stable Diffusion Fine-Tuning for Product Placement

Objective: Test candidate’s ability to adapt Stability models (ControlNet/DreamBooth) for AR use cases.

Description:

\- Given:  
\- A set of product images (e.g., 2–3 TV/painting variants).  
\- A room photo (wall image).  
\- Use Stable Diffusion + ControlNet (depth or inpaint) to:  
1\. Generate an image where the chosen product is realistically placed on the wall.  
2\. Show at least two product size variations (e.g., 42” vs 55” TV).

Evaluation Criteria:

\- Ability to set up Stable Diffusion pipeline (Hugging Face / Diffusers / Automatic1111).  
\- Correct ControlNet conditioning (depth/inpainting).  
\- Output quality (alignment, scaling, shadows).  
\- Understanding of fine-tuning/custom training (even at prototype level).