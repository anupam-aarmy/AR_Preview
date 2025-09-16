# Windows GPU VM Setup Guide (Azure & GCP)

Purpose: Provision a Windows Server VM with NVIDIA GPU, install CUDA-enabled PyTorch + project dependencies for the AR_Preview generative pipeline.

## 1. Recommended Specs
| Component | Minimum | Preferred |
|-----------|---------|----------|
| GPU | NVIDIA T4 (16GB shared) / L4 (24GB) | RTX A4000 / A10 / A100 40GB if available |
| vCPU | 4 | 8 |
| RAM | 16 GB | 32 GB |
| Disk (SSD) | 150 GB | 200 GB (room for models + cache) |
| OS | Windows Server 2022 Datacenter | Same |

---
## 2. Azure Deployment Steps
### 2.1 Portal Deployment
1. Azure Portal → Create Resource → Virtual Machine.
2. Basics:
   - Subscription / Resource Group: choose or create.
   - VM Name: `ar-preview-gpu-win`.
   - Region: pick region supporting your GPU size (e.g. `East US`, `West Europe`).
   - Image: `Windows Server 2022 Datacenter: Azure Edition`.
   - Size: Click “See all sizes” → Filter "GPU" → choose `Standard_NC4as_T4_v3` (T4) or `Standard_NV12ads_A10_v5` (A10). Note pricing.
   - Username: `dev` (example). Password (complex) or SSH key (if Win32 OpenSSH).
3. Disks:
   - OS Disk: Premium SSD 128–200 GB.
   - Enable temp disk if offered (not for model storage).
4. Networking:
   - Create VNet or reuse existing.
   - Public IP: Yes (static). Inbound ports: RDP (3389) only.
5. Management:
   - Enable Boot diagnostics.
   - Auto-shutdown: Enabled (e.g. 20:00) to save cost.
6. Review + Create → Deploy.

### 2.2 Enable GPU Driver
After VM is provisioned:
1. RDP into VM.
2. Open PowerShell as Administrator and run:
   ```powershell
   # Install NVIDIA drivers (Azure VM Extension)
   Set-ExecutionPolicy Bypass -Scope Process -Force
   Invoke-WebRequest https://aka.ms/azvmimagebuilder/nvidiadriver -OutFile driver.ps1
   .\driver.ps1
   ```
   Or use Azure Extension: VM → Extensions + applications → Add → `NvidiaGpuDriverWindows`.
3. Reboot VM.
4. Verify driver:
   ```powershell
   nvidia-smi
   ```

### 2.3 Install Toolchain & Python
```powershell
# Install Chocolatey (optional convenience)
Set-ExecutionPolicy Bypass -Scope Process -Force; \
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12; \
iwr https://community.chocolatey.org/install.ps1 -UseBasicParsing | iex

# Install Git & Python if not present
choco install git -y
choco install python --version=3.11.7 -y
refreshenv

# Clone repo
cd C:\
mkdir dev; cd dev
git clone https://github.com/anupam-aarmy/AR_Preview.git
cd AR_Preview

# Create venv
py -3.11 -m venv venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2.4 Install CUDA PyTorch + Dependencies
Determine CUDA compatibility (T4/A10 typically CUDA 12.x capable):
```powershell
# Install CUDA-enabled torch (12.1 build example)
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Core libs
pip install -r requirements.txt

# Optional performance
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
```

### 2.5 Validate Environment
```powershell
python sd_environment_test.py
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---
## 3. GCP Deployment Steps
### 3.1 Enable APIs
- In GCP Console: Enable Compute Engine API.

### 3.2 Create VM (Console)
1. Compute Engine → VM Instances → Create Instance.
2. Name: `ar-preview-gpu-win`.
3. Region/Zone: Choose GPU-supported zone (e.g. `us-central1-a`).
4. Machine Configuration:
   - Series: `G2` (L4) or older `N1`/`A2` (T4/A100) depending on availability.
   - Select machine type: start with 4–8 vCPU, 16–32GB RAM.
5. GPU:
   - Add GPU → `NVIDIA T4` (1) or `NVIDIA L4` (1) or A100 (costly).
6. Boot Disk:
   - Change → Operating System: `Windows Server`.
   - Version: `Windows Server 2022 Datacenter`.
   - Size: 150–200 GB SSD (Balanced).
7. Access:
   - Allow RDP (TCP 3389) in firewall.
8. Create.

### 3.3 Install NVIDIA Driver
GCP provides driver installer:
1. RDP into VM after creation.
2. Open PowerShell (Admin):
   ```powershell
   Invoke-WebRequest -Uri https://storage.googleapis.com/nvidia-drivers-us-public/GRID/windows/550.54/GridDriver-WinServer2022-550.54.exe -OutFile C:\driver.exe
   Start-Process C:\driver.exe -ArgumentList "/s" -Wait
   reboot
   ```
3. Validate after reboot:
   ```powershell
   nvidia-smi
   ```

### 3.4 Install Git, Python, Repo (same as Azure)
Repeat steps from Azure §2.3 and §2.4.

---
## 4. Model Assets
SAM model download:
```powershell
python download_sam.py
```
Stable Diffusion Inpainting weights are pulled automatically by `diffusers` (cached in `%USERPROFILE%\.cache\huggingface`). Consider setting `HF_HOME` to a data disk:
```powershell
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\\hf_cache", "Machine")
```
Restart shell afterward.

---
## 5. Running Pipelines (GPU)
```powershell
venv\Scripts\activate
python main.py                # Deterministic (fast)
python generative_pipeline.py --steps 20  # After fast mode added
```
Expected GPU run time (T4, 20 steps, 768px): ~1.5–2.5 min/image (pre-optimizations).

---
## 6. Cost Optimization
| Technique | Benefit |
|-----------|---------|
| Auto-shutdown | Avoid idle GPU billing |
| Spot/Preemptible (GCP) | 60–70% cheaper (non-critical dev) |
| Lower steps in early tuning | Faster iteration |
| Downscale testing resolution | Reduces VRAM + time |

---
## 7. Troubleshooting
| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available()` False | Reinstall driver, verify GPU quota, reboot |
| OOM during generation | Reduce resolution or steps, free VRAM via closing apps |
| Slow performance | Confirm running on GPU (nvidia-smi shows python process) |
| Diffusers download timeout | Set `HF_HOME` and retry; ensure outbound HTTPS allowed |

---
## 8. Next After Provisioning
1. Pull latest branch updates.
2. Implement fast mode changes (incoming).
3. Test short 10–15 step generations.
4. Begin ControlNet depth integration.

---
This guide will evolve as Task 2 matures.
