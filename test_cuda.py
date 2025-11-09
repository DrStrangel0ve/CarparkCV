"""
Quick CUDA availability test script.
Run this to verify GPU setup before running CV_Vehicle_tracking.py
"""

import torch
import sys

print("=" * 60)
print("CUDA / GPU AVAILABILITY TEST")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print()

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print()
    
    # Print info for each GPU
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        
        # Get memory info
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)  # Convert to GB
        print(f"  Total memory: {total_memory:.2f} GB")
    
    print()
    print("✓ GPU setup looks good! Your video processing will run on GPU.")
    print()
    
    # Test a quick tensor operation
    print("Running quick GPU tensor test...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("✓ GPU tensor operation successful!")
else:
    print()
    print("✗ CUDA is not available. Your system will use CPU.")
    print()
    print("To enable CUDA:")
    print("1. Verify you have an NVIDIA GPU")
    print("2. Install/update NVIDIA drivers")
    print("3. Reinstall PyTorch with CUDA support from pytorch.org:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("   (Replace 'cu121' with your CUDA version, e.g., cu118, cu117)")
    print()

print("=" * 60)
