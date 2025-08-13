#!/usr/bin/env python3
"""
Test script to verify memory estimation is working correctly
"""

import torch
from train_wpo_sgm_memory_efficient import estimate_memory_requirements

def test_memory_estimation():
    """Test the memory estimation function with different scenarios"""
    
    print("Testing memory estimation function...")
    print("=" * 50)
    
    # Test case 1: Small CIFAR-10 setup
    print("\n1. Small CIFAR-10 setup:")
    print("   - 50 centers, 3072 dimensions, batch_size=8")
    mem1 = estimate_memory_requirements(50, 3072, 8)
    
    # Test case 2: Medium setup
    print("\n2. Medium setup:")
    print("   - 200 centers, 3072 dimensions, batch_size=8")
    mem2 = estimate_memory_requirements(200, 3072, 8)
    
    # Test case 3: Large setup
    print("\n3. Large setup:")
    print("   - 500 centers, 3072 dimensions, batch_size=8")
    mem3 = estimate_memory_requirements(500, 3072, 8)
    
    # Test case 4: Very large setup
    print("\n4. Very large setup:")
    print("   - 1000 centers, 3072 dimensions, batch_size=8")
    mem4 = estimate_memory_requirements(1000, 3072, 8)
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  50 centers:   {mem1:.2f} GB")
    print(f"  200 centers:  {mem2:.2f} GB")
    print(f"  500 centers:  {mem3:.2f} GB")
    print(f"  1000 centers: {mem4:.2f} GB")
    
    # Check if estimates make sense
    print("\nSanity checks:")
    print(f"  Memory scales roughly linearly with centers: {mem2/mem1:.1f}x for 4x centers")
    print(f"  Memory scales roughly quadratically: {mem4/mem1:.1f}x for 20x centers")
    
    # Compare with actual GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nYour GPU has {gpu_memory:.1f} GB memory")
        print(f"Recommendations:")
        if mem1 < gpu_memory * 0.8:
            print(f"  ✅ 50 centers should fit comfortably")
        else:
            print(f"  ❌ Even 50 centers might be too much")
            
        if mem2 < gpu_memory * 0.8:
            print(f"  ✅ 200 centers should fit")
        else:
            print(f"  ⚠️  200 centers will need chunking or model parallel")
            
        if mem3 < gpu_memory * 0.8:
            print(f"  ✅ 500 centers should fit")
        else:
            print(f"  ❌ 500 centers will definitely need chunking")
            
        if mem4 < gpu_memory * 0.8:
            print(f"  ✅ 1000 centers should fit")
        else:
            print(f"  ❌ 1000 centers will need aggressive chunking or gradient accumulation")

if __name__ == "__main__":
    test_memory_estimation()
