import numpy as np
import sys
import json
from pathlib import Path

def print_npz_contents(file_path):
    """
    Load and print the contents of an NPZ file in a format suitable for an LLM to read.
    """
    # Load the NPZ file
    data = np.load(file_path)
    
    # Get list of arrays in the file
    array_keys = list(data.keys())
    
    # Print summary info
    print(f"NPZ File: {Path(file_path).name}")
    print(f"Contains {len(array_keys)} arrays:")
    
    # Print details for each array
    for key in array_keys:
        array = data[key]
        print(f"\nArray: {key}")
        print(f"  Shape: {array.shape}")
        print(f"  Data Type: {array.dtype}")
        
        # For small arrays, print the full content
        if array.size <= 100:
            print(f"  Data: {array.tolist()}")
        else:
            # For larger arrays, print summary statistics and a sample
            print(f"  Summary Statistics:")
            try:
                print(f"    Min: {array.min()}")
                print(f"    Max: {array.max()}")
                print(f"    Mean: {array.mean()}")
                print(f"    Standard Deviation: {array.std()}")
            except (TypeError, ValueError):
                print("    Cannot compute statistics for this data type")
            
            # Print a small sample (first and last few elements)
            flat_array = array.flatten()
            sample_size = min(5, len(flat_array))
            print(f"  First {sample_size} elements: {flat_array[:sample_size].tolist()}")
            print(f"  Last {sample_size} elements: {flat_array[-sample_size:].tolist()}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_npz.py <path_to_npz_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    print_npz_contents(file_path) 