"""
Data structure inspection script for EEG data.

Run as a script:
    python scripts/data_inspector.py data/raw
"""

import os
from pathlib import Path
import sys


def inspect_data(data_dir):
    """Inspect directory structure and file information."""
    data_dir = Path(data_dir)
    
    print(f"\n{'='*50}")
    print(f"DATA STRUCTURE INSPECTION")
    print(f"{'='*50}")
    print(f"Directory: {data_dir.resolve()}\n")
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # File count and sizes
    total_size = 0
    file_counts = {}
    extensions = ['edf', 'csv', 'npy', 'pkl', 'txt']
    
    print("File Summary:")
    print("-" * 50)
    for ext in extensions:
        files = list(data_dir.glob(f"**/*.{ext}"))
        if files:
            size = sum(f.stat().st_size for f in files)
            file_counts[ext] = len(files)
            total_size += size
            size_gb = size / 1e9
            print(f"  .{ext}: {len(files):6d} files  {size_gb:10.2f} GB")
    
    print("-" * 50)
    total_gb = total_size / 1e9
    print(f"  Total: {sum(file_counts.values()):6d} files  {total_gb:10.2f} GB\n")
    
    # Directory structure (first 2 levels)
    print("Directory Structure:")
    print("-" * 50)
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(str(data_dir), '').count(os.sep)
        if level < 2:
            indent = '  ' * level
            rel_path = os.path.relpath(root, data_dir)
            print(f'{indent}{rel_path}/')
            
            subindent = '  ' * (level + 1)
            # Show file types and counts at this level
            if files:
                ext_summary = {}
                for f in files:
                    _, ext = os.path.splitext(f)
                    ext = ext.lstrip('.')
                    ext_summary[ext] = ext_summary.get(ext, 0) + 1
                
                for ext, count in sorted(ext_summary.items()):
                    print(f'{subindent}[{count}x .{ext}]')
    
    print("\n" + f"{'='*50}\n")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    inspect_data(data_dir)
