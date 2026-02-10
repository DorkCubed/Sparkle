#!/usr/bin/env python3
"""
Script to create a manifest with 100 randomly sampled files from manifest_5000.json
"""

import json
import random
import os
from pathlib import Path


def sample_manifest_files(input_path, output_path, num_samples=100, seed=42):
    """
    Sample a specified number of files from a manifest and save to new manifest.

    Args:
        input_path: Path to input manifest file
        output_path: Path to save the sampled manifest
        num_samples: Number of files to sample (default: 100)
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Load the original manifest
    print(f"Loading manifest from: {input_path}")
    with open(input_path, "r") as f:
        manifest_data = json.load(f)

    total_files = len(manifest_data)
    print(f"Total files in original manifest: {total_files}")

    if total_files <= num_samples:
        print(
            f"Warning: Original manifest has {total_files} files, which is less than or equal to {num_samples}. Using all files."
        )
        sampled_data = manifest_data
    else:
        # Randomly sample files
        sampled_data = random.sample(manifest_data, num_samples)
        print(f"Randomly sampled {num_samples} files from {total_files} total files")

    # Save the sampled manifest
    print(f"Saving sampled manifest to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(sampled_data, f, indent=2)

    print(f"Successfully created manifest with {len(sampled_data)} files")

    # Show some sample entries
    print("\nSample entries:")
    for i, entry in enumerate(sampled_data[:3]):
        print(f"  {i + 1}. Flow: {entry.get('flow', 'N/A')}")
        print(f"     Packet file: {entry.get('packet', 'N/A')}")

    return len(sampled_data)


def main():
    # Define paths
    project_root = Path(__file__).parent
    input_manifest = project_root / "manifest" / "manifest_5000.json"
    output_manifest = project_root / "manifest" / "manifest_100.json"

    # Verify input file exists
    if not input_manifest.exists():
        print(f"Error: Input manifest file not found: {input_manifest}")
        return 1

    # Create sampled manifest
    try:
        num_files = sample_manifest_files(
            input_path=input_manifest, output_path=output_manifest, num_samples=100
        )
        print(f"\n✅ Successfully created manifest_100.json with {num_files} files")
        print(f"📍 Location: {output_manifest}")
        print(f"🎲 Random seed: 42 (for reproducibility)")
        return 0
    except Exception as e:
        print(f"❌ Error creating sampled manifest: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
