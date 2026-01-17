"""
Utility functions for grouping videos by family to prevent data leakage.

A video family consists of:
- Original video: "Backside-8897-2.mov"
- Flipped version: "flipped_Backside-8897-2.mov"
- Augmented version: "augmented_Backside-8897-2.mov"

When testing on a video, all family members must be excluded from training.
"""

import pandas as pd
from typing import Dict, List, Set
import re


def get_base_name(original_name: str) -> str:
    """
    Extract base video name by removing augmentation prefixes.
    
    Args:
        original_name: Video filename from metadata (e.g., "flipped_Backside-8897-2.mov")
    
    Returns:
        Base name without prefix (e.g., "Backside-8897-2.mov")
    
    Examples:
        >>> get_base_name("flipped_Backside-8897-2.mov")
        'Backside-8897-2.mov'
        >>> get_base_name("augmented_Side-6088-2.mov")
        'Side-6088-2.mov'
        >>> get_base_name("Backside-8897-2.mov")
        'Backside-8897-2.mov'
    """
    # Remove "flipped_" or "augmented_" prefix if present
    name = str(original_name).strip()
    
    # Match common prefixes
    prefixes = ['flipped_', 'augmented_']
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    
    return name


def group_videos_by_family(metadata_df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Group video IDs by their base name (family).
    
    Args:
        metadata_df: DataFrame with columns 'id' and 'original_name'
    
    Returns:
        Dictionary mapping base_name -> list of video IDs
        Example: {'Backside-8897-2.mov': [1, 51, 52]}
    """
    families = {}
    
    for _, row in metadata_df.iterrows():
        video_id = row['id']
        original_name = row['original_name']
        base_name = get_base_name(original_name)
        
        if base_name not in families:
            families[base_name] = []
        families[base_name].append(video_id)
    
    return families


def get_family_members(video_id: int, metadata_df: pd.DataFrame) -> Set[int]:
    """
    Get all video IDs in the same family as the given video.
    
    Args:
        video_id: ID of the video to find family for
        metadata_df: DataFrame with columns 'id' and 'original_name'
    
    Returns:
        Set of video IDs including the input video and all its family members
    """
    # Find the original name of this video
    row = metadata_df[metadata_df['id'] == video_id]
    if row.empty:
        return {video_id}
    
    original_name = row.iloc[0]['original_name']
    base_name = get_base_name(original_name)
    
    # Find all videos with the same base name
    families = group_videos_by_family(metadata_df)
    return set(families.get(base_name, [video_id]))


def get_train_ids_excluding_family(
    all_ids: List[int],
    test_id: int,
    metadata_df: pd.DataFrame
) -> List[int]:
    """
    Get training IDs excluding the test video and all its family members.
    
    Args:
        all_ids: List of all available video IDs in this view/split
        test_id: ID of the video being used for testing
        metadata_df: DataFrame with metadata
    
    Returns:
        List of IDs safe to use for training (no data leakage)
    """
    # Get all family members of the test video
    family_to_exclude = get_family_members(test_id, metadata_df)
    
    # Return all IDs except those in the family
    train_ids = [vid for vid in all_ids if vid not in family_to_exclude]
    
    return train_ids


def print_family_statistics(metadata_df: pd.DataFrame):
    """Print statistics about video families in the dataset."""
    families = group_videos_by_family(metadata_df)
    
    print("\n" + "="*60)
    print("  VIDEO FAMILY STATISTICS")
    print("="*60)
    print(f"Total videos: {len(metadata_df)}")
    print(f"Total families: {len(families)}")
    
    # Count family sizes
    family_sizes = {}
    for base_name, members in families.items():
        size = len(members)
        family_sizes[size] = family_sizes.get(size, 0) + 1
    
    print("\nFamily size distribution:")
    for size in sorted(family_sizes.keys()):
        count = family_sizes[size]
        print(f"  {size} member(s): {count} families")
    
    # Show a few example families
    print("\nExample families (first 3):")
    for i, (base_name, members) in enumerate(list(families.items())[:3]):
        print(f"  {i+1}. {base_name}")
        print(f"     Members: {members}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the utilities
    df = pd.read_csv("video_metadata.csv")
    
    print_family_statistics(df)
    
    # Test specific family
    test_id = 1
    family = get_family_members(test_id, df)
    print(f"\nFamily of video {test_id}: {family}")
    
    # Test training set exclusion
    all_ids = df['id'].tolist()
    train_ids = get_train_ids_excluding_family(all_ids[:10], test_id, df)
    print(f"\nOriginal IDs: {all_ids[:10]}")
    print(f"Training IDs (excluding family of {test_id}): {train_ids}")
