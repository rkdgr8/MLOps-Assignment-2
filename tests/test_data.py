import os
import pytest
from src.data_prep import split_data

def test_data_prep_dummy_creation(tmp_path):
    """
    Test that data_prep correctly splits dummy distributions when real data is missing.
    Matches the M3 requirement for testing a data preprocessing utility.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    
    os.makedirs(raw_dir / "NORMAL")
    os.makedirs(raw_dir / "PNEUMONIA")
    
    # Create 10 dummy files in PNEUMONIA
    for i in range(10):
        with open(raw_dir / "PNEUMONIA" / f"dummy_{i}.txt", "w") as f:
            f.write(f"test {i}")
            
    split_data(raw_dir, processed_dir, split_ratio=(0.8, 0.1, 0.1))
    
    # 80% train, 10% val, 10% test 
    assert len(os.listdir(processed_dir / "train" / "PNEUMONIA")) == 8
    assert len(os.listdir(processed_dir / "val" / "PNEUMONIA")) == 1
    assert len(os.listdir(processed_dir / "test" / "PNEUMONIA")) == 1
