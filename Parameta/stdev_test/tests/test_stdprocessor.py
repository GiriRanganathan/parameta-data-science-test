import pandas as pd
import numpy as np
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from Parameta.stdev_test.scripts.stdprocessor import RollingStandardDeviationCalculator # hypothetical class


# ---------------- Fixtures ---------------- #

@pytest.fixture
def sample_std_data(tmp_path):
    """Create sample input data for testing rolling std with various scenarios"""
    data = []
    
    # Security 1: Complete hourly data (25 hours - enough for rolling window)
    timestamps1 = pd.date_range("2025-01-01 00:00", periods=25, freq="H")
    for i, ts in enumerate(timestamps1):
        data.append({
            "timestamp": ts,
            "security_id": "SEC1",
            "price": 100 + i * 0.5 + np.random.normal(0, 0.1)
        })
    
    # Security 2: Data with a gap at hour 15
    timestamps2 = pd.date_range("2025-01-01 00:00", periods=23, freq="H").tolist()
    timestamps2.remove(pd.Timestamp("2025-01-01 15:00"))
    for i, ts in enumerate(timestamps2):
        data.append({
            "timestamp": ts,
            "security_id": "SEC2", 
            "price": 200 + i * 0.3 + np.random.normal(0, 0.05)
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['security_id', 'timestamp']).reset_index(drop=True)
    
    file_path = tmp_path / "rates_price_data.parq"
    df.to_parquet(file_path)
    
    return tmp_path, df


@pytest.fixture
def empty_data(tmp_path):
    """Create empty dataset for edge case testing"""
    df = pd.DataFrame(columns=["timestamp", "security_id", "price"])
    file_path = tmp_path / "empty_data.parq"
    df.to_parquet(file_path)
    return tmp_path, df


# ---------------- Unit Tests ---------------- #

class TestDataLoading:
    def test_load_data_success(self, sample_std_data):
        tmp_path, expected_df = sample_std_data
        processor = RollingStandardDeviationCalculator(data_dir=tmp_path)
        processor.load_data()
        assert "timestamp" in processor.price_data.columns
        assert "security_id" in processor.price_data.columns
        assert "price" in processor.price_data.columns
        assert pd.api.types.is_datetime64_any_dtype(processor.price_data["timestamp"])
        assert len(processor.price_data) > 0
    
    def test_load_empty_data(self, empty_data):
        tmp_path, _ = empty_data
        processor = RollingStandardDeviationCalculator(data_dir=tmp_path)
        processor.load_data()
        assert len(processor.price_data) == 0
    
    def test_load_nonexistent_file(self, tmp_path):
        processor = RollingStandardDeviationCalculator(data_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            processor.load_data()


class TestContiguityChecking:
    def test_is_contiguous_true_single_hour_gap(self):
        ts = pd.date_range("2025-01-01 01:00", periods=5, freq="H")
        processor = RollingStandardDeviationCalculator()
        assert processor.is_contiguous(ts)
    
    def test_is_contiguous_false_with_gap(self):
        ts = pd.to_datetime([
            "2025-01-01 01:00", 
            "2025-01-01 02:00", 
            "2025-01-01 04:00"  # missing 03:00
        ])
        processor = RollingStandardDeviationCalculator()
        assert not processor.is_contiguous(ts)


class TestRollingStdCalculation:
    def test_calculate_rolling_std_sufficient_data(self, sample_std_data):
        tmp_path, _ = sample_std_data
        processor = RollingStandardDeviationCalculator(data_dir=tmp_path)
        processor.load_data()
        result = processor.calculate_rolling_std(window=20)
        sec1_data = result[result["security_id"] == "SEC1"]
        populated_rows = sec1_data.dropna(subset=["rolling_std"])
        assert len(populated_rows) > 0
        assert all(populated_rows["rolling_std"] >= 0)
    
    def test_calculate_rolling_std_gap_handling(self, sample_std_data):
        tmp_path, _ = sample_std_data
        processor = RollingStandardDeviationCalculator(data_dir=tmp_path)
        processor.load_data()
        result = processor.calculate_rolling_std(window=20)
        sec2_data = result[result["security_id"] == "SEC2"]
        row_after_gap = sec2_data[sec2_data["timestamp"] == "2025-01-01 16:00"]
        if not row_after_gap.empty:
            assert pd.isna(row_after_gap.iloc[0]["rolling_std"])
    
    def test_calculate_rolling_std_insufficient_data(self, sample_std_data):
        tmp_path, _ = sample_std_data
        processor = RollingStandardDeviationCalculator(data_dir=tmp_path)
        processor.load_data()
        result = processor.calculate_rolling_std(window=20)
        sec2_data = result[result["security_id"] == "SEC2"]
        # Should still have NaNs until enough contiguous rows exist
        assert result["rolling_std"].isna().sum() > 0


# ---------------- Integration Tests ---------------- #

class TestIntegration:
    def test_full_process_pipeline(self, sample_std_data):
        tmp_path, _ = sample_std_data
        processor = RollingStandardDeviationCalculator(data_dir=tmp_path)
        output_file = Path("Parameta/rates_test/results/std_result.csv")
        if output_file.exists():
            output_file.unlink()
        
        result = processor.process_pipeline(window=20)
        assert output_file.exists()
        saved_data = pd.read_csv(output_file)
        required_columns = ["security_id", "timestamp", "price", "rolling_std"]
        assert all(col in result.columns for col in required_columns)
        assert all(col in saved_data.columns for col in required_columns)
        assert len(result) == len(saved_data)
    
    def test_contiguity_validation_in_pipeline(self, sample_std_data):
        tmp_path, _ = sample_std_data
        processor = RollingStandardDeviationCalculator(data_dir=tmp_path)
        result = processor.process_pipeline(window=20)
        populated_rows = result.dropna(subset=["rolling_std"])
        for _, row in populated_rows.iterrows():
            subset = result[
                (result["security_id"] == row["security_id"]) &
                (result["timestamp"] <= row["timestamp"])
            ].tail(20)
            assert processor.is_contiguous(subset["timestamp"])
