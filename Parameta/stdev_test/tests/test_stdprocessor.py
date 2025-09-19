import pandas as pd
import numpy as np
import pytest
import sys
import os
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from Parameta.stdev_test.scripts.stdprocessor import RollingStandardDeviationCalculator

@pytest.fixture
def sample_price_data(tmp_path):
    """Sample input data for testing rolling std with various scenarios"""
    data = []
    
    # Security 1: Complete hourly data (25 hours)
    timestamps1 = pd.date_range("2021-11-20 00:00", periods=25, freq="H")
    for i, ts in enumerate(timestamps1):
        base_price = 100 + i * 0.5
        data.append({
            "snap_time": ts,
            "security_id": "SEC1",
            "bid": base_price - 0.01 + np.random.normal(0, 0.1),
            "mid": base_price + np.random.normal(0, 0.1),
            "ask": base_price + 0.01 + np.random.normal(0, 0.1)
        })
    
    # Security 2: Data with a gap at hour 15 (will break contiguity)
    timestamps2 = pd.date_range("2021-11-20 00:00", periods=25, freq="H").tolist()
    timestamps2.remove(pd.Timestamp("2021-11-20 15:00"))  # Create gap
    for i, ts in enumerate(timestamps2):
        base_price = 200 + i * 0.3
        data.append({
            "snap_time": ts,
            "security_id": "SEC2",
            "bid": base_price - 0.01 + np.random.normal(0, 0.05),
            "mid": base_price + np.random.normal(0, 0.05),
            "ask": base_price + 0.01 + np.random.normal(0, 0.05)
        })
    
    # Security 3: Insufficient data (only 15 hours - less than window size)
    timestamps3 = pd.date_range("2021-11-20 00:00", periods=15, freq="H")
    for i, ts in enumerate(timestamps3):
        base_price = 300 + i * 0.2
        data.append({
            "snap_time": ts,
            "security_id": "SEC3",
            "bid": base_price - 0.01 + np.random.normal(0, 0.08),
            "mid": base_price + np.random.normal(0, 0.08),
            "ask": base_price + 0.01 + np.random.normal(0, 0.08)
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['security_id', 'snap_time']).reset_index(drop=True)
    
    file_path = tmp_path / "stdev_price_data.parq"
    df.to_parquet(file_path)
    
    return file_path, df


@pytest.fixture
def empty_data(tmp_path):
    """Create empty dataset for edge case testing"""
    df = pd.DataFrame(columns=["snap_time", "security_id", "bid", "mid", "ask"])
    file_path = tmp_path / "empty_data.parq"
    df.to_parquet(file_path)
    return file_path, df


# ---------------- Unit Tests ---------------- #

class TestDataLoading:
    def test_load_data_success(self, sample_price_data):
        file_path, expected_df = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(file_path)
        
        assert "snap_time" in df.columns
        assert "security_id" in df.columns
        assert "bid" in df.columns
        assert "mid" in df.columns
        assert "ask" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["snap_time"])
        assert len(df) > 0
        # Original logic does not reorder timestamps
        assert df.equals(df.sort_values(['security_id', 'snap_time']).reset_index(drop=True))
    
    def test_load_empty_data(self, empty_data):
        file_path, _ = empty_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(file_path)
        assert len(df) == 0
        assert list(df.columns) == ["snap_time", "security_id", "bid", "mid", "ask"]
    
    def test_load_nonexistent_file(self):
        calculator = RollingStandardDeviationCalculator(window_size=20)
        with pytest.raises(FileNotFoundError):
            calculator.load_data("nonexistent_file.parq")


class TestRollingStdCalculation:
    def test_calculate_rolling_std_sufficient_data(self, sample_price_data):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(file_path)
        
        result = calculator.calculate_rolling_std(df)
        
        # Check structure
        expected_columns = ['snap_time', 'security_id', 'bid_std', 'mid_std', 'ask_std']
        assert all(col in result.columns for col in expected_columns)
        
        # SEC1 has enough data
        sec1_results = result[result['security_id'] == 'SEC1']
        assert len(sec1_results) > 0
        
        # Rolling std should be non-negative for computed values
        std_columns = ['bid_std', 'mid_std', 'ask_std']
        for col in std_columns:
            non_null_values = sec1_results[col].dropna()
            if len(non_null_values) > 0:
                assert all(non_null_values >= 0)
    
    def test_calculate_rolling_std_insufficient_data(self, sample_price_data):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(file_path)
        
        result = calculator.calculate_rolling_std(df)
        
        # SEC3 has only 15 hours < window, so all std values should be NaN
        sec3_results = result[result['security_id'] == 'SEC3']
        assert len(sec3_results) > 0
        assert sec3_results[['bid_std', 'mid_std', 'ask_std']].isna().all().all()
    
    def test_calculate_rolling_std_with_gaps(self, sample_price_data):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(file_path)
        
        result = calculator.calculate_rolling_std(df)
        
        # SEC2 has gaps; ensure rolling std is NaN before window is full and forward-filled correctly
        sec2_results = result[result['security_id'] == 'SEC2']
        assert len(sec2_results) > 0
        # Check first few rows have NaNs until window fills
        first_std_values = sec2_results.iloc[:19][['bid_std', 'mid_std', 'ask_std']]
        assert first_std_values.isna().all().all()


class TestSaveResults:
    def test_save_results(self, sample_price_data, tmp_path):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(file_path)
        result = calculator.calculate_rolling_std(df)
        
        output_file = tmp_path / "test_output.csv"
        calculator.save_results(result, output_file)
        
        assert output_file.exists()
        
        # Verify saved data
        saved_df = pd.read_csv(output_file)
        assert len(saved_df) == len(result)
        assert 'snap_time' in saved_df.columns
        # snap_time should remain string in output
        assert saved_df['snap_time'].dtype == 'object'


class TestIntegration:
    def test_full_process_pipeline(self, sample_price_data, tmp_path):
        file_path, _ = sample_price_data
        output_file = tmp_path / "rolling_std_results.csv"
        
        calculator = RollingStandardDeviationCalculator(window_size=20)
        result = calculator.process_file(
            input_file=file_path,
            output_file=output_file,
            start_time="2021-11-20 00:00:00",
            end_time="2021-11-21 00:00:00"
        )
        
        assert output_file.exists()
        expected_columns = ['snap_time', 'security_id', 'bid_std', 'mid_std', 'ask_std']
        assert all(col in result.columns for col in expected_columns)
        
        # Saved file matches results
        saved_df = pd.read_csv(output_file)
        assert len(result) == len(saved_df)
