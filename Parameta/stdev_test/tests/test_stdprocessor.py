import pandas as pd
import numpy as np
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from Parameta.stdev_test.scripts.stdprocessor import RollingStandardDeviationCalculator


# ---------------- Fixtures ---------------- #

@pytest.fixture
def sample_price_data(tmp_path):
    """Create sample input data for testing rolling std with various scenarios"""
    data = []
    
    # Security 1: Complete hourly data (25 hours - enough for rolling window)
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
        df = calculator.load_data(str(file_path))
        
        assert "snap_time" in df.columns
        assert "security_id" in df.columns
        assert "bid" in df.columns
        assert "mid" in df.columns
        assert "ask" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["snap_time"])
        assert len(df) > 0
        assert df.equals(df.sort_values(['security_id', 'snap_time']).reset_index(drop=True))
    
    def test_load_empty_data(self, empty_data):
        file_path, _ = empty_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(str(file_path))
        assert len(df) == 0
        assert list(df.columns) == ["snap_time", "security_id", "bid", "mid", "ask"]
    
    def test_load_nonexistent_file(self):
        calculator = RollingStandardDeviationCalculator(window_size=20)
        with pytest.raises(FileNotFoundError):
            calculator.load_data("nonexistent_file.parq")


class TestContiguousSequences:
    def test_identify_contiguous_sequences_no_gaps(self, sample_price_data):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(str(file_path))
        
        # Test with SEC1 which has no gaps
        sec1_data = df[df['security_id'] == 'SEC1'].copy()
        result = calculator._identify_contiguous_sequences(sec1_data)
        
        assert 'group_id' in result.columns
        assert result['group_id'].nunique() == 1  # Should be one contiguous group
    
    def test_identify_contiguous_sequences_with_gaps(self, sample_price_data):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(str(file_path))
        
        # Test with SEC2 which has a gap
        sec2_data = df[df['security_id'] == 'SEC2'].copy()
        result = calculator._identify_contiguous_sequences(sec2_data)
        
        assert 'group_id' in result.columns
        assert result['group_id'].nunique() > 1  # Should have multiple groups due to gap


class TestRollingStdCalculation:
    def test_calculate_rolling_std_sufficient_data(self, sample_price_data):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(str(file_path))
        
        result = calculator.calculate_rolling_std(df)
        
        # Check structure
        expected_columns = ['snap_time', 'security_id', 'bid_std', 'mid_std', 'ask_std']
        assert all(col in result.columns for col in expected_columns)
        
        # Check that we have results for securities with sufficient data
        sec1_results = result[result['security_id'] == 'SEC1']
        assert len(sec1_results) > 0
        
        # Check that standard deviations are non-negative
        std_columns = ['bid_std', 'mid_std', 'ask_std']
        for col in std_columns:
            non_null_values = result[col].dropna()
            if len(non_null_values) > 0:
                assert all(non_null_values >= 0)
    
    def test_calculate_rolling_std_insufficient_data(self, sample_price_data):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(str(file_path))
        
        result = calculator.calculate_rolling_std(df)
        
        # SEC3 has only 15 hours of data, so should have no results
        sec3_results = result[result['security_id'] == 'SEC3']
        assert len(sec3_results) == 0
    
    def test_calculate_rolling_std_custom_time_range(self, sample_price_data):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(str(file_path))
        
        start_time = "2021-11-20 10:00:00"
        end_time = "2021-11-20 15:00:00"
        
        result = calculator.calculate_rolling_std(df, start_time, end_time)
        
        # Check that results are within specified time range
        if len(result) > 0:
            assert result['snap_time'].min() >= pd.to_datetime(start_time)
            assert result['snap_time'].max() <= pd.to_datetime(end_time)
    
    def test_calculate_rolling_std_empty_data(self, empty_data):
        file_path, _ = empty_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(str(file_path))
        
        result = calculator.calculate_rolling_std(df)
        
        expected_columns = ['snap_time', 'security_id', 'bid_std', 'mid_std', 'ask_std']
        assert list(result.columns) == expected_columns
        assert len(result) == 0


class TestSaveResults:
    def test_save_results(self, sample_price_data, tmp_path):
        file_path, _ = sample_price_data
        calculator = RollingStandardDeviationCalculator(window_size=20)
        df = calculator.load_data(str(file_path))
        result = calculator.calculate_rolling_std(df)
        
        output_file = tmp_path / "test_output.csv"
        calculator.save_results(result, str(output_file))
        
        assert output_file.exists()
        
        # Verify saved data
        saved_df = pd.read_csv(output_file)
        assert len(saved_df) == len(result)
        assert 'snap_time' in saved_df.columns
        
        # Check that timestamp format is string
        assert saved_df['snap_time'].dtype == 'object'


# ---------------- Integration Tests ---------------- #

class TestIntegration:
    def test_full_process_pipeline(self, sample_price_data, tmp_path):
        file_path, _ = sample_price_data
        output_file = tmp_path / "rolling_std_results.csv"
        
        calculator = RollingStandardDeviationCalculator(window_size=20)
        result = calculator.process_file(
            input_file=str(file_path),
            output_file=str(output_file),
            start_time="2021-11-20 00:00:00",
            end_time="2021-11-21 00:00:00"
        )
        
        # Check that output file was created
        assert output_file.exists()
        
        # Check return value structure
        expected_columns = ['snap_time', 'security_id', 'bid_std', 'mid_std', 'ask_std']
        assert all(col in result.columns for col in expected_columns)
        
        # Check that saved file matches return value
        saved_df = pd.read_csv(output_file)
        assert len(result) == len(saved_df)
    
    def test_process_file_with_gaps_handling(self, sample_price_data, tmp_path):
        file_path, _ = sample_price_data
        output_file = tmp_path / "results_with_gaps.csv"
        
        calculator = RollingStandardDeviationCalculator(window_size=20)
        result = calculator.process_file(
            input_file=str(file_path),
            output_file=str(output_file)
        )
        
        # Should handle gaps gracefully and produce some results
        # (at least for SEC1 which has complete data)
        sec1_results = result[result['security_id'] == 'SEC1']
        assert len(sec1_results) > 0
        
        # Verify output file structure
        saved_df = pd.read_csv(output_file)
        required_columns = ['snap_time', 'security_id', 'bid_std', 'mid_std', 'ask_std']
        assert all(col in saved_df.columns for col in required_columns)