import pandas as pd
import numpy as np
import pytest
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from pathlib import Path
from Parameta.rates_test.scripts.ratesprocessor import RateProcessor 


@pytest.fixture
def sample_data(tmp_path):
    """Create sample input data for testing"""

    # rates_ccy_data.csv
    ccy_data = pd.DataFrame({
        "ccy_pair": ["EURUSD", "GBPUSD", "JPYUSD"],
        "convert_price": [True, False, True],
        "conversion_factor": [2.0, 1.0, 100.0]
    })
    ccy_file = tmp_path / "rates_ccy_data.csv"
    ccy_data.to_csv(ccy_file, index=False)

    # rates_price_data.parq
    price_data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-01 10:00:00",
            "2025-01-01 10:30:00",
            "2025-01-01 11:00:00"
        ]),
        "security_id": [1, 2, 3],
        "price": [100, 200, 300],
        "ccy_pair": ["EURUSD", "GBPUSD", "JPYUSD"]
    })
    price_file = tmp_path / "rates_price_data.parq"
    price_data.to_parquet(price_file)

    # rates_spot_rate_data.parq
    spot_data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-01 09:30:00",
            "2025-01-01 10:15:00"
        ]),
        "ccy_pair": ["EURUSD", "JPYUSD"],
        "spot_mid_rate": [1.1, 110.0]
    })
    spot_file = tmp_path / "rates_spot_rate_data.parq"
    spot_data.to_parquet(spot_file)

    return tmp_path


# ---------------- Unit Tests ---------------- #

def test_load_data(sample_data):
    processor = RateProcessor(data_dir=sample_data)
    processor.load_data()
    assert "ccy_pair" in processor.ccy_data.columns
    assert pd.api.types.is_datetime64_any_dtype(processor.price_data["timestamp"])


def test_find_spot_rates(sample_data):
    processor = RateProcessor(data_dir=sample_data)
    processor.load_data()
    merged = processor.find_spot_rates()
    assert "spot_mid_rate" in merged.columns
    eurusd_row = merged[merged["ccy_pair"] == "EURUSD"].iloc[0]
    # EURUSD should get spot rate from 09:30 (within 1 hour window)
    assert eurusd_row["spot_mid_rate"] == 1.1


def test_calculate_new_prices(sample_data):
    processor = RateProcessor(data_dir=sample_data)
    processor.load_data()
    processor.calculate_new_prices()
    result = processor.result

    eurusd_row = result[result["ccy_pair"] == "EURUSD"].iloc[0]
    # Conversion required: (100 / 2.0) + 1.1 = 51.1
    assert np.isclose(eurusd_row["new_price"], 51.1)
    assert eurusd_row["conversion_status"] == "converted"

    gbpusd_row = result[result["ccy_pair"] == "GBPUSD"].iloc[0]
    # No conversion → price unchanged
    assert gbpusd_row["new_price"] == 200
    assert gbpusd_row["conversion_status"] == "no_conversion_required"

    jpyusd_row = result[result["ccy_pair"] == "JPYUSD"].iloc[0]
    # Conversion required and has spot rate
    assert np.isclose(jpyusd_row["new_price"], (300 / 100) + 110.0)
    assert jpyusd_row["conversion_status"] == "converted"


# ---------------- Integration Test ---------------- #

def test_full_process_pipeline(sample_data):
    processor = RateProcessor(data_dir=sample_data)
    result = processor.process()

    # Check output file exists
    output_file = Path("Parameta/rates_test/results/rates_result.csv")
    assert output_file.exists()

    # Check essential columns
    assert all(col in result.columns for col in [
        "ccy_pair", "timestamp", "price", "new_price", "conversion_status"
    ])

    # Ensure all statuses are valid
    valid_statuses = {"converted", "insufficient_spot", "no_conversion_required", "unsupported_ccy"}
    assert set(result["conversion_status"].unique()).issubset(valid_statuses)
