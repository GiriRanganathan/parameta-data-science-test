import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time

# Initating Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateProcessor:
    def __init__(self, data_dir="Parameta/rates_test/data"):
        self.data_path = Path(data_dir)
        self.ccy_data = None
        self.price_data = None
        self.spot_rate_data = None
        self.result = None

    def load_data(self):
        # Load currency reference data file
        ccy_file = self.data_path / "rates_ccy_data.csv"
        self.ccy_data = pd.read_csv(ccy_file)
        logger.info(f"Loaded currency data: {len(self.ccy_data)} rows")

        # Load price data file
        price_file = self.data_path / "rates_price_data.parq"
        self.price_data = pd.read_parquet(price_file)
        self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
        logger.info(f"Loaded price data: {len(self.price_data)} rows")

        # Load spot rate data file
        spot_file = self.data_path / "rates_spot_rate_data.parq"
        self.spot_rate_data = pd.read_parquet(spot_file)
        self.spot_rate_data['timestamp'] = pd.to_datetime(self.spot_rate_data['timestamp'])
        logger.info(f"Loaded spot rate data: {len(self.spot_rate_data)} rows")

    def find_spot_rates(self):
        """
        Finding the most recent spot rate within 1 hour for each price record.
        """
        start_time = time.time()
        logger.info("Finding spot rates...")

        price_pairs = self.price_data['ccy_pair'].values
        price_times = self.price_data['timestamp'].values

        spot_pairs = self.spot_rate_data['ccy_pair'].values
        spot_times = self.spot_rate_data['timestamp'].values
        spot_rates = self.spot_rate_data['spot_mid_rate'].values

        # Pre-sorting spot data
        sort_start = time.time()
        spot_sort_idx = np.lexsort((spot_times, spot_pairs))
        spot_pairs_sorted = spot_pairs[spot_sort_idx]
        spot_times_sorted = spot_times[spot_sort_idx]
        spot_rates_sorted = spot_rates[spot_sort_idx]
        logger.info(f"Sorting completed in {time.time() - sort_start:.3f} seconds")

        pair_boundaries = {}
        for pair in np.unique(spot_pairs_sorted):
            mask = spot_pairs_sorted == pair
            indices = np.where(mask)[0]
            if len(indices) > 0:
                pair_boundaries[pair] = (indices[0], indices[-1] + 1)

        # Results
        result_rates = np.full(len(price_pairs), np.nan)
        one_hour_ns = pd.Timedelta(hours=1).value

        lookup_start = time.time()
        for pair in np.unique(price_pairs):
            if pair not in pair_boundaries:
                continue

            # Price data for this pair
            price_mask = price_pairs == pair
            price_indices = np.where(price_mask)[0]
            pair_price_times = price_times[price_mask]

            # Spot data boundaries
            start_idx, end_idx = pair_boundaries[pair]
            pair_spot_times = spot_times_sorted[start_idx:end_idx]
            pair_spot_rates = spot_rates_sorted[start_idx:end_idx]

            #lookup
            insert_positions = np.searchsorted(pair_spot_times, pair_price_times, side='right') - 1
            valid_positions = insert_positions >= 0

            if np.any(valid_positions):
                valid_insert_pos = insert_positions[valid_positions]
                valid_price_times = pair_price_times[valid_positions]
                valid_spot_times = pair_spot_times[valid_insert_pos]

                # Time difference check
                time_diffs = valid_price_times - valid_spot_times
                within_tolerance = time_diffs <= one_hour_ns

                final_valid_mask = valid_positions.copy()
                final_valid_mask[valid_positions] = within_tolerance

                result_indices = price_indices[final_valid_mask]
                spot_value_indices = start_idx + insert_positions[final_valid_mask]
                result_rates[result_indices] = spot_rates_sorted[spot_value_indices]

        logger.info(f"Lookup completed in {time.time() - lookup_start:.3f} seconds")

        # Create result DataFrame
        result = self.price_data.copy()
        result['spot_mid_rate'] = result_rates

        logger.info(f"find_spot_rates completed in {time.time() - start_time:.3f} seconds")
        return result

    def calculate_new_prices(self):
        """Calculate new prices based on conversion rules."""
        logger.info("Calculating new prices...")

        merged_data = self.find_spot_rates()

        # Merge with currency reference data
        result = merged_data.merge(
            self.ccy_data[['ccy_pair', 'convert_price', 'conversion_factor']],
            on='ccy_pair',
            how='left'
        )

        # Flags
        result['has_sufficient_data'] = (~result['spot_mid_rate'].isna()) & (result['convert_price'] == True)
        result['supported'] = ~result['convert_price'].isna()

        # new_price from price
        result['new_price'] = result['price']

        # conversion where applicable
        conversion_mask = result['has_sufficient_data']
        result.loc[conversion_mask, 'new_price'] = (
            result.loc[conversion_mask, 'price'] / result.loc[conversion_mask, 'conversion_factor']
            + result.loc[conversion_mask, 'spot_mid_rate']
        )

        # Handling insufficient cases
        insufficient_mask = (result['convert_price'] == True) & (~result['has_sufficient_data'])
        result.loc[insufficient_mask, 'new_price'] = np.nan

        # Conversion status
        result['conversion_status'] = np.where(
            result['convert_price'] != True,
            'no_conversion_required',
            np.where(
                result['has_sufficient_data'],
                'converted',
                np.where(result['supported'], 'insufficient_spot', 'unsupported_ccy')
            )
        )

        self.result = result
        logger.info(f"Calculated new prices for {len(result)} records")

    def save_result(self, output_path="Parameta/rates_test/results", filename="rates_result.csv"):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / filename
        output_cols = [
            'security_id',
            'ccy_pair',
            'timestamp',
            'price',
            'new_price',
            'spot_mid_rate',
            'conversion_factor',
            'convert_price',
            'conversion_status',
        ]

        self.result[output_cols].to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return output_file


    def process_pipeline(self, output_path="Parameta/rates_test/results", filename="rates_result.csv"):
        """Main pipeline"""
        self.load_data()
        self.calculate_new_prices()
        output_file = self.save_result(output_path=output_path, filename=filename)
        return self.result


if __name__ == '__main__':
    start_time = time.time()

    processor = RateProcessor()
    result = processor.process_pipeline()

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {len(result)} records")
    print(f"Records with sufficient data: {result['has_sufficient_data'].sum()}")
    print(f"Records requiring conversion: {(result['convert_price'] == True).sum()}")
