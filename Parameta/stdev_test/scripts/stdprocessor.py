import pandas as pd
import numpy as np
from pathlib import Path
import time


class RollingStandardDeviationCalculator:
    """
    Efficient rolling standard deviation calculator for price data.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_columns = ['bid', 'mid', 'ask']

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load price data from parquet file.
        """
        df = pd.read_parquet(file_path)
        df['snap_time'] = pd.to_datetime(df['snap_time'])
        df = df.sort_values(['security_id', 'snap_time']).reset_index(drop=True)
        return df

    def _identify_contiguous_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify contiguous hourly sequences for each security.
        """
        df = df.copy()
        df['time_diff'] = df.groupby('security_id')['snap_time'].diff()
        expected_diff = pd.Timedelta(hours=1)
        df['is_break'] = (df['time_diff'] != expected_diff) | df['time_diff'].isna()
        df['sequence_group'] = df.groupby('security_id')['is_break'].cumsum()
        df['group_id'] = df['security_id'].astype(str) + '_' + df['sequence_group'].astype(str)
        return df.drop(['time_diff', 'is_break', 'sequence_group'], axis=1)

    def calculate_rolling_std(
        self,
        df: pd.DataFrame,
        start_time: str = "2021-11-20 00:00:00",
        end_time: str = "2021-11-23 09:00:00"
    ) -> pd.DataFrame:
        """
        Calculate rolling standard deviations for the specified time period.
        """
        print("Identifying contiguous sequences...")
        df_with_groups = self._identify_contiguous_sequences(df)

        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)

        # Keep only data up to end_time (need history before start_time)
        df_with_groups = df_with_groups[df_with_groups['snap_time'] <= end_dt]

        print("Calculating rolling STD...")

        def _calc_group(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) < self.window_size:
                return pd.DataFrame()

            rolling_std = group[self.price_columns].rolling(self.window_size).std(ddof=1)

            result = group[['snap_time', 'security_id']].copy()
            for col in self.price_columns:
                result[f"{col}_std"] = rolling_std[col].values

            # Drop rows without full window
            return result.iloc[self.window_size - 1:]

        results = (
            df_with_groups.groupby("group_id", group_keys=False)
            .apply(_calc_group)
        )

        if results.empty:
            return pd.DataFrame(columns=['snap_time', 'security_id'] + [f"{c}_std" for c in self.price_columns])

        # Filter final output to requested window
        results = results[(results['snap_time'] >= start_dt) & (results['snap_time'] <= end_dt)]

        # Sort for consistent output
        results = results.sort_values(['snap_time', 'security_id']).reset_index(drop=True)

        return results

    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """
        Save results to CSV file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df_output = results_df.copy()
        results_df_output['snap_time'] = results_df_output['snap_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        results_df_output.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    def process_file(
        self,
        input_file: str,
        output_file: str,
        start_time: str = "2021-11-20 00:00:00",
        end_time: str = "2021-11-23 09:00:00"
    ) -> pd.DataFrame:
        """
        process pipeline: load, calculate, and save results.
        """
        start_processing = time.time()

        print("Loading data...")
        df = self.load_data(input_file)
        print(f"Loaded {len(df):,} rows")

        print(f"Calculating rolling standard deviations from {start_time} to {end_time}...")
        results = self.calculate_rolling_std(df, start_time, end_time)

        print(f"Generated {len(results):,} result rows")

        self.save_results(results, output_file)

        processing_time = time.time() - start_processing
        print(f"Total processing time: {processing_time:.2f} seconds")

        return results


if __name__ == '__main__':
    # Configuration
    input_file = "/workspaces/parameta-data-science-test/Parameta/stdev_test/data/stdev_price_data.parq"
    output_file = "/workspaces/parameta-data-science-test/Parameta/stdev_test/results/rolling_std_results.csv"

    calculator = RollingStandardDeviationCalculator(window_size=20)

    try:
        results = calculator.process_file(
            input_file=input_file,
            output_file=output_file,
            start_time="2021-11-20 00:00:00",
            end_time="2021-11-23 09:00:00"
        )

        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total results: {len(results):,}")
        print(f"Unique securities: {results['security_id'].nunique()}")
        print(f"Time range: {results['snap_time'].min()} to {results['snap_time'].max()}")

    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please ensure the file exists in the correct location.")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
