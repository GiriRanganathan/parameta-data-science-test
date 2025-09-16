import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse

class RollingStandardDeviationCalculator:
    """
    Rolling standard deviation calculator for price data.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_columns = ['bid', 'mid', 'ask']

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load price data from parquet file.
        """
        df = pd.read_parquet(file_path,engine="pyarrow")
        df['snap_time'] = pd.to_datetime(df['snap_time'])
        df = df.sort_values(['security_id', 'snap_time']).reset_index(drop=True)
        return df

    def _identify_contiguous_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifying contiguous hourly sequences for each security.
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
        start_time: str = None,
        end_time: str = None
    ) -> pd.DataFrame:
        """
        Calculating rolling standard deviations for the specified time period.
        """
        df_with_groups = self._identify_contiguous_sequences(df)

        if start_time:
            start_dt = pd.to_datetime(start_time)
        else:
            start_dt = df['snap_time'].min()

        if end_time:
            end_dt = pd.to_datetime(end_time)
        else:
            end_dt = df['snap_time'].max()

        # Keep only data up to end_time (need history before start_time)
        df_with_groups = df_with_groups[df_with_groups['snap_time'] <= end_dt]

        def _calc_group(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) < self.window_size:
                return pd.DataFrame()

            rolling_std = group[self.price_columns].rolling(self.window_size).std(ddof=1)

            result = group[['snap_time', 'security_id']].copy()
            for col in self.price_columns:
                result[f"{col}_std"] = rolling_std[col].values

            return result.iloc[self.window_size - 1:]

        # FutureWarning fix - exclude grouping columns from apply
        results = (
            df_with_groups.groupby("group_id", group_keys=False)
            .apply(_calc_group, include_groups=False)
        )

        if results.empty:
            return pd.DataFrame(columns=['snap_time', 'security_id'] + [f"{c}_std" for c in self.price_columns])

        # Filter final output to requested window
        results = results[(results['snap_time'] >= start_dt) & (results['snap_time'] <= end_dt)]
        results = results.sort_values(['snap_time', 'security_id']).reset_index(drop=True)
        return results

    def save_results(self, results_df: pd.DataFrame, output_path: Path):
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
        input_file: Path,
        output_file: Path,
        start_time: str = None,
        end_time: str = None
    ) -> pd.DataFrame:
        """
        Full processing: load data, calculate rolling std, save results.
        """
        start_processing = time.time()

        print(f"Loading data from: {input_file}")
        df = self.load_data(input_file)
        print(f"Loaded {len(df):,} rows")

        print(f"Calculating rolling standard deviations...")
        results = self.calculate_rolling_std(df, start_time, end_time)
        print(f"Generated {len(results):,} result rows")

        self.save_results(results, output_file)

        processing_time = time.time() - start_processing
        print(f"Total processing time: {processing_time:.2f} seconds")
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate rolling standard deviation")
    
    # Default paths relative to this script
    default_input = Path(__file__).parent.parent / "data" / "stdev_price_data.parq"
    default_output = Path(__file__).parent.parent / "results" / "rolling_std_results.csv"

    parser.add_argument("--input", type=str, default=default_input, help="Path to input parquet file")
    parser.add_argument("--output", type=str, default=default_output, help="Path to output CSV file")
    parser.add_argument("--start_time", type=str, default=None, help="Optional start time filter (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end_time", type=str, default=None, help="Optional end time filter (YYYY-MM-DD HH:MM:SS)")

    args = parser.parse_args()

    calculator = RollingStandardDeviationCalculator(window_size=20)
    calculator.process_file(
        input_file=Path(args.input),
        output_file=Path(args.output),
        start_time=args.start_time,
        end_time=args.end_time
    )
