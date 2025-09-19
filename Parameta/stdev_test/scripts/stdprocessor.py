import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class CalculationConfig:
    window_size: int = 20
    price_columns: List[str] = None
    
    def __post_init__(self):
        if self.price_columns is None:
            self.price_columns = ['bid', 'mid', 'ask']

class RollingStandardDeviationCalculator:
    def __init__(self, window_size: int = 20):
        self.config = CalculationConfig(window_size=window_size)

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load and prepare data with optimized dtypes"""
        df = pd.read_parquet(file_path, engine="pyarrow")
        df['snap_time'] = pd.to_datetime(df['snap_time'])
        # Pre-sort once to avoid repeated sorting
        return df.sort_values(['security_id', 'snap_time'])

    def _vectorized_rolling_std_single_security(self, values: np.ndarray, full_range_size: int, 
                                               output_indices: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation for a single security using NumPy operations
        """
        n_rows, n_cols = values.shape
        rolling_std = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        
        # Creating sliding window view using numpy 
        window_buffer = np.full((self.config.window_size, n_cols), np.nan, dtype=np.float64)
        buffer_pos = 0
        last_valid_std = None
        window_count = 0
        
        # Vectorized NaN detection for all rows at once
        nan_mask = np.isnan(values).any(axis=1)
        
        for i in range(n_rows):
            if nan_mask[i]:
                # Reset window on NaN but keep last valid std
                window_count = 0
                buffer_pos = 0
                if last_valid_std is not None:
                    rolling_std[i] = last_valid_std
            else:
                # Add valid data to circular buffer
                window_buffer[buffer_pos] = values[i]
                buffer_pos = (buffer_pos + 1) % self.config.window_size
                window_count = min(window_count + 1, self.config.window_size)
                
                if window_count == self.config.window_size:
                    # Calculate std using vectorized numpy operations
                    # Reorder buffer to correct sequence
                    if buffer_pos == 0:
                        ordered_window = window_buffer
                    else:
                        ordered_window = np.vstack([
                            window_buffer[buffer_pos:],
                            window_buffer[:buffer_pos]
                        ])
                    
                    last_valid_std = np.std(ordered_window, axis=0, ddof=1)
                    rolling_std[i] = last_valid_std
                elif last_valid_std is not None:
                    rolling_std[i] = last_valid_std
        
        # Return only the requested output indices
        return rolling_std[output_indices]

    def calculate_rolling_std(self, df: pd.DataFrame, start_time: str = None, 
                            end_time: str = None) -> pd.DataFrame:
        """Optimized calculation using vectorized operations"""
        
        # Vectorized datetime parsing
        if start_time:
            start_dt = pd.to_datetime(start_time)
        else:
            start_dt = df['snap_time'].min()
        if end_time:
            end_dt = pd.to_datetime(end_time)
        else:
            end_dt = df['snap_time'].max()

        # Generate output range once
        output_snaps = pd.date_range(start=start_dt, end=end_dt, freq='h')
        
        # Group by security_id for vectorized processing
        grouped = df.groupby('security_id', sort=False)
        
        # Pre-allocate result arrays for efficiency
        n_securities = df['security_id'].nunique()
        n_snaps = len(output_snaps)
        total_rows = n_securities * n_snaps
        
        # Pre-allocate output arrays
        result_snap_times = np.tile(output_snaps, n_securities)
        result_security_ids = np.repeat(df['security_id'].unique(), n_snaps)
        result_stds = np.full((total_rows, len(self.config.price_columns)), np.nan, dtype=np.float64)
        
        row_idx = 0
        
        for sec_id, sec_df in grouped:
            if len(sec_df) == 0:
                # Skip to next security, NaN values already pre-allocated
                row_idx += n_snaps
                continue
            
            # Vectorized reindexing and processing
            sec_df_indexed = sec_df.set_index('snap_time').sort_index()
            
            # Calculate lookback range
            earliest_data = sec_df_indexed.index.min()
            lookback_start = min(earliest_data, start_dt - pd.Timedelta(hours=self.config.window_size))
            
            # Create full range for this security
            full_range = pd.date_range(start=lookback_start, end=end_dt, freq='h')
            full_df = sec_df_indexed.reindex(full_range)
            
            # Extract values as numpy array for vectorized operations
            values = full_df[self.config.price_columns].values
            
            # Get indices for output snaps
            output_indices = full_range.get_indexer(output_snaps)
            
            # Calculate rolling std using vectorized function
            sec_std_results = self._vectorized_rolling_std_single_security(
                values, len(full_range), output_indices
            )
            
            # Store results in pre-allocated arrays
            end_idx = row_idx + n_snaps
            result_stds[row_idx:end_idx] = sec_std_results
            row_idx = end_idx
        
        # Create DataFrame from pre-allocated arrays (most efficient)
        results = pd.DataFrame({
            'snap_time': result_snap_times,
            'security_id': result_security_ids,
            'bid_std': result_stds[:, 0],
            'mid_std': result_stds[:, 1],
            'ask_std': result_stds[:, 2]
        })
        
        # Final sort using pandas optimized sorting
        return results.sort_values(['snap_time', 'security_id']).reset_index(drop=True)

    def save_results(self, results_df: pd.DataFrame, output_path: Path):
        """Optimized save with minimal memory copying"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use copy() only when necessary and vectorized string formatting
        results_df_output = results_df.copy()
        results_df_output['snap_time'] = results_df_output['snap_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        results_df_output.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    def process_file(self, input_file: Path, output_file: Path, start_time: str = None, 
                    end_time: str = None) -> pd.DataFrame:
        """Main processing pipeline with timing"""
        start_processing = time.time()
        
        df = self.load_data(input_file)
        results = self.calculate_rolling_std(df, start_time, end_time)
        self.save_results(results, output_file)
        
        processing_time = time.time() - start_processing
        print(f"Output shape: {results.shape}")
        print(f"Time range: {results['snap_time'].min()} to {results['snap_time'].max()}")
        
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate rolling standard deviation with lookback")
    default_input = Path(__file__).parent.parent / "data" / "stdev_price_data.parq"
    default_output = Path(__file__).parent.parent / "results" / "rolling_std_results.csv"

    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--output", type=str, default=default_output)
    parser.add_argument("--start_time", type=str, default="2021-11-20 00:00:00")
    parser.add_argument("--end_time", type=str, default="2021-11-23 09:00:00")

    args = parser.parse_args()

    calculator = RollingStandardDeviationCalculator(window_size=20)
    calculator.process_file(
        input_file=Path(args.input),
        output_file=Path(args.output),
        start_time=args.start_time,
        end_time=args.end_time
    )