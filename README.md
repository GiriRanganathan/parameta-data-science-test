# parameta-data-science-test
Repository with solution for parameta datascience test
1. **Problem Statment-01**:
   Goal: Generate a new price for each row in rates_price_data. The new price depends on whether that ccy_pair is supported, needs to be converted and has sufficient data to convert:
         If conversion is not required, then the new price is simply the ‘existing price’
         If conversion is required, then the new price is: (‘existing price’/ ‘conversion factor’) + ‘spot_mid_rate’
         If there is insufficient data to create a new price then capture this fact in some way
   
   **Requirements**:
   Code requirements are in requirements.txt
   Command to install:
   pip install -r requirements.txt

   **Run Code**:
   Command to run rateprocessor.py code
   python Parameta/rates_test/scripts/rateprocessor.py

3. **Problem Statment-02**:
   Goal: To generate a standard deviation for a security id at a given hourly snap time, you need the most recent set of 20 contiguous hourly snap values for the security id. By contiguous we mean there are no gaps in the set of hourly snaps.
   
   **Requirements**:
   Code requirements are in requirements.txt
   Command to install:
   pip install -r requirements.txt

   **Run Code**:
   Command to run stdprocessor.py code
   python Parameta/stdev_test/scripts/stdprocessor.py \
    --start_time "2021-11-20 00:00:00" \
    --end_time "2021-11-23 09:00:00"
