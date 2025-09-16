# parameta-data-science-test
Repository with solution for parameta datascience test
1. Problem Statment-01:
   Goal: Generate a new price for each row in rates_price_data. The new price depends on whether that ccy_pair is supported, needs to be converted and has sufficient data to convert:
         If conversion is not required, then the new price is simply the ‘existing price’
         If conversion is required, then the new price is: (‘existing price’/ ‘conversion factor’) + ‘spot_mid_rate’
         If there is insufficient data to create a new price then capture this fact in some way
2. Requirements:
   Code requirements are in requirements.txt
   Command to install:
   pip install -r requirements.txt

3. Run Code:
   Command to run rateprocessor.py code
   python Parameta/rates_test/scripts/rateprocessor.py
