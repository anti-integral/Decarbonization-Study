import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.utils import save_forecaster
import warnings
warnings.filterwarnings('ignore')

class WeatherForecast:
    def __init__(self, api_key="lJpHWX5EOvF21grBvTZldyZcX4p7heuB7vQX0E3j"):
        self.api_key = api_key
        self.data = None
        self.mdf = None
        self.grp = None
    
    def fetch_data(self, years, lat, lon):
        """Previous fetch_data implementation remains the same"""
        all_data = []
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        for year in years:
            print(f"Fetching data for {year}...")
            url = f"http://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.json?api_key={self.api_key}&full_name=Sample+User&email=faisal3e@gmail.com&affiliation=Test+Organization&reason=Example&mailing_list=true&wkt=POINT({lon}+{lat})&names={year}&attributes=air_temperature,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,dew_point,dhi,dni,fill_flag,ghi,relative_humidity,solar_zenith_angle,surface_albedo,surface_pressure,total_precipitable_water,wind_direction,wind_speed&leap_day=false&utc=false&interval=60"
            
            try:
                r = requests.get(url, headers=headers).json()
                time.sleep(10)
                data = pd.read_csv(r["outputs"]["downloadUrl"], skiprows=2)
                time.sleep(7)
                
                data.columns = data.columns.str.lower().str.replace(" ", "_")
                data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
                data.wind_speed = data.wind_speed * 1.83478607022
                
                all_data.append(data)
                
            except Exception as e:
                print(f"Error fetching data for {year}: {str(e)}")
                continue
        
        if all_data:
            self.data = pd.concat(all_data).reset_index(drop=True)
            self._process_data()
            return True
        return False
    
    def _process_data(self):
        """Previous _process_data implementation remains similar"""
        self.data = self.data.sort_values('datetime')
        
        # Create complete datetime range
        date_range = pd.date_range(
            start=self.data.datetime.min(),
            end=self.data.datetime.max(),
            freq='H'
        )
        
        # Ensure data completeness
        full_range = pd.DataFrame(date_range, columns=['datetime'])
        self.data = pd.merge_asof(
            full_range,
            self.data,
            on='datetime',
            direction='nearest'
        )
        
        # Extract time components
        self.data['year'] = self.data.datetime.dt.year
        self.data['month'] = self.data.datetime.dt.month
        self.data['day'] = self.data.datetime.dt.day
        self.data['hour'] = self.data.datetime.dt.hour
        
        # Calculate averages
        self.grp = self.data.groupby(['month', 'day', 'hour'])[['temperature', 'wind_speed', 'dni']].mean()
        self.grp = self.grp.reset_index()
        
        # Merge with averages
        self.mdf = self.data.merge(
            self.grp,
            on=['month', 'day', 'hour'],
            suffixes=('_actual', '_avg')
        )
        
        # Calculate differences
        self.mdf['temp_diff'] = self.mdf['temperature_actual'] - self.mdf['temperature_avg']
        self.mdf['dni_diff'] = self.mdf['dni_actual'] - self.mdf['dni_avg']
        self.mdf['wind_speed_diff'] = self.mdf['wind_speed_actual'] - self.mdf['wind_speed_avg']

    def make_forecast(self, target, n_years=5, save_path=None, plot=True):
        """
        Make forecast with both difference and absolute values
        """
        if self.mdf is None:
            print("No data available. Please fetch data first.")
            return None
            
        # Map diff variables to their average counterparts
        avg_map = {
            'temp_diff': 'temperature',
            'dni_diff': 'dni',
            'wind_speed_diff': 'wind_speed'
        }
        
        if target not in avg_map:
            raise ValueError(f"Target must be one of {list(avg_map.keys())}")
            
        # Prepare the data
        train_data = self.mdf.copy()
        
        # Reset index properly to ensure continuous indexing
        train_data = train_data.reset_index()
        last_idx = train_data.index[-1]
        
        # Calculate forecast dates
        forecast_start = pd.Timestamp("2023-01-01 00:00:00")
        forecast_end = forecast_start + pd.DateOffset(years=n_years) - pd.Timedelta(hours=1)
        
        print(f"Generating forecast from {forecast_start} to {forecast_end}")
        
        # Create future dates
        future_dates = pd.date_range(
            start=forecast_start,
            end=forecast_end,
            freq="H"
        )
        
        # Create exogenous features for future dates
        future_exog = pd.DataFrame(index=range(last_idx + 1, last_idx + 1 + len(future_dates)))
        future_exog['month'] = future_dates.month
        future_exog['day'] = future_dates.day
        future_exog['hour'] = future_dates.hour
        future_exog['year'] = future_dates.year
        
        # Initialize model
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Initialize forecaster
        forecaster = ForecasterAutoreg(
            regressor=model,
            lags=168  # One week of lags
        )
        
        # Prepare training data with proper indexing
        train_y = train_data[target]
        train_exog = train_data[['month', 'day', 'hour', 'year']]
        
        print("Fitting the model...")
        # Fit the model
        forecaster.fit(y=train_y, exog=train_exog)
        
        print("Making predictions...")
        # Make predictions
        predictions = forecaster.predict(
            steps=len(future_dates),
            exog=future_exog
        )
        
        # Create forecast DataFrame with consistent column names
        future = pd.DataFrame({
            'datetime': future_dates,
            'diff_pred': predictions
        })
        
        # Apply random variation
        coef = np.random.uniform(low=0.75, high=0.95, size=len(future))
        future['diff_pred'] = future['diff_pred'] * coef
        
        # Get average values
        avg_values = self.grp.copy()
        future['month'] = future.datetime.dt.month
        future['day'] = future.datetime.dt.day
        future['hour'] = future.datetime.dt.hour
        
        # Merge with average values
        future = future.merge(
            avg_values[['month', 'day', 'hour', avg_map[target]]],
            on=['month', 'day', 'hour'],
            how='left'
        )
        
        # Calculate absolute predictions
        future['abs_pred'] = future['diff_pred'] + future[avg_map[target]]
        
        # Handle specific cases
        if target == "dni_diff":
            night_hours = [17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            future.loc[future.hour.isin(night_hours), ['diff_pred', 'abs_pred']] = 0
            future.abs_pred = future.abs_pred.clip(lower=0)
        
        elif target == "wind_speed_diff":
            future.abs_pred = future.abs_pred.clip(lower=0)
        
        # Clean up and reorder columns
        result = future[['datetime', 'diff_pred', avg_map[target], 'abs_pred']].copy()
        result = result.rename(columns={avg_map[target]: f"{avg_map[target]}_avg"})
        
        
        # Display results
        print("\nForecast statistics for {}:".format(target))
        print("First 5 predictions:")
        print(result.head().to_string())
        
        print("\nSummary statistics:")
        print("\nDifference predictions:")
        print(result['diff_pred'].describe().to_string())
        print("\nAbsolute predictions:")
        print(result['abs_pred'].describe().to_string())
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot differences
            train_data.set_index('datetime')[target].tail(24*30).plot(ax=ax1, label='Historical (diff)')
            result.set_index('datetime')['diff_pred'].head(24*30).plot(ax=ax1, label='Forecast (diff)')
            ax1.set_title(f'{target} Forecast - Differences')
            ax1.legend()
            
            # Plot absolute values
            actual_var = avg_map[target] + '_actual'
            train_data.set_index('datetime')[actual_var].tail(24*30).plot(ax=ax2, label='Historical (absolute)')
            result.set_index('datetime')['abs_pred'].head(24*30).plot(ax=ax2, label='Forecast (absolute)')
            ax2.set_title(f'{avg_map[target]} Forecast - Absolute Values')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
        
        if save_path:
            result.to_csv(save_path, index=False)
            print(f"\nForecast saved to {save_path}")
        
        # Save the model
        save_forecaster(forecaster, file_name=f"{target}_average_model.joblib", verbose=False)
        
        return result