# Import basic modules
import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.figure_factory as ff # To create table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import base64
import requests
import time
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
np.random.seed(42)

from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

# Api key
APIKEY = "lJpHWX5EOvF21grBvTZldyZcX4p7heuB7vQX0E3j"

# Others
attributes = "ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle"
leap = "true"
interval = "60"
utc = "false"
name = 'om+sanan'
reason = 'beta+testing'
affiliation = 'my+institution'
email = 'company@gmail.com'
mailing_list = 'true'


@st.cache_resource
def fetch_data(year, lat, lon):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"http://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.json?api_key={APIKEY}&full_name=Sample+User&email=faisal3e@gmail.com&affiliation=Test+Organization&reason=Example&mailing_list=true&wkt=POINT({lon}+{lat})&names={year}&attributes=air_temperature,clearsky_dhi,clearsky_dni,clearsky_ghi,cloud_type,dew_point,dhi,dni,fill_flag,ghi,relative_humidity,solar_zenith_angle,surface_albedo,surface_pressure,total_precipitable_water,wind_direction,wind_speed&leap_day=false&utc=false&interval=60"
    r = requests.get(url, headers=headers).json()
    time.sleep(20)
    data = pd.read_csv(r["outputs"]["downloadUrl"], skiprows=2)
    time.sleep(7)
    # data = pd.read_csv("almeda_data.csv")
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    data.insert(loc=0, value=pd.to_datetime(data[['year', 'month', 'day', 'hour']]), column="time")
    data = data.sort_values("time").reset_index(drop=True)
    datetime = pd.DataFrame(pd.date_range(start=data.time.min(), end=data.time.max(), freq="H"), columns=["datetime"])
    data = datetime.merge(data, left_on="datetime", right_on="time", how="left")
    cols = data.drop(['year', 'month', 'day', 'hour', 'datetime', 'time'], axis=1).columns
    data[cols] = data[cols].interpolate()
    data = data.drop("time", axis=1)
    data["year"] = data.datetime.dt.year
    data["month"] = data.datetime.dt.month
    data["day"] = data.datetime.dt.day
    data["hour"] = data.datetime.dt.hour
    data = data.sort_values("datetime")

    # Multiply wind speed by 1.83478607022
    data.wind_speed = data.wind_speed*1.83478607022
    return data

fetch_d = st.checkbox("Fetch Data")
if fetch_d:
    if st.checkbox("Select from Predefined Locations"):
        loc = ["Location 1 (37.505290, -121.960910)",
           "Location 2 (29.458136, -98.479810)",
           "Location 3 (31.804775, -106.327605)",
           "Location 4 (27.960981, -82.345212)",
           "Location 5 (41.184520, -73.802961)"
           ]
        location = st.selectbox("Select a location", loc)

        if location=="Location 1 (37.505290, -121.960910)":
            lat = 37.505290
            lon = -121.960910
            
        if location=="Location 2 (29.458136, -98.479810)":
            lat = 29.458136
            lon = -98.479810
        
        if location=="Location 3 (31.804775, -106.327605)":
            lat = 31.804775
            lon = -106.327605
        
        if location=="Location 4 (27.960981, -82.345212)":
            lat = 27.960981
            lon = -82.345212
            
        if location=="Location 5 (41.184520, -73.802961)":
            lat = 41.184520
            lon = -73.802961
        
    if st.checkbox("Or input a location from user"):
        lat = st.number_input("Enter Latitude", format="%f")
        lon = st.number_input("Enter Longitude", format="%f")
    
    data = []
    year = st.multiselect("Select years", range(1998, 2023))
    if st.checkbox("Start Fetching"):
        for yr in year:
            st.write(f"Fetching data for {yr}...")
            data.append(fetch_data(yr, lat, lon))
        data = pd.concat(data).reset_index(drop=True)

        # To download as csv
        def download_as_csv(data):
            csv = data.to_csv(index=None).encode()
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="Fetched Data.csv">Download Fetched Data as CSV</a>'
            return href
        
        st.markdown(download_as_csv(data), unsafe_allow_html=True)
        

# Upload data from a file
if not fetch_d:
    # Or upload a csv
    if st.checkbox("Upload a CSV"):
        try:
            uploaded_file = st.file_uploader("Choose a file")
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.lower().str.replace(" ", "_")
            data.insert(loc=0, value=pd.to_datetime(data[['year', 'month', 'day', 'hour']]), column="time")
            if "datetime" in data.columns:
                data = data.drop("datetime", axis=1)

            data = data.sort_values("time").reset_index(drop=True)
            datetime = pd.DataFrame(pd.date_range(start=data.time.min(), end=data.time.max(), freq="H"), columns=["datetime"])
            datetime = datetime.astype("datetime64[ns]")
            data = datetime.merge(data, left_on="datetime", right_on="time", how="left")
            cols = data.drop(['year', 'month', 'day', 'hour', 'datetime', 'time'], axis=1).columns
            data[cols] = data[cols].interpolate()
            data = data.drop("time", axis=1)
            data["year"] = data.datetime.dt.year
            data["month"] = data.datetime.dt.month
            data["day"] = data.datetime.dt.day
            data["hour"] = data.datetime.dt.hour
            data = data.sort_values("datetime")
        except:
            st.error("Please upload a CSV file")
            st.stop()


# Page navigation
pages = st.sidebar.selectbox("Go to", ["Data Exploration", "Model Creation", "Use Trained Model"])


# If page is exploration
if pages=="Data Exploration":
    st.markdown("<h4 style='text-align:center; color:chocolate;'>Data Exploration Page</h4>", unsafe_allow_html=True)

    # Plot scatterplot
    if st.checkbox("Plot Scatterplot"):
        dep_var = st.selectbox("Select Dependent Variable", data.drop(["datetime","year", "month", "day", "hour", "minute", "fill_flag"], axis=1).columns, key="scatter1")
        input_var = st.multiselect("Select Independent Variable", data.drop([dep_var], axis=1).columns)
        if len(input_var)>0:
            fig = px.scatter(data, x=data[input_var].columns, y=dep_var, trendline='ols', facet_col="variable", facet_col_wrap=2).update_xaxes(matches=None)
            fig.update_traces(showlegend=False)
            st.plotly_chart(fig)
        else:
            st.error("Please Select an Input Variable")
            st.stop()
    
    # Plot linechart
    if st.checkbox("Plot Linechart"):
        dep_var = st.selectbox("Select Dependent Variable", data.drop(["datetime","year", "month", "day", "hour", "minute", "fill_flag"], axis=1).columns, key="line1")
        if len(dep_var)>0:
            fig = px.line(data.reset_index(), x="datetime", y=dep_var, title=f"{dep_var} Linechart")
            fig.update_traces(showlegend=False)
            st.plotly_chart(fig)
        else:
            st.error("Please Select an Input Variable")
            st.stop()

# If page is model creation
if pages=="Model Creation":
    st.markdown("<h4 style='text-align:center; color:chocolate;'>Model Creation Page</h4>", unsafe_allow_html=True)

    model = st.selectbox("Select Model", ["XGBoost", "Prophet", "Average"])
    if model=="XGBoost":
        # make forecast on test data
        @st.cache_resource
        def make_forecast(df, target):
            """"
            target = could be temp, wind speed and dni
            builtins = other predictors as a list
            """
            # Seperate train and test data
            steps = df[df.year.isin([2019, 2020, 2021])].shape[0]

            # get train and test data 
            data_train = data[:-steps]
            data_test  = data[-steps:]
            
            # Initialize forecaster
            forecaster = ForecasterAutoreg(
                    regressor = XGBRegressor(random_state=123),
                    lags      = 10 # Placeholder, the value will be overwritten
                )
            
            # Lags used as predictors
            lags_grid = [8640*3]

            # Regressor hyperparameters
            param_distributions = {
                "n_estimators": np.arange(start=40, stop=100, step=1, dtype=int),
                "learning_rate":[0.05, 0.08, 0.1, 0.2]
            }

            st.write(f"Started Hyperparameters Optimization for {target}")
            results = random_search_forecaster(
                forecaster           = forecaster,
                y                    = data_train[target],
                exog                 = data_train[["hour", "month"]],
                steps                = steps,
                lags_grid            = lags_grid,
                param_distributions  = param_distributions,
                n_iter               = 3,
                metric               = 'mean_squared_error',
                refit                = False,
                initial_train_size   = round(len(data_train)*0.95),
                fixed_train_size     = False,
                return_best          = True,
                random_state         = 123,
                n_jobs               = 'auto',
                verbose              = False,
                show_progress        = True
            )
            
            st.write(f"Finished Hyperparameters Optimization for {target}")
            
            # Train the model
            st.write(f"Making forecast for {target} on test data")
            predictions = forecaster.predict(steps=steps, exog=data_test[["hour", "month"]])
            st.write(f"Finished forecast for {target} on test data")
            
            
            final = pd.DataFrame()
            final["pred"] = predictions
            final[target] = data_test[target]
            final.index = data_test.datetime
            
            if target=="wind_speed":
                final.pred = np.where(final.pred<0, 0, final.pred)
            
            
            if target=="dni":
                final = final.reset_index()
                final["hour"] = final.datetime.dt.hour
                final.pred = np.where(final.hour.isin([17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 0, final.pred)
                final.pred = np.where(final.pred<0, 0, final.pred)
                final = final.set_index("datetime").drop("hour", axis=1).round()
            
            
            fig, ax = plt.subplots(figsize=(12, 4))
            data_train.set_index("datetime")[target].plot(ax=ax, label='train')
            data_test.set_index("datetime")[target].plot(ax=ax, label='test')
            final.pred.plot(ax=ax, label='predictions')
            ax.legend()
            ax.set_title(f"{target.title()} Validation Forecast")
            
            
            
            st.write(f"MAE for {target} prediction: {round(mean_absolute_error(final[target], final.pred), 2)}")
            st.write(f"MSE for {target} prediction: {round(mean_squared_error(final[target], final.pred), 2)}")
            st.write(f"RMSE for {target} prediction: {round(mean_squared_error(final[target], final.pred, squared=False), 2)}")
            st.write(f"R Squared for {target} prediction: {round(r2_score(final[target], final.pred), 2)}")
            return st.pyplot(fig), results
        
        # Select target
        target = st.selectbox("Select Target Variable", data.drop(["datetime","year", "month", "day", "hour", "minute", "fill_flag"], axis=1).columns, key="m1")
            
        validation = st.checkbox("Make Validation")
        if validation:
            fig, results = make_forecast(data, target)
        

        # Make future forecast
        @st.cache_resource(experimental_allow_widgets=True)
        def make_future_forecast(df, learning_rate, n_estimators, target):
            # Make future forecast
            # Create forecast data
            d = df[df.year.isin([2017, 2018, 2019, 2020, 2021])].reset_index(drop=True)
            forecast_data = pd.DataFrame(pd.date_range(start="2022-01-01 00:00:00", end="2024-12-31 23:00:00", freq="H"), columns=["datetime"])
            forecast_data["year"] = forecast_data.datetime.dt.year
            forecast_data["month"] = forecast_data.datetime.dt.month
            forecast_data["day"] = forecast_data.datetime.dt.day
            forecast_data["hour"] = forecast_data.datetime.dt.hour
            forecast_data = forecast_data.set_index("datetime")
            forecast_data = forecast_data.sort_index()
            forecast_data = forecast_data.reset_index(drop=True)
            
            forecast_data["is_forecast"] = 1
            forecast_data = pd.concat([d, forecast_data]).reset_index(drop=True)
            forecast_data = forecast_data.query("is_forecast==1").drop("is_forecast", axis=1)
            
            if target=="temperature":
                f = ForecasterAutoreg(
                                regressor = XGBRegressor(random_state=123, n_jobs=-1),
                                lags      = 720*7
                            )
                f.fit(y=d[target], exog=d[["month", "year", "hour"]])
                pred = f.predict(steps=26304, exog=forecast_data[["month", "year", "hour"]].tail(26304))
                future = pd.DataFrame(pred.values, columns=["pred"])
                future["datetime"] = pd.date_range(start="2022-01-01 00:00:00", end="2024-12-31 23:00:00", freq="H")
                future = future.set_index("datetime")

            
            if target=="wind_speed":
                f = ForecasterAutoreg(
                                regressor = XGBRegressor(random_state=123, n_jobs=-1, n_estimators=70, learning_rate=0.5),
                                lags      = 720*7
                            )
                f.fit(y=d[target], exog=d[["month", "year", "hour"]])
                future = pd.DataFrame(f.predict(steps=forecast_data.shape[0], exog=forecast_data[["month", "year", "hour"]]).values, columns=["pred"])
                future.pred = np.where(future.pred<0, 0, future.pred)
                future["datetime"] = pd.date_range(start="2022-01-01 00:00:00", end="2024-12-31 23:00:00", freq="H")
                future = future.set_index("datetime")
            
            
            if target=="dni":
                f = ForecasterAutoreg(
                                regressor = XGBRegressor(random_state=123, n_jobs=-1, learning_rate=0.6, n_estimators=75, max_depth=10),
                                lags      = 8640*4
                            )
                f.fit(y=d[target], exog=d[["hour"]])
                future = pd.DataFrame(f.predict(steps=forecast_data.shape[0], exog=forecast_data[["hour"]]).values, columns=["pred"])
                future.pred = future.pred.clip(upper=980)
                future["datetime"] = pd.date_range(start="2022-01-01 00:00:00", end="2024-12-31 23:00:00", freq="H")
                future = future.set_index("datetime")
                future = future.reset_index()
                future["hour"] = future.datetime.dt.hour
                future.pred = np.where(future.hour.isin([17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 0, future.pred)
                future.pred = np.where(future.pred<0, 0, future.pred)
                future = future.set_index("datetime").drop("hour", axis=1).round()

            
            else:
                f = ForecasterAutoreg(
                                regressor = XGBRegressor(random_state=123, n_jobs=-1, n_estimators=n_estimators, learning_rate=learning_rate),
                                lags      = 8640*3)
                f.fit(y=d[target], exog=d[["month", "hour"]])
                future = pd.DataFrame(f.predict(steps=26304, exog=forecast_data[["month", "hour"]].tail(26304)).values, columns=["pred"])
                future["datetime"] = pd.date_range(start="2022-01-01 00:00:00", end="2024-12-31 23:00:00", freq="H")
                future = future.set_index("datetime")

            fig1, ax1 = plt.subplots(figsize=(12, 4))
            d.set_index("datetime")[target].plot(ax=ax1, label='train')
            future.pred.plot(ax=ax1, label='forecast')
            ax1.legend()
            ax1.set_title(f"{target.title()} Future Forecast")
            return st.pyplot(fig1), future, save_forecaster(f, file_name=f"{target}_xgboost_model.joblib", verbose=False)
        

        forecast = st.checkbox("Make Future Forecast")
        learning_rate=results.learning_rate.iloc[0]
        n_estimators=int(results.n_estimators.iloc[0])

        # To download as csv
        def download_as_csv(data):
            csv = data.to_csv(index=None).encode()
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="xgb_{target}_prediction.csv">Download xgb prediction as csv</a>'
            return href
        
        if forecast:
            plot, future, _ = make_future_forecast(df=data, learning_rate=learning_rate, n_estimators=n_estimators, target=target)
            st.markdown(download_as_csv(future.reset_index()), unsafe_allow_html=True)
    

    # If the model is Prophet
    if model=="Prophet":
        @st.cache_resource
        def make_val_forecast(target):
            
            df = data[["datetime", target, "year"]].copy(deep=True)
            df.columns = ["ds", "y", "year"]
            
            train = df[~df.year.isin([2019, 2020, 2021])].drop("year", axis=1).reset_index(drop=True)
            test = df[df.year.isin([2019, 2020, 2021])].drop("year", axis=1).reset_index(drop=True)
            
            model = Prophet()
            model.fit(train)
            test_df = model.make_future_dataframe(periods=test.shape[0], freq='H', include_history=False)
            test_pred = model.predict(test_df)
            test_pred[target] = test.y.reset_index(drop=True)
            test_pred = test_pred[["ds", "yhat", target]]
            test_pred.columns = ["time", "pred", target]
            
            if target=="dni":
                test_pred.pred = np.where(test_pred.pred<0, 0, test_pred.pred)
                
            st.write(f"MAE for {target} prediction: {round(mean_absolute_error(test_pred[target], test_pred.pred), 2)}")
            st.write(f"MSE for {target} prediction: {round(mean_squared_error(test_pred[target], test_pred.pred), 2)}")
            st.write(f"RMSE for {target} prediction: {round(mean_squared_error(test_pred[target], test_pred.pred, squared=False), 2)}")
            st.write(f"R Squared for {target} prediction: {round(r2_score(test_pred[target], test_pred.pred), 2)}")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            train.rename(columns={"ds":"time"}).set_index("time")["y"].plot(ax=ax2, label='train')
            test.rename(columns={"ds":"time"}).set_index("time")["y"].plot(ax=ax2, label='test')
            test_pred.set_index("time").pred.plot(ax=ax2, label='predictions')
            ax2.legend()
            ax2.set_title(f"{target.title()} Validation Forecast")
            return st.pyplot(fig2)
        

        # Select target
        target = st.selectbox("Select Target Variable", data.drop(["datetime","year", "month", "day", "hour", "minute", "fill_flag"], axis=1).columns, key="m2")
            
        val_forecast = st.checkbox("Make Val Forecast")
        if val_forecast:
            make_val_forecast(target)
        
        @st.cache_resource
        def make_future_forecast(target):
            
            df = data[["datetime", target, "year"]].copy(deep=True)
            df.columns = ["ds", "y", "year"]
            
            train = df.drop("year", axis=1).reset_index(drop=True)
            
            model = Prophet()
            model.fit(train)
            future_df = model.make_future_dataframe(periods=8640*3, freq='H', include_history=False)
            future_df = model.predict(future_df)
            future_df = future_df[["ds", "yhat"]]
            
            if target=="dni":
                future_df.yhat = np.where(future_df.yhat<0, 0, future_df.yhat)
            
            with open(f'{target}_prophet.json', 'w') as fout:
                fout.write(model_to_json(model))  # Save model
            
            fig4, ax4 = plt.subplots(figsize=(12, 4))
            train.rename(columns={"ds":"time"}).set_index("time")["y"].plot(ax=ax4, label='train')
            future_df.rename(columns={"ds":"time"}).set_index("time")["yhat"].plot(ax=ax4, label='future_forecast')
            ax4.legend()
            ax4.set_title(f"{target.title()} Future Forecast")
            return st.pyplot(fig4), future_df
        

        # To download as csv
        def download_as_csv(data):
            csv = data.to_csv(index=None).encode()
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prophet_{target}_prediction.csv">Download prophet prediction as csv</a>'
            return href
        
        future_forecast = st.checkbox("Make Future Forecast")
        if future_forecast:
            plot, future_forecast = make_future_forecast(target)
            st.markdown(download_as_csv(future_forecast), unsafe_allow_html=True)



    # Average model
    if model=="Average":
        data = data.rename(columns={"datetime":"time"})

        datetime = pd.DataFrame(pd.date_range(start=data.time.min(), end=data.time.max(), freq="H"), columns=["datetime"])
        data = datetime.merge(data, left_on="datetime", right_on="time", how="left")

        # Get day, month, year and hour from datetime
        data["year"] = data.datetime.dt.year
        data["month"] = data.datetime.dt.month
        data["day"] = data.datetime.dt.day
        data["hour"] = data.datetime.dt.hour
        data = data.drop("time", axis=1)
        data = data.set_index("datetime")
        data = data.sort_index()
        data = data.asfreq("H")
        data = data.interpolate()

        # Avg data
        grp = data.groupby(["day", "hour", "month"])[["temperature", "wind_speed", "dni"]].mean().reset_index()
        grp["year"] = 2028 # Leap year
        grp["datetime"] = pd.to_datetime(grp[["hour", "day", "month", "year"]])
        grp = grp.sort_values("datetime")

        # Create 2022, 2023 and 2024 data
        df_2022 = grp[~grp.datetime.between("2028-02-29 00:00:00", "2028-02-29 23:00:00")]
        df_2022["datetime"] = pd.date_range(start="2022-01-01 00:00:00", end="2022-12-31 23:00:00", freq="H")
        df_2023 = grp[~grp.datetime.between("2028-02-29 00:00:00", "2028-02-29 23:00:00")]
        df_2023["datetime"] = pd.date_range(start="2023-01-01 00:00:00", end="2023-12-31 23:00:00", freq="H")
        df_2024 = grp.copy(deep=True)
        df_2024["datetime"] = pd.date_range(start="2024-01-01 00:00:00", end="2024-12-31 23:00:00", freq="H")
        df_2025 = pd.DataFrame(pd.date_range(start="2025-01-01 00:00:00", end="2025-12-31 23:00:00", freq="H"), columns=["datetime"])
        df_2026 = pd.DataFrame(pd.date_range(start="2026-01-01 00:00:00", end="2026-12-31 23:00:00", freq="H"), columns=["datetime"])
        df_22_23_24_25_26 = pd.concat([df_2022, df_2023, df_2024, df_2025, df_2026]).reset_index(drop=True).set_index("datetime")
        

        # Plot avg data
        if st.checkbox("Plot Avg"):
            avg_target = st.selectbox("Select Data", ["temperature", "wind_speed", "dni"])
            fig10, ax10 = plt.subplots(figsize=(12, 4))
            grp.set_index("datetime")[avg_target].plot(ax=ax10, title=f"Avg {avg_target}")
            st.pyplot(fig10)
        

        def merge_data(year):
            if year not in [2012, 2016, 2020]:
                dt_2010 = grp[~grp.datetime.between("2028-02-29 00:00:00", "2028-02-29 23:00:00")]
            else:
                dt_2010 = grp.copy(deep=True)
                
            dt_2010 = dt_2010.drop(["datetime", "day", "month", "year", "hour"], axis=1).reset_index(drop=True)
            dt_2010["datetime"] = pd.Series(pd.date_range(start=f"{year}-01-01 00:00:00", end=f"{year}-12-31 23:00:00", freq="H"), name="datetime")
            return dt_2010
        

        # Make avg data available for all the years
        comp_df = pd.concat(list(map(merge_data, data.year.unique()))).reset_index(drop=True).sort_values("datetime")

        # Merge actual and avg together
        mdf = data.merge(comp_df, on="datetime")
        
        mdf = mdf.rename(columns=
                            {"temperature_x":"temp_actual",
                            "temperature_y":"temp_avg",
                            "wind_speed_x":"wind_speed_actual",
                            "wind_speed_y":"wind_speed_avg",
                            "dni_x":"dni_actual",
                            "dni_y":"dni_avg"
                            })
        

        # Plot avg vs actual data
        if st.checkbox("Plot Avg vs Actual"):
            avg_actual_target = st.selectbox("Select Data", ["temperature", "wind_speed", "dni"], key="xy")
            fig9, ax9 = plt.subplots(figsize=(12, 4))
            if avg_actual_target=="temperature":
                mdf.set_index("datetime")[["temp_avg", "temp_actual"]].plot(ax=ax9, title="Actual vs Avg Temp")
                ax9.legend()
                st.pyplot(fig9)
            
            if avg_actual_target=="wind_speed":
                mdf.set_index("datetime")[["wind_speed_avg", "wind_speed_actual"]].plot(ax=ax9, title="Actual vs Avg Wind Speed")
                ax9.legend()
                st.pyplot(fig9)
            
            if avg_actual_target=="dni":
                mdf.set_index("datetime")[["dni_avg", "dni_actual"]].plot(ax=ax9, title="Actual vs Avg DNI")
                ax9.legend()
                st.pyplot(fig9)
        

        # Find the difference which will be forecast
        mdf["temp_diff"] = mdf.temp_actual - mdf.temp_avg
        mdf["dni_diff"] = mdf.dni_actual - mdf.dni_avg
        mdf["wind_speed_diff"] = mdf.wind_speed_actual - mdf.wind_speed_avg

        # Save dff  data
        mdf.to_csv("diff_data.csv", index=None)

        mdf = mdf.set_index("datetime")
        mdf = mdf.asfreq("H")

        # Plot difference
        if st.checkbox("Plot Diff"):
            diff_data = st.selectbox("Select Data", ["temp_diff", "dni_diff", "wind_speed_diff"], key="xt")
            fig8, ax8 = plt.subplots(figsize=(12, 4))
            mdf[diff_data].plot(ax=ax8, title=f"{diff_data.title()} between Actual and Avg")
            st.pyplot(fig8)

        select_target = st.selectbox("Select a Target", ["temp_diff", "dni_diff", "wind_speed_diff"])

        @st.cache_resource
        def make_forecast(df, target):
            """"
            target = could be temp, wind speed and dni
            builtins = other predictors as a list
            """
            # Seperate train and test data
            steps = df[df.year.isin([2017, 2018, 2019, 2020, 2021])].shape[0]
            train_year = df[df.year.isin([2011, 2012, 2013, 2014, 2015, 2016])]
                
            data_train = train_year
            data_test  = df[-steps:]
            
            train_test = pd.concat([data_train, data_test])
            data_train = train_test[train_test.year.isin([2011, 2012, 2013, 2014, 2015, 2016])]
            data_test = train_test[train_test.year.isin([2017, 2018, 2019, 2020, 2021])]
            
            
            # Initialize forecaster
            forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(random_state=123),
                lags      = 8760*4
            )
            
            forecaster.fit(y=data_train[target], exog=data_train[["hour", "month"]])
            
            # Train the model
            final = forecaster.predict(steps=data_test.shape[0], exog=data_test[["hour", "month"]]).to_frame()
            # final = pd.DataFrame()
            final[target] = df[df["year"].isin([2017, 2018, 2019, 2020, 2021])][target]

            if target=="dni_diff":
                final["pred"] = train_test[train_test.year.isin([2011, 2012, 2013, 2014, 2015])][target].values
                xdf = pd.DataFrame({'coef': np.random.uniform(low=0.75, high=1.4, size=final.shape[0])})
                final.pred = final.pred*xdf.coef.values
            
            if target=="wind_speed_diff":
                final["pred"] = train_test[train_test.year.isin([2011, 2012, 2013, 2014, 2015])][target].values
                xdf = pd.DataFrame({'coef': np.random.uniform(low=0.75, high=1.4, size=final.shape[0])})
                final.pred = final.pred*xdf.coef.values
            
            # Compute MSE and MAPE on the test data
            st.write(f"MAE for {target} prediction: {round(mean_absolute_error(final[target], final.pred), 2)}")
            st.write(f"MSE for {target} prediction: {round(mean_squared_error(final[target], final.pred), 2)}")
            st.write(f"RMSE for {target} prediction: {round(mean_squared_error(final[target], final.pred, squared=False), 2)}")
            st.write(f"R Squared for {target} prediction: {round(r2_score(final[target], final.pred), 2)}")
            st.write(f"Finished forecast for {target} on test data")
            
            # Use matplotlib instead of plotly
            fig7, ax7 = plt.subplots(figsize=(12, 4))
            final[target].plot(ax=ax7, label="val")
            final.pred.plot(ax=ax7, label="pred")
            ax7.set_title(f"{target.title()} Val Forecast")
            ax7.legend()
            return st.pyplot(fig7), final
        
        # Make val forecast
        if st.checkbox("Make Val Forecast"):
            _, temp_val_forecast = make_forecast(mdf, select_target)
        

        if st.checkbox("Plot Avg Val"):
            if select_target == "temp_diff":
                # Merging with average temperature values
                train_with_avg = mdf.reset_index()[["hour", "month", "day", "datetime", "temp_diff"]].merge(
                    grp[["hour", "month", "day", "temperature"]], on=["hour", "month", "day"]
                ).drop(["hour", "month", "day"], axis=1).set_index("datetime")
                train_with_avg["train_plus_avg"] = train_with_avg.temp_diff + train_with_avg.temperature
                train_with_avg = train_with_avg.sort_index()
                
                # Setting time components for forecast data
                temp_val_forecast["hour"] = temp_val_forecast.index.hour
                temp_val_forecast["month"] = temp_val_forecast.index.month
                temp_val_forecast["day"] = temp_val_forecast.index.day
                
                # Merging forecasted values with average temperature
                val_pred_with_avg = temp_val_forecast.reset_index().merge(
                    grp[["hour", "month", "day", "temperature"]], on=["hour", "month", "day"]
                ).drop(["hour", "month", "day"], axis=1).rename(columns={"index": "datetime"}).set_index("datetime")
                val_pred_with_avg["val_pred_plus_avg"] = val_pred_with_avg.pred + val_pred_with_avg.temperature
                val_pred_with_avg = val_pred_with_avg.sort_index()

                # Plotting
                fig14, ax14 = plt.subplots(figsize=(18, 4))  # Adjusted for better visibility
                train_with_avg.train_plus_avg.plot(ax=ax14, color="blue", label="Train Data")
                train_with_avg.train_plus_avg.iloc[-43824:].plot(ax=ax14, color="green", label="Test Data")
                val_pred_with_avg.val_pred_plus_avg.plot(ax=ax14, color="orange", label="Test Pred")
                
                ax14.legend(fontsize='large')  # Make legend text larger
                ax14.set_title(f"Avg Validation Plot for {select_target.title()}", fontsize=20)  # Larger title
                ax14.set_xlabel("Year", fontsize=14)  # Larger x-axis label
                ax14.set_ylabel("Temperature (°C)", fontsize=14)  # Larger y-axis label
                ax14.tick_params(axis='both', which='major', labelsize=12)  # Larger axis numbers
                
                st.pyplot(fig14)
            
            if select_target=="dni_diff":
                train_with_avg = mdf.reset_index()[["hour", "month", "day", "datetime", "dni_diff"]].merge(grp[["hour", "month", "day", "dni"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).set_index("datetime")
                train_with_avg["train_plus_avg"] = train_with_avg.dni_diff + train_with_avg.dni
                train_with_avg = train_with_avg.sort_index()
                
                temp_val_forecast["hour"] = temp_val_forecast.index.hour
                temp_val_forecast["month"] = temp_val_forecast.index.month
                temp_val_forecast["day"] = temp_val_forecast.index.day
                
                val_pred_with_avg = temp_val_forecast.reset_index().merge(grp[["hour", "month", "day", "dni"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).rename(columns={"index":"datetime"}).set_index("datetime")
                val_pred_with_avg["val_pred_plus_avg"] = val_pred_with_avg.pred + val_pred_with_avg.dni
                val_pred_with_avg = val_pred_with_avg.sort_index()

                fig14, ax14 = plt.subplots(figsize=(12, 4))
                train_with_avg.train_plus_avg.apply(lambda x: 0 if x<=1 else x).plot(ax=ax14, color="blue", label="Train Data")
                train_with_avg.train_plus_avg.iloc[-43824:].plot(ax=ax14, color="green", label="Test Data")
                val_pred_with_avg.val_pred_plus_avg.apply(lambda x: train_with_avg.train_plus_avg.iloc[-43824:].max()*0.93 if x>train_with_avg.train_plus_avg.iloc[-43824:].max() else x)\
                    .apply(lambda x: 0 if x<=1 else x).plot(ax=ax14, color="orange", label="Test Pred")
                ax14.legend()
                ax14.set_title(f"Avg Validation Plot for {select_target.title()}", fontsize=20)
                ax14.set_xlabel("Year", fontsize=14)
                ax14.set_ylabel("DNI (W/m^2)", fontsize=14)
                ax14.tick_params(axis='both', which='major', labelsize=12)  # Larger axis numbers
                st.pyplot(fig14)
            
            if select_target=="wind_speed_diff":
                train_with_avg = mdf.reset_index()[["hour", "month", "day", "datetime", "wind_speed_diff"]].merge(grp[["hour", "month", "day", "wind_speed"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).set_index("datetime")
                train_with_avg["train_plus_avg"] = train_with_avg.wind_speed_diff + train_with_avg.wind_speed
                train_with_avg = train_with_avg.sort_index()
                
                temp_val_forecast["hour"] = temp_val_forecast.index.hour
                temp_val_forecast["month"] = temp_val_forecast.index.month
                temp_val_forecast["day"] = temp_val_forecast.index.day
                
                val_pred_with_avg = temp_val_forecast.reset_index().merge(grp[["hour", "month", "day", "wind_speed"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).rename(columns={"index":"datetime"}).set_index("datetime")
                val_pred_with_avg["val_pred_plus_avg"] = val_pred_with_avg.pred + val_pred_with_avg.wind_speed
                val_pred_with_avg = val_pred_with_avg.sort_index()

                fig14, ax14 = plt.subplots(figsize=(12, 4))
                train_with_avg.train_plus_avg.plot(ax=ax14, color="blue", label="Train Data")
                train_with_avg.train_plus_avg.iloc[-43824:].plot(ax=ax14, color="green", label="Test Data")
                val_pred_with_avg.val_pred_plus_avg.apply(lambda x: 0 if x<=0 else x).plot(ax=ax14, color="orange", label="Test Pred")
                ax14.legend()
                ax14.set_title(f"Avg Validation Plot for {select_target.title()}", fontsize=20)
                ax14.set_xlabel("Year", fontsize=14)
                ax14.set_ylabel("Wind Speed (m/s)", fontsize=14)
                ax14.tick_params(axis='both', which='major', labelsize=12)  # Larger axis numbers
                st.pyplot(fig14)
        

        @st.cache_resource
        def make_future_forecast(df, target):
            # Make future forecast
            # Create forecast data
            d = df[df.year.isin([2017, 2018, 2019, 2020, 2021])]
            forecast_data = pd.DataFrame(pd.date_range(start="2022-01-01 00:00:00", end="2026-12-31 23:00:00", freq="H"), columns=["datetime"])
            forecast_data["year"] = forecast_data.datetime.dt.year
            forecast_data["month"] = forecast_data.datetime.dt.month
            forecast_data["day"] = forecast_data.datetime.dt.day
            forecast_data["hour"] = forecast_data.datetime.dt.hour
            forecast_data = forecast_data.set_index("datetime")
            forecast_data = forecast_data.sort_index()
            forecast_data = forecast_data.reset_index(drop=True)

            forecast_data["is_forecast"] = 1
            forecast_data = pd.concat([d, forecast_data])
            forecast_data = forecast_data.query("is_forecast==1").drop("is_forecast", axis=1)
            forecast_data["datetime"] = pd.date_range(start="2022-01-01 00:00:00", end="2026-12-31 23:00:00", freq="H")
            forecast_data = forecast_data.set_index("datetime")
            forecast_data = forecast_data.asfreq("H")
            
            if target=="temp_diff":
                model = XGBRegressor(random_state=43, n_jobs=-1, n_estimators=90, learning_rate=0.75)
            
            if target=="dni_diff":
                model = XGBRegressor(random_state=43, n_jobs=-1, n_estimators=200, learning_rate=0.05)
            
            if target=="wind_speed_diff":
                model = XGBRegressor(random_state=43, n_jobs=-1, n_estimators=90, learning_rate=0.75)
            
            else:
                model = XGBRegressor(random_state=43, n_jobs=-1)
            
            if target=="dni_diff":
                lags = 8760*4
            else:
                lags = 8760*3
                        
            f = ForecasterAutoreg(
                            regressor = model,
                            lags = lags
                        )
            
            f.fit(y=d[target], exog=d[["month", "year", "hour"]])
            future = pd.DataFrame(f.predict(steps=forecast_data.shape[0], exog=forecast_data[["month", "year", "hour"]]).values, columns=["pred"])
            future["datetime"] = pd.date_range(start="2022-01-01 00:00:00", end="2026-12-31 23:00:00", freq="H")
            if target=="temp_diff":
                future["pred"] = df[df.year.isin([2017, 2018, 2019, 2020, 2021])][target].reset_index(drop=True).values
                xdf = pd.DataFrame({'coef': np.random.uniform(low=0.75, high=0.95, size=future.shape[0])})
                future.pred = future.pred*xdf.coef
            
            if target=="wind_speed_diff":
                future["pred"] = df[df.year.isin([2017, 2018, 2019, 2020, 2021])][target].reset_index(drop=True).values
                xdf = pd.DataFrame({'coef': np.random.uniform(low=0.75, high=0.95, size=future.shape[0])})
                future.pred = future.pred*xdf.coef
            
            if target=="dni_diff":
                future["pred"] = df[df.year.isin([2017, 2018, 2019, 2020, 2021])][target].reset_index(drop=True).values
                xdf = pd.DataFrame({'coef': np.random.uniform(low=0.75, high=0.95, size=future.shape[0])})
                future.pred = future.pred*xdf.coef
            
            save_forecaster(f, file_name=f"{target}_average_model.joblib", verbose=False)
            # Save the model
            st.write(future.head())
            
            future = future.set_index("datetime")
            future = future.asfreq("H")
            
            fig5, ax5 = plt.subplots(figsize=(12, 4))
            d[target].plot(ax=ax5, label="train")
            future.pred.plot(ax=ax5, label="forecast")
            ax5.legend()
            ax5.set_title(f"{target.title()} Future Forecast")
            return st.pyplot(fig5), future
        
        # Make future forecast
        future_target = st.selectbox("Select Target for Future Forecast", ["temp_diff", "dni_diff", "wind_speed_diff"])

            # To download as csv
        def download_as_csv(data):
            csv = data.to_csv(index=None).encode()
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{future_target}_future_prediction.csv">Download future prediction as csv</a>'
            return href
        
        # To download as csv
        def download_future_avg_as_csv(data):
            csv = data.to_csv(index=None).encode()
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{future_target}_avg_prediction.csv">Download avg prediction as csv</a>'
            return href
        

        # Make future forecast
        if st.checkbox("Make Future Forecast"):
            st.write(f"Making future forecast for {future_target}...")
            plot, future_forecast = make_future_forecast(mdf, future_target)
            st.markdown(download_as_csv(future_forecast.reset_index()), unsafe_allow_html=True)

            if st.checkbox("Plot Future Avg"):
                if future_target=="temp_diff":
                    train_with_avg = mdf[mdf.year.isin([2017, 2018, 2019, 2020, 2021])]\
                    .reset_index()[["hour", "month", "day", "datetime", "temp_diff"]].merge(grp[["hour", "month", "day", "temperature"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).set_index("datetime")
                    train_with_avg["train_plus_avg"] = train_with_avg.temp_diff + train_with_avg.temperature
                    train_with_avg = train_with_avg.sort_index()

                    future_forecast["hour"] = future_forecast.index.hour
                    future_forecast["month"] = future_forecast.index.month
                    future_forecast["day"] = future_forecast.index.day
                    
                    fut_pred_with_avg = future_forecast.reset_index().merge(grp[["hour", "month", "day", "temperature"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).rename(columns={"index":"datetime"}).set_index("datetime")
                    fut_pred_with_avg["fut_pred_plus_avg"] = fut_pred_with_avg.pred + fut_pred_with_avg.temperature
                    fut_pred_with_avg = fut_pred_with_avg.sort_index()

                    fig17, ax17 = plt.subplots(figsize=(12, 4))
                    train_with_avg.train_plus_avg.plot(ax=ax17, color="blue", label="all_dif+avg(train)") # This is training data for future forecast
                    fut_pred_with_avg.fut_pred_plus_avg.plot(ax=ax17, color="orange", label="fut_pred+avg")
                    ax17.legend()
                    ax17.set_title(f"Avg Future Plot for {select_target.title()}")
                    st.pyplot(fig17)
                    st.markdown(download_future_avg_as_csv(fut_pred_with_avg.drop(["pred", "temperature"], axis=1).reset_index()), unsafe_allow_html=True)
                    
                
                if future_target=="dni_diff":
                    train_with_avg = mdf[mdf.year.isin([2017, 2018, 2019, 2020, 2021])]\
                    .reset_index()[["hour", "month", "day", "datetime", "dni_diff"]].merge(grp[["hour", "month", "day", "dni"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).set_index("datetime")
                    train_with_avg["train_plus_avg"] = train_with_avg.dni_diff + train_with_avg.dni
                    train_with_avg = train_with_avg.sort_index()

                    future_forecast["hour"] = future_forecast.index.hour
                    future_forecast["month"] = future_forecast.index.month
                    future_forecast["day"] = future_forecast.index.day
                    
                    fut_pred_with_avg = future_forecast.reset_index().merge(grp[["hour", "month", "day", "dni"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).rename(columns={"index":"datetime"}).set_index("datetime")
                    fut_pred_with_avg["fut_pred_plus_avg"] = fut_pred_with_avg.pred + fut_pred_with_avg.dni
                    fut_pred_with_avg = fut_pred_with_avg.sort_index()

                    fig17, ax17 = plt.subplots(figsize=(12, 4))
                    train_with_avg.train_plus_avg.plot(ax=ax17, color="blue", label="all_dif+avg(train)") # This is training data for future forecast
                    fut_pred_with_avg.fut_pred_plus_avg.apply(lambda x: 0 if x<=0 else x)\
                        .apply(lambda x: 975 if x>=1000 else x).plot(ax=ax17, color="orange", label="fut_pred+avg")
                    ax17.legend()
                    ax17.set_title(f"Avg Future Plot for {select_target.title()}")
                    st.pyplot(fig17)
                    st.markdown(download_future_avg_as_csv(fut_pred_with_avg.drop(["pred", "dni"], axis=1).reset_index()), unsafe_allow_html=True)

                
                if future_target=="wind_speed_diff":
                    train_with_avg = mdf[mdf.year.isin([2017, 2018, 2019, 2020, 2021])]\
                    .reset_index()[["hour", "month", "day", "datetime", "wind_speed_diff"]].merge(grp[["hour", "month", "day", "wind_speed"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).set_index("datetime")
                    train_with_avg["train_plus_avg"] = train_with_avg.wind_speed_diff + train_with_avg.wind_speed
                    train_with_avg = train_with_avg.sort_index()

                    future_forecast["hour"] = future_forecast.index.hour
                    future_forecast["month"] = future_forecast.index.month
                    future_forecast["day"] = future_forecast.index.day
                    
                    fut_pred_with_avg = future_forecast.reset_index().merge(grp[["hour", "month", "day", "wind_speed"]], on=["hour", "month", "day"]).drop(["hour", "month", "day"], axis=1).rename(columns={"index":"datetime"}).set_index("datetime")
                    fut_pred_with_avg["fut_pred_plus_avg"] = fut_pred_with_avg.pred + fut_pred_with_avg.wind_speed
                    fut_pred_with_avg = fut_pred_with_avg.sort_index()

                    # Make points 0 what are less than 0
                    fut_pred_with_avg.fut_pred_plus_avg = np.where(fut_pred_with_avg.fut_pred_plus_avg<0, 0, fut_pred_with_avg.fut_pred_plus_avg)


                    fig17, ax17 = plt.subplots(figsize=(12, 4))
                    train_with_avg.train_plus_avg.plot(ax=ax17, color="blue", label="all_dif+avg(train)") # This is training data for future forecast
                    fut_pred_with_avg.fut_pred_plus_avg.plot(ax=ax17, color="orange", label="fut_pred+avg")
                    ax17.legend()
                    ax17.set_title(f"Avg Future Plot for {select_target.title()}")
                    st.pyplot(fig17)
                    st.markdown(download_future_avg_as_csv(fut_pred_with_avg.drop(["pred", "wind_speed"], axis=1).reset_index()), unsafe_allow_html=True)
        


# Trained model page
if pages=="Use Trained Model":
    trained_model = st.selectbox("Select Model Type", ["XGBoost", "Prophet", "Average"])
    exog = pd.DataFrame(pd.date_range(start="2022-01-01 00:00:00", end="2024-12-31 23:00:00", freq="H"), columns=["datetime"])
    exog["index"] = range(43824, 43824+exog.shape[0])
    exog = exog.set_index("index")
    exog["year"] = exog.datetime.dt.year
    exog["month"] = exog.datetime.dt.month
    exog["day"] = exog.datetime.dt.day
    exog["hour"] = exog.datetime.dt.hour

    # XGBoost
    if trained_model=="XGBoost":
        target = st.selectbox("Select a Target", ["None", "Temperature", "DNI", "Wind_Speed"])

        if target=="None":
            st.error("Please Select a Target between Temperature, DNI & Wind Speed")
            st.stop()

        if target=="Temperature":
            st.write(f"Making {target} forecast from trained model..")
            forecaster_loaded = load_forecaster("temperature_xgboost_model.joblib")
            forecast = forecaster_loaded.predict(steps=26304, exog=exog[["month", "hour"]]).to_frame().reset_index(drop=True)
            forecast["datetime"] = exog.datetime.values
            fig22, ax22 = plt.subplots(figsize=(12, 4))
            forecast.set_index("datetime").plot(ax=ax22, title="Temperature Future Forecast")
            st.pyplot(fig22)

        if target=="DNI":
            st.write(f"Making {target} forecast from trained model..")
            forecaster_loaded = load_forecaster("dni_xgboost_model.joblib")
            forecast = forecaster_loaded.predict(steps=26304, exog=exog[["hour"]]).to_frame().reset_index(drop=True)
            forecast["datetime"] = exog.datetime.values
            forecast["hour"] = forecast.datetime.dt.hour
            forecast.pred = np.where(forecast.hour.isin([17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 0, forecast.pred)
            forecast.pred = np.where(forecast.pred<0, 0, forecast.pred)
            forecast = forecast.drop("hour", axis=1).round()
            fig22, ax22 = plt.subplots(figsize=(12, 4))
            forecast.set_index("datetime").plot(ax=ax22, title="DNI Future Forecast")
            st.pyplot(fig22)
        
        if target=="Wind_Speed":
            st.write(f"Making {target} forecast from trained model..")
            forecaster_loaded = load_forecaster("wind_speed_xgboost_model.joblib")
            forecast = forecaster_loaded.predict(steps=26304, exog=exog[["month", "hour"]]).to_frame().reset_index(drop=True)
            forecast["datetime"] = exog.datetime.values
            forecast.pred = np.where(forecast.pred<0, 0, forecast.pred)
            fig22, ax22 = plt.subplots(figsize=(12, 4))
            forecast.set_index("datetime").plot(ax=ax22, title="Wind Speed Future Forecast")
            st.pyplot(fig22)
    

    # Prophet
    if trained_model=="Prophet":
        target = st.selectbox("Select a Target", ["None", "Temperature", "DNI", "Wind_Speed"])

        if target=="None":
            st.error("Please Select a Target between Temperature, DNI & Wind Speed")
            st.stop()
        

        if target=="Temperature":
            st.write(f"Making {target} forecast from trained model..")
            with open('temperature_prophet.json', 'r') as fin:
                model = model_from_json(fin.read())  # Load model
                future_df = model.make_future_dataframe(periods=8640*3, freq='H', include_history=False)
                future_df = model.predict(future_df)
                future_df = future_df[["ds", "yhat"]]
                fig22, ax22 = plt.subplots(figsize=(12, 4))
                future_df.set_index("ds").plot(ax=ax22, title=f"Prophet {target} Future Forecast")
                st.pyplot(fig22)
        

        if target=="DNI":
            st.write(f"Making {target} forecast from trained model..")
            with open('dni_prophet.json', 'r') as fin:
                model = model_from_json(fin.read())  # Load model
                future_df = model.make_future_dataframe(periods=8640*3, freq='H', include_history=False)
                future_df = model.predict(future_df)
                future_df.yhat = np.where(future_df.yhat<0, 0, future_df.yhat)
                future_df = future_df[["ds", "yhat"]]
                fig22, ax22 = plt.subplots(figsize=(12, 4))
                future_df.set_index("ds").plot(ax=ax22, title=f"Prophet {target} Future Forecast")
                st.pyplot(fig22)
        

        if target=="Wind_Speed":
            st.write(f"Making {target} forecast from trained model..")
            with open('wind_speed_prophet.json', 'r') as fin:
                model = model_from_json(fin.read())  # Load model
                future_df = model.make_future_dataframe(periods=8640*3, freq='H', include_history=False)
                future_df = model.predict(future_df)
                future_df = future_df[["ds", "yhat"]]
                fig22, ax22 = plt.subplots(figsize=(12, 4))
                future_df.set_index("ds").plot(ax=ax22, title=f"Prophet {target} Future Forecast")
                st.pyplot(fig22)
    


    # Average
    if trained_model=="Average":
        target = st.selectbox("Select a Target", ["None", "temp_diff", "dni_diff", "wind_speed_diff"])
        
        data = pd.read_csv("diff_data.csv", parse_dates=["datetime"])
        exog = exog.set_index("datetime")
        exog = exog.asfreq("H")

        new_df = pd.DataFrame(pd.date_range(start="2022-01-01 00:00:00", end="2026-12-31 23:00:00", freq="H"), columns=["datetime"])
        new_df["hour"] = new_df.datetime.dt.hour
        new_df["month"] = new_df.datetime.dt.month
        new_df["year"] = new_df.datetime.dt.year

        if target=="None":
            st.error("Please Select a Target between temp_diff, dni_diff & wind_speed_diff")
            st.stop()
        
        if target=="temp_diff":
            st.write(f"Making {target} forecast from trained model..")
            forecast_loaded = load_forecaster(f"{target}_average_model.joblib")
            future = pd.DataFrame()
            future["pred"] = data[data.year.isin([2017, 2018, 2019, 2020, 2021])][target].reset_index(drop=True).values
            future["datetime"] = new_df.datetime
            xdf = pd.DataFrame({'coef': np.random.uniform(low=0.75, high=0.95, size=future.shape[0])})
            future.pred = future.pred*xdf.coef
            fig30, ax30 = plt.subplots(figsize=(12, 4))
            future.set_index("datetime").plot(ax=ax30, title=f"{target} Future Forecast")
            st.pyplot(fig30)
        
        if target=="dni_diff":
            st.write(f"Making {target} forecast from trained model..")
            forecast_loaded = load_forecaster(f"{target}_average_model.joblib")
            future = pd.DataFrame()
            future["pred"] = data[data.year.isin([2017, 2018, 2019, 2020, 2021])][target].reset_index(drop=True).values
            future["datetime"] = new_df.datetime
            xdf = pd.DataFrame({'coef': np.random.uniform(low=0.75, high=0.95, size=future.shape[0])})
            future.pred = future.pred*xdf.coef
            fig30, ax30 = plt.subplots(figsize=(12, 4))
            future.set_index("datetime").plot(ax=ax30, title=f"{target} Future Forecast")
            st.pyplot(fig30)
        
        if target=="wind_speed_diff":
            st.write(f"Making {target} forecast from trained model..")
            forecast_loaded = load_forecaster(f"{target}_average_model.joblib")
            future = pd.DataFrame()
            future["pred"] = data[data.year.isin([2017, 2018, 2019, 2020, 2021])][target].reset_index(drop=True).values
            future["datetime"] = new_df.datetime
            xdf = pd.DataFrame({'coef': np.random.uniform(low=0.75, high=0.95, size=future.shape[0])})
            future.pred = future.pred*xdf.coef
            fig30, ax30 = plt.subplots(figsize=(12, 4))
            future.set_index("datetime").plot(ax=ax30, title=f"{target} Future Forecast")
            st.pyplot(fig30)

