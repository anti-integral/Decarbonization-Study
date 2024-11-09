import pandas as pd
import numpy as np

def test_re_calculations():
    """
    Test RE calculations against sample data
    """
    # Create sample data from the spreadsheet
    data = {
        'datetime': ['1/1/23 0:00', '1/1/23 7:00', '1/1/23 8:00', '1/1/23 14:00', '1/1/23 17:00'],
        'dni': [0.0, 0.031219, 0.165352, 0.105942, 0.010289],
        'temperature': [15.19122924, 14.48516369, 14.35907726, 12.57489905, 8.15464205],
        'wind_speed': [2.93107763, 7.0672138, 9.32052648, 10.0805553, 11.6256762]
    }
    df = pd.DataFrame(data)
    
    # Calculate RE production using our function
    df_calc = calculate_re_production(df)
    
    # Print comparison
    print("\nPV Power Comparison:")
    print("Time             DNI     Temp    Calc Ppv   Excel Ppv")
    print("-" * 55)
    for idx, row in df.iterrows():
        print(f"{row['datetime']:12} {row['dni']:8.6f} {row['temperature']:8.2f} {df_calc['Ppv'][idx]:10.3f}  ")
    
    print("\nWind Power Comparison:")
    print("Time             Wind Speed  Calc Pw    Excel Pw")
    print("-" * 50)
    for idx, row in df.iterrows():
        print(f"{row['datetime']:12} {row['wind_speed']:10.2f} {df_calc['Pw'][idx]:10.2f}")

    return df_calc


def calculate_re_production(weather_data):
    """
    Calculate renewable energy production based on weather forecasts
    and technical specifications, matching Excel calculations
    """
    df = weather_data.copy()
    
    # PV calculations
    n_inv = 0.95  # Inverter efficiency
    n_b = 1.00    # Battery efficiency
    n_r = 0.225   # Rated solar cell efficiency
    beta = -0.0037  # Temperature Coefficient of Efficiency
    A_pv = 3.1    # Area of each module (m2)
    NOCT = 44     # Nominal Operating Cell Temperature (°C)
    P_rated_pv = 0.7  # Power rating of PV panel (kW)
    
    # Calculate cell temperature
    df['Tcell'] = df['temperature'] + 1 * (NOCT-20)/800
    df['Tc'] = (1 - beta * (df['Tcell'] - 25))
    
    # Calculate PV power output per panel
    # DNI is in kWh/m2/day - convert to hourly power
    df['Ppv'] = (n_inv * n_b * n_r * df['Tc'] * A_pv * df['dni'] * 1000) / 24  # Convert to hourly kW
    df['Ppv'] = df['Ppv'].clip(lower=0, upper=P_rated_pv)
    
    # Wind calculations
    rho = 1.225   # Air density (kg/m3)
    A_wind = 11310  # Rotor swept area (m2)
    Cp = 0.28     # Power coefficient
    P_rated_wind = 2500  # Power rating of wind turbine (kW)
    
    # Calculate wind power output per turbine (in kW)
    df['Pw'] = (0.5 * rho * A_wind * df['wind_speed']**3 * Cp) / 1000
    df['Pw'] = df['Pw'].clip(lower=0, upper=P_rated_wind)
    
    return df