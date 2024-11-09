# Define a dictionary to map actual column names to display names
column_name_map = {
    'Solution_Index': 'Solution',
    'n_wt': 'Wind Turbines',
    'n_csp': 'CSP Units',
    'n_pv': 'PV Panels',
    'n_batt': 'Batteries',
    'Total_Energy_Cost': 'Total Cost ($)',
    'Total_CO2': 'Total CO2 (ton)',
    'Demand': 'Demand (kWh)',
    'Util_Energy': 'Utility Energy (kWh)',
    'Batt_Charge': 'Battery Charge (kWh)',
    'Total_RE_Prod': 'Total RE Prod. (kWh)',
    'RE_Cost': 'RE Cost ($)',
    'Util_Cost': 'Utility Cost ($)',
    'Batt_Cost': 'Battery Cost ($)',
    'RE_CO2': 'RE CO2 (ton)',
    'Util_CO2': 'Utility CO2 (ton)',
    'Batt_CO2': 'Battery CO2 (ton)',
    'RE/Demand': 'RE/Demand',
    'RE+BU/Demand': '(RE+BU)/Demand'
}

upload_style = {
    'width': '100%',
    'height': '75px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '5px',
    'fontSize': '0.9em',
}

dd_style = {
    'width': '170px',
    'height': '30px',
    'fontSize': '0.9em',
}

dd_style2 = {
    'width': '270px',
    'height': '40px',
    'fontSize': '0.9em',
}

cc_style = {
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center',
}

small_button_style = {
    'width': '70%',
    'height': '75px',
    'background-color': 'lightblue',
    'font-size': '11px',
    'padding': '15px 10px',
    'margin': '0px',
    'border': 'none',
    'border-radius': '3px',
    'cursor': 'pointer'
}

# More customized tab styles
tab_style = {
    'font-size': '18px',
    'font-weight': 'bold',
    'padding': '10px 15px',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#f9f9f9',
    'margin-bottom': '20px'
}

tab_selected_style = {
    'font-size': '18px',
    'font-weight': 'bold',
    'padding': '10px 15px',
    'borderTop': '3px solid #1890ff',  # Blue line on top when selected
    'borderBottom': '1px solid white',
    'backgroundColor': '#ffffff',
    'color': '#1890ff',  # Text becomes blue when selected
    'margin-bottom': '20px'
}

main_button_style ={
    'background-color': 'darkorange', 
    'font-size': '14px', 
    'font-weight': 'bold',
    'padding': '15px 15px',  # Add some padding for better appearance
    'border': 'none',  # Remove default border
    'cursor': 'pointer',  # Change cursor on hover
    'border-radius': '5px',  # Rounded corners     
    'width': '165px',
    'height': '45px'                            
}

aux_button_style = {
    'background-color': 'darkorange',
    'font-size': '12px',
    'padding': '10px 20px',
    'margin': '15px',
    'border': 'none',
    'border-radius': '8px',
    'cursor': 'pointer',
    'width': '145px',
    'height': '40px',
    'font-weight': 'bold',
    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',  # Add shadow
    'transition': 'all 0.3s ease',    # Smooth transition for hover effect
    #'text-transform': 'uppercase'     # Make text uppercase
}

# You can also add hover effect by adding this to your app's CSS:
# Note: This would need to be added to a separate CSS file or assets folder
"""
#store-forecast-button:hover {
    background-color: #ff8c00;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}
"""