import base64
import io
import zipfile
import dash
from dash import Dash, dcc
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from main_opt import run_optimization, process_results
from utility import load_preset_data, calculate_re_production
from weather_forecast import WeatherForecast
from styles import *
from IPython import embed

app = dash.Dash(__name__)

all_solutions = []
processed_solutions = pd.DataFrame()

# Global variables to store preset data
preset_re_data, preset_demand_data, preset_constants = load_preset_data()


app.layout = dbc.Container([
    html.H1("Multi-Objective Optimization for Renewable Energy"),
    
    dcc.Tabs([

        dcc.Tab(label='Optimization', children=[

            html.Br(),
            html.Br(),

            html.Div(className="four columns", children=[
                dbc.Row([
                    html.Div([
                        html.H4('Upload RE Prod Data'),
                        dbc.Col(dcc.Upload(id='upload-re-data', children=html.Div(['Drag and Drop or Browse Data File', 
                                        #html.A('Select RE Production Data File')
                                        ]), 
                                        style=upload_style, multiple=False)),
                    ], className="mb-4"),

                    html.Div([
                        html.H4('Upload Plant Demand Data'),
                        dbc.Col(dcc.Upload(id='upload-demand-data', children=html.Div(['Drag and Drop or Browse Data File', 
                                    #html.A('Select Plant Demand Data File')
                                    ]), 
                                    style=upload_style, multiple=False)),
                    ], className="mb-4 mx-4"),

                    html.Div([
                        html.H4('Upload Constants'),
                        dbc.Col(dcc.Upload(id='upload-constants', children=html.Div(['Drag and Drop or Browse Data File', 
                                    #html.A('Select Constants File')
                                    ]), 
                                    style=upload_style, multiple=False)),
                    ], className="mb-4"),

                    html.Div([
                        html.H4('Data Sample'),
                        html.Button('Download Data Sample', 
                            id='download-presets-button', 
                            style=small_button_style,
                            n_clicks=0),
                        dcc.Download(id='download-presets'),
                    ], className="mb-4"),

                    html.Br(),
                    html.Div(id='upload-count', children='Files uploaded: 0', className="mt-3", 
                            style = {'font-size': '16px', 'color':'blue'}),

                ], style={'display': 'flex', 'justifyContent': 'space-evenly' }, className="justify-content-between" ),

                html.Br(),
                html.Div(id='upload-status', className="mt-3", style = {'font-size': '14px', 'color':'blue'}),

                dbc.Row([
                    html.Div([
                        html.H4('Scenario'),
                        dbc.Col(dcc.Dropdown(id='scenario-dropdown', 
                                            style=dd_style,
                                            options=[{'label': 'Battery Usage', 'value': 'battery'}, 
                                                    {'label': 'Utility Usage', 'value': 'utility'}
                                                    ], 
                                                    value='battery')
                                ),
                    ]),

                    html.Div([
                        html.H4('RE Sources'),
                        dbc.Col(dcc.Dropdown(id='re-sources-dropdown', 
                                            style=dd_style,
                                            options=[
                                                {'label': 'All (WT, PV, CSP)', 'value': 'all'},
                                                {'label': 'WT and PV', 'value': 'wt_pv'},
                                                {'label': 'WT and CSP', 'value': 'wt_csp'},
                                                {'label': 'PV and CSP', 'value': 'pv_csp'}
                                            ],
                                            value='all')
                                ),
                    ]),

                    html.Div([
                        html.H4('Time Granularity'),
                        dbc.Col(dcc.Dropdown(
                            id='time-granularity-dropdown', 
                            style=dd_style,
                            options=[
                                {'label': 'Daily', 'value': 'daily'},
                                {'label': 'Hourly', 'value': 'hourly'}
                            ],
                            value='daily'
                        )),
                    ]),

                    html.Div([
                        html.H4('RE Use: Min'),
                        dbc.Col(dcc.Dropdown(id='re-min', 
                                            style=dd_style,
                                            options=[0,0.25,0.5,0.99], 
                                            value= 0.5)),
                    ]),

                    html.Div([
                        html.H4('RE Use: Max'),
                        dbc.Col(dcc.Dropdown(id='re-max', 
                                            style=dd_style,
                                            options=[0.25,0.5,1.0], 
                                            value= 1.0)),
                    ]),

                    html.Div([
                        html.H4('Number of generations'),
                        dbc.Col(dcc.Dropdown(id='num-gen', 
                                            style=dd_style,
                                            options=[25, 50, 100, 200], 
                                            value= 25)),
                    ]),

                ], style={'display': 'flex', 'justifyContent': 'center'}), #'space-between'
            ]),
            
            html.Br(),
            html.Br(),
            html.Br(),
            dcc.Loading(
                    id="loading",
                    type='default', #"circle",
                    children=[

                        html.Div([
                            html.Div(id="loading-output", 
                                    style = {'font-size': '16px', 'color':'blue'}),
                            ], style = cc_style
                        ),

                        html.Br(),
                        html.Div([
                            html.Button('Run Optimization', 
                                        id='run-button', 
                                        style=main_button_style,
                                        n_clicks=0),
                            ], style  = cc_style
                        ),

                        html.Br(),
                        html.Div([
                            html.Div(id="optimization-results", 
                                    style = {'font-size': '14px', 'color':'blue'}),
                            ], style = cc_style
                        ),

                        html.Br(),
                        html.Br(),
                        html.Div([
                            html.H3("Pareto Front For Optimized Solutions"),
                            dcc.Graph(id='pareto-plot', style={'width': '60vh', 'height': '50vh'}),
                        ]),

                        html.Br(),
                        html.Div([
                            html.H3("All Solutions"),
                            dcc.Checklist(id='select-all-checklist',
                                        options=[{'label': 'Select All', 'value': 'select_all'}],
                                        value=[],
                            ),
                            dash_table.DataTable(
                                id='all-solutions-table',
                                #columns=[{'name': i, 'id': i} for i in ['Solution_Index', 'n_wt', 'n_csp', 'n_pv', 'n_batt', 'Total_Energy_Cost', 'Total_CO2']],
                                columns=[{'name': column_name_map.get(i, i), 'id': i} 
                                        for i in ['Solution_Index', 'n_wt', 'n_csp', 'n_pv', 'n_batt', 'Total_Energy_Cost', 'Total_CO2']],
                                style_cell={'textAlign': 'left', 'font-size': '12px'},
                                style_table={'height': '250px', 'overflowY': 'auto'},
                                row_selectable='multi',
                                selected_rows=[],
                                sort_action='native',
                                sort_mode='single',
                            )
                        ])
                    
                    ]
            ),

            
            html.Br(),
            html.Button('Show Details', 
                        id='get-results-button', 
                        style=aux_button_style,
                        n_clicks=0),
            
            html.Br(),
            html.Br(),
            html.Div(id='selected-solutions-results', style = {'font-size': '14px','color':'blue'}),

            html.Br(),
            html.Br(),
            html.Div([
                html.H4(f"Detailed Results for Selected Solutions"),
                dash_table.DataTable(
                    id='selected-solutions-table',
                    columns=[],  # We'll set this dynamically in the callback
                    style_cell={'textAlign': 'left', 'font-size': '12px'},
                    style_table={'overflowX': 'auto'},
                    sort_action='native',
                    sort_mode='single'
                )
            ]),

            html.Br(),
            html.Button('Download Results', 
                        id='download-button', 
                        style=aux_button_style, 
                        n_clicks=0),

            dcc.Download(id='download-results'),

            html.Br(),
            html.Br(),
            html.Br(),
            html.Br()

    ],
        style=tab_style,
        selected_style=tab_selected_style
    ),

    dcc.Tab(label='Weather Forecast', children=[


        #html.Div(className="four columns", children=[

        dbc.Row([

            html.Div([
                html.H4('Location'),
                dbc.Col(dcc.Dropdown(
                    id='location-dropdown',
                    options=[
                        {"label": "Tampa (27.960981, -82.345212)", "value": "loc0"},
                        {"label": "Location 1 (37.505290, -121.960910)", "value": "loc1"},
                        {"label": "Location 2 (29.458136, -98.479810)", "value": "loc2"},
                        {"label": "Location 3 (31.804775, -106.327605)", "value": "loc3"},
                        {"label": "Location 5 (41.184520, -73.802961)", "value": "loc4"},
                        
                    ],
                    value='loc0',
                    style=dd_style2
                )),
            ]),

            html.Div([
                html.H4('Forecast Years'),
                dbc.Col(dcc.Dropdown(
                    id='forecast-years-dropdown',
                    options=[
                        {'label': f'{i} years', 'value': i} for i in range(1, 6)
                    ],
                    value=1,
                    style=dd_style2
                )),

            ])
        ], style={'display': 'flex', 'justifyContent': 'center'}), 
        #]),

        html.Br(),
        html.Br(),
        html.Div([
            html.Button('Get Predictions', 
                        id='forecast-button', 
                        style=main_button_style,
                        n_clicks=0),
            ], style  = cc_style
        ),

        html.Br(),
        html.Div([
            html.Div(id='forecast-status', style={'font-size': '14px', 'color': 'blue'}),
        ], style=cc_style),

        dcc.Loading(
            id="forecast-loading",
            type='default',
            children=[
                html.Div([
                    html.Div(id='forecast-results'),
                ], style=cc_style),
                html.Div([
                    dcc.Graph(id='forecast-plot')
                ], style=cc_style),
                # Add Store button after plots
                html.Br(),
                html.Div([
                    html.Button(
                        'Store Forecast',
                        id='store-forecast-button',
                        style=aux_button_style,
                        n_clicks=0
                    ),
                ], style=cc_style),
                dcc.Download(id='download-forecast')
            ]
        )

    ],
        style=tab_style,
        selected_style=tab_selected_style
    )

  ])

], fluid=True)

@app.callback(
    Output('download-forecast', 'data'),
    Input('store-forecast-button', 'n_clicks'),
    State('forecast-results', 'children'),
    prevent_initial_call=True
)
def store_forecast(n_clicks, forecast_results):
    if n_clicks == 0 or not forecast_results:
        return None
        
    if 'stored_forecasts' not in globals():
        return None
        
    forecasts = stored_forecasts
    
    # Combine weather forecasts into a single DataFrame
    df = pd.DataFrame({
        'datetime': forecasts['temp_diff']['datetime'],
        'temperature': forecasts['temp_diff']['abs_pred'],
        'dni': forecasts['dni_diff']['abs_pred'],
        'wind_speed': forecasts['wind_speed_diff']['abs_pred']
    })
    
    # Calculate RE production
    df_with_re = calculate_re_production(df)
    
    # Add simple statistics to results_div
    stats_div = html.Div([
        html.H4('Average Power Production per Unit:'),
        html.Ul([
            html.Li(f"PV Panel: {df_with_re['Ppv'].mean():.2f} kW"),
            html.Li(f"Wind Turbine: {df_with_re['Pw'].mean():.2f} kW"),
            html.Li(f"CSP Unit: {df_with_re['Pcs'].mean():.2f} kW")
        ])
    ])
    
    return dcc.send_data_frame(df_with_re.to_csv, "weather_and_re_forecast.csv", index=False)

@app.callback(
    [Output('forecast-status', 'children'),
     Output('forecast-results', 'children'),
     Output('forecast-plot', 'figure')],
    [Input('forecast-button', 'n_clicks')],
    [State('location-dropdown', 'value'),
     State('forecast-years-dropdown', 'value')]
)
def update_forecast(n_clicks, location, forecast_years):

    global stored_forecasts

    if n_clicks == 0:
        return "Select location and forecast period, then click 'Get Predictions'.", None, {}

    # Location coordinates mapping
    location_coords = {
        'loc0': (27.960981, -82.345212),
        'loc1': (37.505290, -121.960910),
        'loc2': (29.458136, -98.479810),
        'loc3': (31.804775, -106.327605),
        'loc4': (41.184520, -73.802961)
    }

    lat, lon = location_coords[location]
    
    # Initialize WeatherForecast
    wf = WeatherForecast()
    
    # Fetch historical data (last 3 years)
    years = list(range(2020, 2023))
    success = wf.fetch_data(years, lat, lon)
    
    if not success:
        return "Error fetching weather data. Please try again.", None, {}

    # Make forecasts for temperature, DNI, and wind speed
    forecasts = {}
    for target in ['temp_diff', 'dni_diff', 'wind_speed_diff']:
        forecasts[target] = wf.make_forecast(target, n_years=forecast_years, plot=False)
        
        # Convert DNI from W/m² to kWh/m²/hr
        if target == 'dni_diff':
            forecasts[target]['abs_pred'] = forecasts[target]['abs_pred'] / 1000  # Convert W/m² to kW/m²
            forecasts[target]['abs_pred'] = forecasts[target]['abs_pred']  # Already per hour

    # Create subplot figure
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=('Temperature Forecast', 
                       'Direct Normal Irradiance (DNI) Forecast',
                       'Wind Speed Forecast'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]],
        row_heights=[0.33, 0.33, 0.33]
    )

    # Update y-axis labels with correct units
    forecast_units = {
        'temp_diff': '°C',
        'dni_diff': 'kWh/m^2/hr',
        'wind_speed_diff': 'm/s'
    }

    # Add each forecast to its own subplot
    for idx, (target, data) in enumerate(forecasts.items(), 1):
        fig.add_trace(
            go.Scatter(
                x=data['datetime'],
                y=data['abs_pred'],
                name=f'{target.replace("_diff", "")} forecast'
            ),
            row=idx, 
            col=1
        )

    # Update layout
    fig.update_layout(
        height=1100,  # Fixed height
        width=1000,   # Fixed width
        showlegend=False,
        title_text=f"Weather Forecasts for Location {location}",
        title_x=0.5,
        margin=dict(t=100, b=50, l=80, r=20),
        autosize=False,  # Disable autosize
        template='plotly_white'
    )
    
    # Update y-axes labels with correct units
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="DNI (kWh/m^2/hr)", row=2, col=1)
    fig.update_yaxes(title_text="Wind Speed (m/s)", row=3, col=1)
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Date", row=3, col=1)

    # Create results summary
    results_div = html.Div([
        html.H3('Forecast Summary'),
        html.Div([
            html.P(f"Location: {location} ({lat}, {lon})"),
            html.P(f"Forecast Period: {forecast_years} years"),
            html.H4('Average Values:'),
            html.Ul([
                html.Li(f"Temperature: {forecasts['temp_diff']['abs_pred'].mean():.2f}°C"),
                html.Li(f"DNI: {forecasts['dni_diff']['abs_pred'].mean():.4f} kWh/m^2/hr"),  # More decimal places for DNI
                html.Li(f"Wind Speed: {forecasts['wind_speed_diff']['abs_pred'].mean():.2f} m/s")
            ])
        ])
    ])

    # Store forecasts globally before returning
    stored_forecasts = forecasts

    return "Forecast generated successfully!", results_div, fig



@app.callback(
    Output('download-presets', 'data'),
    Input('download-presets-button', 'n_clicks'),
    prevent_initial_call=True
)
def download_presets(n_clicks):
    if n_clicks == 0:
        return dash.no_update

    # Create a BytesIO object to store the zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr('re_prod_data.csv', preset_re_data.to_csv(index=False))
        zip_file.writestr('plant_demand_data.csv', preset_demand_data.to_csv(index=False))
        zip_file.writestr('constants_data.csv', preset_constants.to_csv(index=False))

    zip_buffer.seek(0)
    return dcc.send_bytes(zip_buffer.getvalue(), "preset_data.zip")


@app.callback(
    [Output('upload-count', 'children'),
     Output('upload-status', 'children'),
     Output('loading-output', 'children')],  # Add this output
    [Input('upload-re-data', 'contents'),
     Input('upload-demand-data', 'contents'),
     Input('upload-constants', 'contents')],
    [State('upload-re-data', 'filename'),
     State('upload-demand-data', 'filename'),
     State('upload-constants', 'filename')]
)
def update_upload_info(re_contents, demand_contents, constants_contents,
                       re_filename, demand_filename, constants_filename):
    count = sum(1 for contents in [re_contents, demand_contents, constants_contents] if contents is not None)
    
    status_messages = []
    all_valid = True
    
    if re_contents:
        status = check_csv_file(re_contents, re_filename, 're_prod')
        status_messages.append(status)
        all_valid = all_valid and 'Valid' in status
    
    if demand_contents:
        status = check_csv_file(demand_contents, demand_filename, 'demand')
        status_messages.append(status)
        all_valid = all_valid and 'Valid' in status
    
    if constants_contents:
        status = check_csv_file(constants_contents, constants_filename, 'constants')
        status_messages.append(status)
        all_valid = all_valid and 'Valid' in status
    
    status_html = html.Ul([html.Li(message) for message in status_messages])
    
    print('count, all_valid',count, all_valid)

    loading_message = "Ready for input"
    if count == 3 and all_valid:
        loading_message = "Data loaded successfully"
    
    return f'Files uploaded: {count}', status_html, loading_message


def check_csv_file(contents, filename, file_type):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    required_columns = {
        're_prod': ['datetime', 'Ppv', 'Pw', 'Pcs'],
        'demand': ['datetime', 'PF'],
        'constants': ['Constant', 'Value']
    }
    
    required_constants = [
        'LCOE_pv', 'LCOE_wind', 'LCOE_csp', 'LCOE_batt', 'LCOE_util',
        'CO2_pv', 'CO2_wind', 'CO2_csp', 'CO2_batt', 'CO2_util',
        'BATTERY_CAPACITY', 'Ndays_per_month', 'Cost_day_pv', 'Cost_day_wind',
        'Cost_day_csp', 'LCOE_csp_kwt', 'CO2_ton_per_gal'
    ]
    
    if set(required_columns[file_type]).issubset(df.columns):
        if len(df) > 10:
            if file_type == 'constants':
                missing_constants = set(required_constants) - set(df['Constant'])
                if not missing_constants:
                    return f"{file_type}, {filename}: Valid. All required columns and constants present."
                else:
                    return f"{file_type}, {filename}: Missing constants: {', '.join(missing_constants)}"
            else:
                return f"{file_type}, {filename}: Valid. All required columns present."
        else:
            return f"{file_type}, {filename}: Invalid. Less than 10 rows."
    else:
        missing_columns = set(required_columns[file_type]) - set(df.columns)
        return f"{file_type}, {filename}: Missing columns: {', '.join(missing_columns)}"



def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None
    except Exception as e:
        print(e)
        return None
    return df

@app.callback(
    [Output('loading-output', 'children', allow_duplicate=True),
     Output('optimization-results', 'children'),
     Output('pareto-plot', 'figure'),
     Output('all-solutions-table', 'data')],
    Input('run-button', 'n_clicks'),
    [State('upload-re-data', 'contents'),
     State('upload-re-data', 'filename'),
     State('upload-demand-data', 'contents'),
     State('upload-demand-data', 'filename'),
     State('upload-constants', 'contents'),
     State('upload-constants', 'filename'),
     State('scenario-dropdown', 'value'),
     State('re-sources-dropdown', 'value'),
     State('time-granularity-dropdown', 'value'),
     State('re-min', 'value'),
     State('re-max', 'value'),
     State('num-gen', 'value')],
    prevent_initial_call=True
)
def run_optimization_callback(n_clicks, re_contents, re_filename, demand_contents, demand_filename, 
                              constants_contents, constants_filename, scenario, re_sources, 
                              time_granularity, re_min, re_max, num_gen):
    global all_solutions, processed_solutions
    
    if n_clicks == 0:
        return "Data loaded successfully", "Upload data and click 'Run Optimization' to see results.", {}, []

    re_data = parse_contents(re_contents, re_filename)
    #if len(re_data)>1 and set(re_data.columns)
    demand_data = parse_contents(demand_contents, demand_filename)
    constants = parse_contents(constants_contents, constants_filename)
    
    if re_data is None or demand_data is None or constants is None:
        return "Error: Unable to parse uploaded files. Please check the file formats.", {}, []
    
    constants_dict = dict(zip(constants['Constant'], constants['Value']))

    data_sel_agg = process_input_data(re_data, demand_data, time_granularity)
    print('Start optimization...')
    all_solutions = run_optimization(data_sel_agg, constants_dict, scenario, re_sources, re_min, re_max, num_gen, time_granularity)
    print('Optimization is done.')

    processed_solutions = pd.concat([process_results(sol, scenario) for sol in all_solutions])
    processed_solutions['Solution_Index'] = range(len(processed_solutions))
    processed_solutions.drop_duplicates(subset=['Total_Energy_Cost','Total_CO2'],inplace=True)

    print('len(processed_solutions):',len(processed_solutions))
    print(processed_solutions)

    fig = px.scatter(processed_solutions, 
                     x='Total_Energy_Cost', 
                     y='Total_CO2',
                     hover_data=['Solution_Index', 'n_wt', 'n_csp', 'n_pv','n_batt','Total_Energy_Cost', 'Total_CO2']
                     )

    fig.update_layout(#title='Pareto Front For Optimized Solutions', 
                      xaxis_title='Total Energy Cost ($)', 
                      yaxis_title='Total CO2 (ton)',
                      width=700,
                      height=500)
    
    TE_Cost = processed_solutions['Total_Energy_Cost']
    T_Co2 = processed_solutions['Total_CO2']
    xmin, xmax = 0.5*min(TE_Cost),1.5*max(TE_Cost)
    ymin, ymax = 0.5*min(T_Co2),1.5*max(T_Co2)

    fig.update_layout(xaxis_range=[xmin, xmax])
    fig.update_layout(yaxis_range=[ymin, ymax])
    fig.update_layout(margin={"r":0,"t":0,"l":100,"b":20})

    table_data = processed_solutions[['Solution_Index', 'n_wt', 'n_csp', 'n_pv','n_batt','Total_Energy_Cost', 'Total_CO2']].to_dict('records')

    return 'Process completed','Optimization finished successfully.', fig, table_data


@app.callback(
    Output('all-solutions-table', 'selected_rows'),
    Input('select-all-checklist', 'value'),
    State('all-solutions-table', 'data')
)
def select_all(selected, data):
    if selected and 'select_all' in selected:
        return list(range(len(data)))
    return []

@app.callback(
    Output('all-solutions-table', 'selected_rows',allow_duplicate=True),
    Input('run-button', 'n_clicks'),
    State('all-solutions-table', 'data'),
    prevent_initial_call=True
)
def update_checklist(n_clicks, data):
    print('data',data)
    # if n_clicks == 0:
    #     return data
    return []


@app.callback(
    Output('selected-solutions-results', 'children'),
    Output('selected-solutions-table', 'data'),
    Output('selected-solutions-table', 'columns'),
    [Input('get-results-button', 'n_clicks')],
    [State('all-solutions-table', 'selected_rows'),
     State('scenario-dropdown', 'value'),
    ]
)
def display_selected_solutions(n_clicks, selected_rows, scenario):
    if n_clicks == 0 or not selected_rows:
        # Return empty data and columns when no selection
        return "Select one or more solutions and click 'Show Details' to see detailed results.", [], []

    selected_results = processed_solutions[processed_solutions.Solution_Index.isin(selected_rows)]

    columns = ['Solution_Index'] + list(selected_results.columns[:-1])
    
    # Create the columns definition with mapped names
    columns_definition = [{'name': column_name_map.get(col, col), 'id': col} for col in columns]

    return 'Results are ready', selected_results[columns].to_dict('records'), columns_definition


@app.callback(
    Output('download-results', 'data'),
    Input('download-button', 'n_clicks'),
    State('selected-solutions-table', 'data'),
    prevent_initial_call=True
)
def download_results(n_clicks, table_data):
    if not table_data:
        return None
    
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_csv, "selected_solutions_results.csv", index=False)


def process_input_data(re_data, demand_data, time_granularity):
    re_data['datetime'] = pd.to_datetime(re_data['datetime'])
    re_data['year'] = re_data.datetime.dt.strftime("%Y")
    re_data['mo'] = re_data.datetime.dt.strftime("%Y-%m")
    re_data['day'] = re_data.datetime.dt.strftime("%Y-%m-%d")

    re_data['Pcs'] = re_data['Pcs'].shift(12).fillna(0)

    demand_data['datetime'] = pd.to_datetime(demand_data['datetime'])
    demand_data['day'] = demand_data.datetime.dt.strftime("%Y-%m-%d")

    if time_granularity == 'daily':
        re_data_agg = re_data.groupby(['day', 'mo', 'year'])[['Ppv', 'Pw', 'Pcs']].sum().reset_index()
        demand_data_agg = demand_data.groupby('day')['PF'].sum().reset_index()
    else:  # hourly
        re_data_agg = re_data
        demand_data_agg = demand_data

    data_m = re_data_agg.merge(demand_data_agg, on='day' if time_granularity == 'daily' else 'datetime', how='left')
    
    print('data_m', len(data_m))

    return data_m.dropna()

if __name__ == '__main__':
    app.run_server(debug=True, port='8080') #,host='0.0.0.0')
