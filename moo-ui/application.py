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
from main_opt import run_optimization, process_results
from utility import load_preset_data
from IPython import embed

app = dash.Dash(__name__)

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

cc_style = {
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center',
}

small_button_style = {
    'width': '70%',
    'height': '75px',
    'background-color': 'lightblue',
    'font-size': '10px',
    'padding': '15px 10px',
    'margin': '0px',
    'border': 'none',
    'border-radius': '3px',
    'cursor': 'pointer'
}


all_solutions = []
processed_solutions = pd.DataFrame()

# Global variables to store preset data
preset_re_data, preset_demand_data, preset_constants = load_preset_data()


app.layout = dbc.Container([
    html.H1("Multi-Objective Optimization for Renewable Energy"),
    
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
                                            value='utility')
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
                                style={'background-color': 'darkorange', 
                                    'font-size': '12px', 
                                    'padding': '10px 20px',  # Add some padding for better appearance
                                    'border': 'none',  # Remove default border
                                    'cursor': 'pointer',  # Change cursor on hover
                                    'border-radius': '5px',  # Rounded corners                                 
                                    }, 
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
                style={'background-color': 'darkorange', 'font-size': '12px'},
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
                style={'background-color': 'darkorange', 'font-size': '12px'}, 
                n_clicks=0),

    dcc.Download(id='download-results'),

    html.Br(),
    html.Br(),
    html.Br(),
    html.Br()

], fluid=True)



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
    app.run_server(debug=True, port='8080',host='0.0.0.0')
