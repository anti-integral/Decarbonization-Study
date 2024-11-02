import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from IPython import embed

CO2_ton_per_gal = 103.5

class RenewableEnergyProblem(ElementwiseProblem):
    def __init__(self, n_periods, wt_production, pv_production, csp_production, demand, 
                 BATTERY_CAPACITY, Cost_day_pv, Cost_day_wind, Cost_day_csp, LCOE_csp_kwt, 
                 LCOE_batt, LCOE_util, 
                 CO2_pv, CO2_wind, CO2_csp, 
                 CO2_batt,CO2_util, scenario, re_min, re_max, 
                 re_sources, time_granularity):
        
        n_batt_max = 20
        if scenario != 'battery':
            n_batt_max = 0.001


        # Adjust upper bounds based on selected RE sources
        xu = np.array([50, 35, 60, n_batt_max])
        EPS = 0.01
        if re_sources == 'wt_pv':
            xu[2] = EPS  # Set CSP upper bound to EPS
        elif re_sources == 'wt_csp':
            xu[1] = EPS  # Set PV upper bound to EPS
        elif re_sources == 'pv_csp':
            xu[0] = EPS  # Set WT upper bound to EPS

        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([0, 0, 0, 0]),
                         xu=xu,
                         vtype=int)

        self.n_periods = n_periods
        self.time_granularity = time_granularity
        self.re_sources = re_sources                 
        self.wt_production = wt_production
        self.pv_production = pv_production
        self.csp_production = csp_production
        self.demand = demand
        self.BATTERY_CAPACITY = BATTERY_CAPACITY
        self.Cost_day_pv = Cost_day_pv
        self.Cost_day_wind = Cost_day_wind
        self.Cost_day_csp = Cost_day_csp
        self.LCOE_csp_kwt = LCOE_csp_kwt
        self.LCOE_batt = LCOE_batt
        self.LCOE_util = LCOE_util
        self.CO2_pv = CO2_pv
        self.CO2_wind = CO2_wind
        self.CO2_csp = CO2_csp
        self.CO2_batt = CO2_batt
        self.CO2_util = CO2_util
        self.scenario = scenario
        self.re_min = min(0.97, re_min+0.03)
        if re_max > 0.95:
            re_max = 1.05   
        self.re_max = re_max

    def _evaluate(self, x, out, *args, **kwargs):
        n_wt, n_pv, n_csp, n_batt = x

        if self.scenario != 'battery':
            n_batt = 0

        # Ensure unused RE sources are set to 0
        if self.re_sources == 'wt_pv':
            n_csp = 0
        elif self.re_sources == 'wt_csp':
            n_pv = 0
        elif self.re_sources == 'pv_csp':
            n_wt = 0

        daily_values = self.calculate_period_values(n_wt, n_pv, n_csp, n_batt)
        
        if self.scenario == 'battery':
            total_energy_cost = daily_values['RE_Cost'].sum() + daily_values['Batt_Cost'].sum()
            total_co2 = daily_values['RE_CO2'].sum() + daily_values['Batt_CO2'].sum()
        else:
            total_energy_cost = daily_values['RE_Cost'].sum() + daily_values['Util_Cost'].sum()
            total_co2 = daily_values['RE_CO2'].sum() + daily_values['Util_CO2'].sum()

        average_sl = daily_values['Service_Level'].mean()
        
        out["F"] = [total_energy_cost, total_co2]
        out["G"] = [self.re_min - average_sl, 
                    average_sl - self.re_max]

    def calculate_period_values(self, n_wt, n_pv, n_csp, n_batt):
        n_pv = n_pv * 1000
        period_values = []
        battery_charge = 0
        
        for period in range(self.n_periods):
            production = (
                n_wt * self.wt_production[period] +
                n_pv * self.pv_production[period] +
                n_csp * self.csp_production[period]
            )
            
            available_energy = production + battery_charge
            excess_energy = production - self.demand[period]
            
            sl_curr = min(available_energy / self.demand[period], 2)
            
            re_cost = (self.Cost_day_pv * n_pv + self.Cost_day_wind * n_wt +
                       (self.LCOE_csp_kwt * self.csp_production[period] + self.Cost_day_csp) * n_csp)
            
            re_co2 = (self.CO2_pv * self.pv_production[period] * n_pv +
                      self.CO2_wind * self.wt_production[period] * n_wt +
                      self.CO2_csp * self.csp_production[period] * n_csp)
            
            new_battery_charge = 0
            batt_cost = 0
            batt_co2 = 0
            util_cost = 0
            util_co2 = 0
            util_energy = 0
            if self.scenario == 'battery':
                batt_cost = self.LCOE_batt * battery_charge 
                batt_co2 = self.CO2_batt * battery_charge
                new_battery_charge = max(0, min(battery_charge + excess_energy, n_batt * self.BATTERY_CAPACITY))
            else: # scenario does not involve a battery
                missing_energy = -excess_energy
                if missing_energy>0:
                    util_energy = missing_energy               
                util_cost = self.LCOE_util * util_energy 
                util_co2 = self.CO2_util * util_energy 

            period_values.append({
                'Period': period + 1,
                'RE_Cost': re_cost,
                'Batt_Cost': batt_cost,
                'Util_Cost': util_cost,
                'RE_CO2': re_co2,
                'Batt_CO2': batt_co2,
                'Util_CO2': util_co2,
                'Service_Level': sl_curr,
                'Batt_Charge': new_battery_charge,
                'Util_Energy': util_energy,
                'Demand': self.demand[period],
                'P_wt': self.wt_production[period],
                'P_pv': self.pv_production[period],
                'P_csp': self.csp_production[period],
                'RE_Production': production,
                'Tot_Energy': production+new_battery_charge
            })
            
            battery_charge = new_battery_charge

        return pd.DataFrame(period_values)


def aggregate_solution_results(problem, solution):
    daily_df = problem.calculate_period_values(
        n_wt=solution['n_wt'],
        n_pv=solution['n_pv'],
        n_csp=solution['n_csp'],
        n_batt=solution['n_batt']
    )
    
    return pd.Series({
        'n_wt': solution['n_wt'],
        'n_pv': solution['n_pv']*1000,
        'n_csp': solution['n_csp'],
        'n_batt': solution['n_batt'],
        'RE_Cost': daily_df['RE_Cost'].sum(),
        'Batt_Cost': daily_df['Batt_Cost'].sum(),
        'Util_Cost': daily_df['Util_Cost'].sum(),
        'RE_CO2': daily_df['RE_CO2'].sum(),
        'Batt_CO2': daily_df['Batt_CO2'].sum(),
        'Util_CO2': daily_df['Util_CO2'].sum(),
        'Demand': daily_df['Demand'].sum(),
        'P_wt': daily_df['P_wt'].sum(),
        'P_pv': daily_df['P_pv'].sum(),
        'P_csp': daily_df['P_csp'].sum(),
        'RE_Production': daily_df['RE_Production'].sum(),
        'Batt_Charge': daily_df['Batt_Charge'].sum(),
        'Util_Energy': daily_df['Util_Energy'].sum(),
        'SL_mean': daily_df['Service_Level'].mean(),
        'SL_std': daily_df['Service_Level'].std(),
        'Total_Energy_Cost': daily_df['RE_Cost'].sum() + daily_df['Batt_Cost'].sum(),
        'Total_CO2': (daily_df['RE_CO2'].sum() + daily_df['Batt_CO2'].sum()) / CO2_ton_per_gal
    })


def load_preset_data():
    # Generate hourly timestamps for a full year
    hourly_timestamps = pd.date_range(start='2023-01-01', end='2023-12-31 23:00:00', freq='h')
    
    # Load RE Production Data (hourly)
    preset_re_data = pd.DataFrame({
        'datetime': hourly_timestamps,
        'Ppv': np.random.rand(len(hourly_timestamps)) * 100,  # Solar production
        'Pw': np.random.rand(len(hourly_timestamps)) * 80,    # Wind production
        'Pcs': np.random.rand(len(hourly_timestamps)) * 60    # Concentrated solar production
    })

    # Adjust for day/night cycle (simple approximation)
    preset_re_data['hour'] = preset_re_data['datetime'].dt.hour
    preset_re_data.loc[(preset_re_data['hour'] < 6) | (preset_re_data['hour'] > 18), 'Ppv'] = 0
    preset_re_data = preset_re_data.drop('hour', axis=1)

    # Load Plant Demand Data (hourly)
    preset_demand_data = pd.DataFrame({
        'datetime': hourly_timestamps,
        'PF': np.random.randint(1000, 2000, len(hourly_timestamps))
    })

    # Adjust for typical daily demand patterns
    preset_demand_data['hour'] = preset_demand_data['datetime'].dt.hour
    preset_demand_data['PF'] = preset_demand_data['PF'] * (1 + 0.3 * np.sin(preset_demand_data['hour'] * np.pi / 12))
    preset_demand_data = preset_demand_data.drop('hour', axis=1)

    # Load Constants Data (unchanged)
    preset_constants = pd.DataFrame({
        'Constant': ['LCOE_pv', 'LCOE_wind', 'LCOE_csp', 'LCOE_batt', 'LCOE_util',
                     'CO2_pv', 'CO2_wind', 'CO2_csp', 'CO2_batt', 'CO2_util',
                     'BATTERY_CAPACITY', 'Ndays_per_month', 'Cost_day_pv', 'Cost_day_wind',
                     'Cost_day_csp', 'LCOE_csp_kwt', 'CO2_ton_per_gal'],
        'Value': np.random.rand(17) * 100  # Example values
    })

    return preset_re_data, preset_demand_data, preset_constants


main_vars = ['Demand', 'Util_Energy', 'Batt_Charge', 'Total_RE_Prod', #'P_pv', 'P_wt', 'P_csp',
             'RE_Cost', 'Util_Cost', 'Batt_Cost','RE_CO2', 'Util_CO2','Batt_CO2',
             'Total_Energy_Cost','Total_CO2']

def finalize(df_res, scenario):

    #embed()

    varE = 'Util_Energy'
    varCost = 'Util_Cost'        
    varCO2 = 'Util_CO2'
    if scenario == 'battery':
        varE = 'Batt_Charge'
        varCost = 'Batt_Cost'
        varCO2 = 'Batt_CO2'

    df_res['P_pv'] = df_res['n_pv'] * df_res['P_pv']
    df_res['P_wt'] = df_res['n_wt'] * df_res['P_wt']
    df_res['P_csp'] = df_res['n_csp'] * df_res['P_csp']

    #df_res['Total_RE_Prod'] = df_res['n_pv'] * df_res['P_pv'] + df_res['n_wt'] * df_res['P_wt'] + df_res['n_csp'] * df_res['P_csp']
    df_res['Total_RE_Prod'] = df_res['P_pv'] + df_res['P_wt'] + df_res['P_csp']

    v_excl = 'Batt'
    if scenario == 'battery':
        v_excl = 'Util'
    vars = [x for x in main_vars if v_excl not in x]
    cols = ['n_wt', 'n_csp', 'n_pv','n_batt'] + vars 

    df_res = df_res[cols].round(0)

    df_res['RE/Demand'] = round(df_res.Total_RE_Prod / df_res.Demand,2)
    df_res['RE+BU/Demand'] = round((df_res.Total_RE_Prod + df_res[varE]) / df_res.Demand, 2)

    return df_res

