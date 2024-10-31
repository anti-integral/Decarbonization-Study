import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from IPython import embed

CO2_ton_per_gal = 103.5

class RenewableEnergyProblem(ElementwiseProblem):
    def __init__(self, n_days, wt_production, pv_production, csp_production, demand, 
                 BATTERY_CAPACITY, Cost_day_pv, Cost_day_wind, Cost_day_csp, LCOE_csp_kwt, 
                 LCOE_batt, LCOE_util, 
                 CO2_pv, CO2_wind, CO2_csp, 
                 CO2_batt,CO2_util, scenario, re_min, re_max):
        
        n_batt_max = 20
        if scenario != 'battery':
            n_batt_max = 0.001

        print('===> n_batt_max',n_batt_max)

        super().__init__(n_var=4,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([0, 0, 0, 0]),
                         xu=np.array([50, 35, 60, n_batt_max]),
                         vtype=int)
        self.n_days = n_days
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

        daily_values = self.calculate_daily_values(n_wt, n_pv, n_csp, n_batt)
        
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

    def calculate_daily_values(self, n_wt, n_pv, n_csp, n_batt):
        n_pv = n_pv * 1000
        daily_values = []
        battery_charge = 0
        
        for day in range(self.n_days):
            production = (
                n_wt * self.wt_production[day] +
                n_pv * self.pv_production[day] +
                n_csp * self.csp_production[day]
            )
            
            available_energy = production + battery_charge
            excess_energy = production - self.demand[day]
            
            sl_curr = min(available_energy / self.demand[day], 2)
            
            re_cost = (self.Cost_day_pv * n_pv + self.Cost_day_wind * n_wt +
                       (self.LCOE_csp_kwt * self.csp_production[day] + self.Cost_day_csp) * n_csp)
            
            re_co2 = (self.CO2_pv * self.pv_production[day] * n_pv +
                      self.CO2_wind * self.wt_production[day] * n_wt +
                      self.CO2_csp * self.csp_production[day] * n_csp)
            
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

            daily_values.append({
                'Day': day + 1,
                'RE_Cost': re_cost,
                'Batt_Cost': batt_cost,
                'Util_Cost': util_cost,
                'RE_CO2': re_co2,
                'Batt_CO2': batt_co2,
                'Util_CO2': util_co2,
                'Service_Level': sl_curr,
                'Batt_Charge': new_battery_charge,
                'Util_Energy': util_energy,
                'Demand': self.demand[day],
                'P_wt': self.wt_production[day],
                'P_pv': self.pv_production[day],
                'P_csp': self.csp_production[day],
                'RE_Production': production,
                'Tot_Energy': production+new_battery_charge
            })
            
            battery_charge = new_battery_charge

        return pd.DataFrame(daily_values)

def aggregate_solution_results(problem, solution):
    daily_df = problem.calculate_daily_values(
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
    cols = ['n_wt', 'n_csp', 'n_pv'] + vars 

    df_res = df_res[cols].round(0)

    df_res['RE/Demand'] = round(df_res.Total_RE_Prod / df_res.Demand,2)
    df_res['RE+BU/Demand'] = round((df_res.Total_RE_Prod + df_res[varE]) / df_res.Demand, 2)

    return df_res

