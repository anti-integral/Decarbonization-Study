import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from utility import RenewableEnergyProblem, aggregate_solution_results, finalize
from pymoo.config import Config
from IPython import embed

Config.warnings['not_compiled'] = False

def run_optimization(data_test, constants, scenario, re_sources, re_min, re_max, num_gen, time_granularity):
    all_solutions = []

    wt_production = data_test.Pw.values
    pv_production = data_test.Ppv.values
    csp_production = data_test.Pcs.values
    demand = data_test.PF.values

    n_periods = len(demand)  # This will be hours or days depending on time_granularity

    # Adjust production values based on selected RE sources
    if re_sources == 'wt_pv':
        csp_production = np.zeros_like(csp_production)
    elif re_sources == 'wt_csp':
        pv_production = np.zeros_like(pv_production)
    elif re_sources == 'pv_csp':
        wt_production = np.zeros_like(wt_production)

    n_days = len(demand)

    print('data_test',data_test)
    print('n_periods',n_periods)

    problem = RenewableEnergyProblem(
        n_periods, 
        wt_production, pv_production, csp_production, demand, 
        constants['BATTERY_CAPACITY'], constants['Cost_day_pv'], constants['Cost_day_wind'], constants['Cost_day_csp'], 
        constants['LCOE_csp_kwt'], constants['LCOE_batt'], constants['LCOE_util'], 
        constants['CO2_pv'], constants['CO2_wind'], constants['CO2_csp'], 
        constants['CO2_batt'], constants['CO2_util'],
        scenario, re_min, re_max, re_sources, time_granularity
    )

    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=100,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=int),
        mutation=PM(prob=1.0, eta=20, vtype=int),
        eliminate_duplicates=True
    )

    res = minimize( problem,
                    algorithm,
                    ('n_gen', num_gen),
                    seed=1,
                    verbose=False)

    print('res.X',res.X)
    
    df = pd.DataFrame(np.unique(res.X, axis=0)+0, columns=['n_wt', 'n_pv', 'n_csp', 'n_batt'])
    df['Total_Energy_Cost'] = res.F[:, 0]
    df['Total_CO2'] = res.F[:, 1]
    print('len(df)',len(df))

    aggregated_results = df.apply(lambda x: aggregate_solution_results(problem, x), axis=1)

    # Add up to 50 random solutions
    random_solutions = aggregated_results.sample(n=min(50, len(aggregated_results)))
    all_solutions.extend([pd.DataFrame([sol]) for _, sol in random_solutions.iterrows()])
    print('len(all_solutions)',len(all_solutions))

    return all_solutions

def process_results(df_sol, scenario):
    return finalize(df_sol, scenario)

