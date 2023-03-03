import os
import datetime
import pandas as pd
import numpy as np
import sys
import re
import random
from idmtools.entities import IAnalyzer	
from idmtools.entities.simulation import Simulation

## For plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

class MonthlyPfPRAnalyzer(IAnalyzer):

    def __init__(self, exp_name, sweep_variables=None, working_dir='./', start_year=2020, end_year=2025,
                 burnin=None, filter_exists=False):

        super(MonthlyPfPRAnalyzer, self).__init__(working_dir=working_dir,
                                                   filenames=["output/MalariaSummaryReport_monthly.json"]
                                                   )
     
        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.exp_name = exp_name
        self.start_year = start_year
        self.end_year = end_year
        self.burnin = burnin
        self.filter_exists = filter_exists

    def filter(self, simulation: Simulation):
        if self.filter_exists:
            file = os.path.join(simulation.get_path(), self.filenames[0])
            return os.path.exists(file)
        else:
            return True

    def map(self, data, simulation: Simulation):
    
        adf = pd.DataFrame()
        fname = self.filenames[0]
        age_bins = data[self.filenames[0]]['Metadata']['Age Bins']
      
        for age in range(len(age_bins)):
            d = data[fname]['DataByTimeAndAgeBins']['PfPR by Age Bin'][:-1]
            pfpr = [x[age] for x in d]
          
            d = data[fname]['DataByTimeAndAgeBins']['Annual Clinical Incidence by Age Bin'][:-1]
            clinical_cases = [x[age] for x in d]
         
            d = data[fname]['DataByTimeAndAgeBins']['Annual Severe Incidence by Age Bin'][:-1]
            severe_cases = [x[age] for x in d]
          
            d = data[fname]['DataByTimeAndAgeBins']['Average Population by Age Bin'][:-1]
            pop = [x[age] for x in d]

            simdata = pd.DataFrame({'month': range(1, len(pfpr)+1),
                                    'PfPR': pfpr,
                                    'Cases': clinical_cases,
                                    'Severe cases': severe_cases,
                                    'Pop': pop})
                       
            simdata['agebin'] = age_bins[age]

            adf = pd.concat([adf, simdata])

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                try:
                    adf[sweep_var] = simulation.tags[sweep_var]
                except:
                    adf[sweep_var] = '-'.join([str(x) for x in simulation.tags[sweep_var]])

        return adf

    def reduce(self, all_data):

        selected = [data for sim, data in all_data.items()]
        print(len(selected))
        if len(selected) == 0:
            print("\nWarning: No data have been returned... Exiting...")
            return

        if not os.path.exists(os.path.join(self.working_dir, self.exp_name)):
            os.mkdir(os.path.join(self.working_dir, self.exp_name))

        print(f'\nSaving outputs to: {os.path.join(self.working_dir, self.exp_name)}')

        adf = pd.concat(selected).reset_index(drop=True)
        adf.to_csv((os.path.join(self.working_dir, self.exp_name, 'PfPR_ClinicalIncidence_monthly.csv')),
                   index=False)
        
if __name__ == "__main__":

    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    
    expts = {#'exp_name' : 'exp_id'
              'tmh6260_indie_pickup_CM' : '6df14b84-a94f-4f23-afa0-cda00f675f29'
    }
    
    jdir =  '/projects/b1139/indie_emodpy/experiments'
    wdir= '/projects/b1139/indie_emodpy/experiments/simulation_outputs'
    
    if not os.path.exists(wdir):
        os.mkdir(wdir)
        
    sweep_variables = ['Run_Number', 'xTLH', 'cm_cov_u5'] 


    with Platform('SLURM_LOCAL',job_directory=jdir) as platform:

        for expname, exp_id in expts.items():
          
            analyzer = [MonthlyPfPRAnalyzer(exp_name=expname,
                                      sweep_variables=sweep_variables,
                                      working_dir=wdir)]
            
            # Create AnalyzerManager with required parameters
            manager = AnalyzeManager(configuration={},ids=[(exp_id, ItemType.EXPERIMENT)],
                                     analyzers=analyzer, partial_analyze_ok=True)
            # Run analyze
            manager.analyze()