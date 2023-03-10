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


class InsetChartAnalyzer(IAnalyzer):

    @classmethod
    def monthparser(self, x):
        if x == 0:
            return 12
        else:
            return datetime.datetime.strptime(str(x), '%j').month

    def __init__(self, expt_name, sweep_variables=None, channels=None, working_dir=".", start_year=1991):
        super(InsetChartAnalyzer, self).__init__(working_dir=working_dir, filenames=["output/InsetChart.json"])
        self.sweep_variables = sweep_variables or ["Run_Number"]
        self.inset_channels = channels or ['Statistical Population', 'New Clinical Cases', 'Blood Smear Parasite Prevalence',
                                           'Infectious Vectors']
        self.expt_name = expt_name
        self.start_year = start_year

    def map(self, data, simulation: Simulation):
        simdata = pd.DataFrame({x: data[self.filenames[0]]['Channels'][x]['Data'] for x in self.inset_channels})
        simdata['Time'] = simdata.index
        simdata['Day'] = simdata['Time'] % 365
        simdata['Year'] = simdata['Time'].apply(lambda x: int(x / 365) + self.start_year)
        simdata['date'] = simdata.apply(
            lambda x: datetime.date(int(x['Year']), 1, 1) + datetime.timedelta(int(x['Day']) - 1), axis=1)

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                simdata[sweep_var] = simulation.tags[sweep_var]
            elif sweep_var == 'Run_Number' :
                simdata[sweep_var] = 0
        return simdata

    def reduce(self, all_data):

        selected = [data for sim, data in all_data.items()]
        if len(selected) == 0:
            print("No data have been returned... Exiting...")
            return

        if not os.path.exists(os.path.join(self.working_dir, self.expt_name)):
            os.mkdir(os.path.join(self.working_dir, self.expt_name))

        adf = pd.concat(selected).reset_index(drop=True)
        adf.to_csv(os.path.join(self.working_dir, self.expt_name, 'All_Age_InsetChart.csv'), index=False)


if __name__ == "__main__":

    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform

    
    expts = {#'exp_name' : 'exp_id'
              'tmh6260_indie_pickup_CM' : '6df14b84-a94f-4f23-afa0-cda00f675f29'
    }
    

    jdir =  '/projects/b1139/indie_emodpy/experiments'
    wdir=os.path.join(jdir, 'simulation_outputs')
    
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    
    sweep_variables = ['Run_Number', 'xTLH', 'cm_cov_u5'] 

    # set desired InsetChart channels to analyze and plot
    channels_inset_chart = ['Statistical Population', 'True Prevalence', 'New Clinical Cases','Infectious Vectors','Rainfall','Air Temperature']

    
    with Platform('SLURM_LOCAL',job_directory=jdir) as platform:

        for expname, exp_id in expts.items():
          
            analyzer = [InsetChartAnalyzer(expt_name=expname,
                                      channels=channels_inset_chart,
                                      sweep_variables=sweep_variables,
                                      working_dir=wdir)]
            
            # Create AnalyzerManager with required parameters
            manager = AnalyzeManager(configuration={},ids=[(exp_id, ItemType.EXPERIMENT)],
                                     analyzers=analyzer, partial_analyze_ok=True)
            # Run analyze
            manager.analyze()
            
            
 
    # read in analyzed InsetChart data
    sweep_variables = ['Run_Number', 'xTLH', 'cm_cov_u5'] 
    expt_name=list(expts.keys())[0]
    years_to_keep = 2
    
    df = pd.read_csv(os.path.join(wdir, expt_name, 'All_Age_InsetChart.csv'))
    end = np.max(df['Time'])
    df = df[df['Time']>=end-2*365]
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby(['date'] + sweep_variables)[channels_inset_chart].agg(np.mean).reset_index()

    # make InsetChart plot
    fig1 = plt.figure('InsetChart', figsize=(12, 6))
    fig1.subplots_adjust(hspace=0.5, left=0.08, right=0.97)
    fig1.suptitle(f'Analyzer: InsetChartAnalyzer')
    axes = [fig1.add_subplot(2, 3, x + 1) for x in range(6)]
    for ch, channel in enumerate(channels_inset_chart):
        ax = axes[ch]
        for p, pdf in df.groupby(sweep_variables):
            ax.plot(pdf['date'], pdf[channel], '-', linewidth=0.8, label=p)
        ax.set_title(channel)
        ax.set_ylabel(channel)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if len(sweep_variables) > 0:
        axes[-1].legend(title=sweep_variables)
    fig1.savefig(os.path.join(wdir, expt_name, 'InsetChart.png'))        

