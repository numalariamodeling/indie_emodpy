import os
import datetime
import pandas as pd
import numpy as np
import sys
import itertools
import re
import random
from idmtools.entities import IAnalyzer	
from idmtools.entities.simulation import Simulation

def construct_spatial_output_df(rawdata, channel, timesteps=[]) :
    n_nodes = rawdata['n_nodes']
    n_tstep = rawdata['n_tstep']
    if 'start' in rawdata :
        start = rawdata['start']
        interval = rawdata['interval']
    else :
        start, interval = 0,1
    nodeids = rawdata['nodeids']
    data = rawdata['data']
    all_timesteps = range(start, (start+n_tstep)*interval, interval)
    df = pd.DataFrame( { channel : [item for sublist in data for item in sublist],'time' : [item for sublist in [[x]*n_nodes for x in all_timesteps] for item in sublist],'node' : [item for sublist in [list(nodeids)*len(all_timesteps)] for item in sublist]} )
    if not timesteps :
        return df
    timesteps = sorted(list(set(all_timesteps).intersection(timesteps)))
    return df[df['time'].isin(timesteps)]

class SpatialAnalyzer(IAnalyzer):
    def __init__(self, expt_name, spatial_channels, age_groups, sweep_variables=None, working_dir='.'):
        super(SpatialAnalyzer, self).__init__(working_dir=working_dir,filenames=['output/SpatialReportMalariaFiltered__%s_%s.bin' % (x,y) for (x,y) in list(itertools.product(age_groups,spatial_channels))])
        self.expt_name = expt_name
        self.sweep_variables = sweep_variables or ['Run_Number','Baseline_Fitting_Rank', 'MTTT', 'Perfect_MTTT', 'Special_Migration', 'Rainfall_Ratio']
        self.spatial_channels = spatial_channels
        self.age_groups = age_groups or ['']
        self.wdir = working_dir
    def map(self, data, simulation):
        simdata = construct_spatial_output_df(data['output/SpatialReportMalariaFiltered__%s_%s.bin' % (self.age_groups[0],self.spatial_channels[0])],self.spatial_channels[0])
        if len(self.age_groups) > 1:
            for ag in self.age_groups[1:]:
                if len(self.spatial_channels) > 1:
                    for ch in self.spatial_channels[1:]:
                        simdata = pd.merge(left=simdata,right=construct_spatial_output_df(data['output/SpatialReportMalariaFiltered__%s_%s.bin' % (ag,ch)], ch),on=['time', 'node'])
                for sweep_var in self.sweep_variables:
                    if sweep_var in simulation.tags.keys():
                        simdata[sweep_var] = simulation.tags[sweep_var]
                    else:
                        simdata[sweep_var] = 0
        return simdata
    
    def reduce(self, all_data):
        selected = [data for sim, data in all_data.items()]
        if len(selected) == 0:
            print("No data have been returned... Exiting...")
            return
        df = pd.concat(selected).reset_index(drop=True)
        df.to_csv(os.path.join(self.wdir, '%s.csv' % self.expt_name), index=False)
   
if __name__ == "__main__":
    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform
    expts = {#'exp_name' : 'exp_id'
              'test_checkpoint' : '0a829f17-ad8b-4824-833c-646533867526'}
    jdir =  '/projects/b1139/indie_emodpy/experiments'
    wdir=os.path.join(jdir, 'simulation_output')
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    # set desired SpatialMalariaReportFiltered channels to analyze and plot
    channels_inset_chart = ['Population','New_Clinical_Cases','Daily_Bites_Per_Human', 'PCR_Parasite_Prevalence']
    # Match age groups used to filter reports
    age_groups = ['under5','5to15','over15']
    # Sweep over tags
    sweep_variables = ['Run_Number', 'xTLH', 'cm_cov_u5'] 
    with Platform('SLURM_LOCAL',job_directory=jdir) as platform:
        for expname, exp_id in expts.items():
            analyzer = [SpatialAnalyzer(expt_name=expname,spatial_channels=channels_inset_chart,age_groups=age_groups,sweep_variables=sweep_variables,working_dir=wdir)]
            # Create AnalyzerManager with required parameters
            manager = AnalyzeManager(configuration={},ids=[(exp_id, ItemType.EXPERIMENT)],analyzers=analyzer, partial_analyze_ok=True)
            # Run analyze
            manager.analyze()