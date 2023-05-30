import struct
import numpy as np

class SpatialOutput:

    def __init__(self):
        self.n_nodes = 0
        self.n_tstep = 0
        self.nodeids = []
        self.data = None
        self.start = 0
        self.interval = 1

    @classmethod
    def from_bytes(cls, bytes, filtered=False):
        # The header size changes if the file is a filtered one
        headersize = 16 if filtered else 8

        # Create the class
        so = cls()

        # Retrive the number of nodes and number of timesteps
        so.n_nodes, = struct.unpack('i', bytes[0:4])
        so.n_tstep, = struct.unpack('i', bytes[4:8])

        # If filtered, retrieve the start and interval
        if filtered:
            start, = struct.unpack('f', bytes[8:12])
            interval, = struct.unpack('f', bytes[12:16])
            so.start = int(start)
            so.interval = int(interval)

        # Get the nodeids
        so.nodeids = struct.unpack(str(so.n_nodes) + 'I', bytes[headersize:headersize + so.n_nodes * 4])
        so.nodeids = np.asarray(so.nodeids)

        # Retrieve the data
        so.data = struct.unpack(str(so.n_nodes * so.n_tstep) + 'f',
                                bytes[headersize + so.n_nodes * 4:headersize + so.n_nodes * 4 + so.n_nodes * so.n_tstep * 4])
        so.data = np.asarray(so.data)
        so.data = so.data.reshape(so.n_tstep, so.n_nodes)

        return so

    def to_dict(self):
        return {'n_nodes': self.n_nodes,
                'n_tstep': self.n_tstep,
                'nodeids': self.nodeids,
                'start': self.start,
                'interval': self.interval,
                'data': self.data}

from typing import Dict
import pandas as pd

def construct_spatial_output_df(rawdata: Dict, channel: str, timesteps=None) -> pd.DataFrame:
    """
    Construct spatial output data frame from a Spatial Output dictionary
    Args:
        rawdata: Spatial output file
        channel: Channel name
        timesteps: Timesteps. Defaults to empty array if not provided
    Returns:
    """
    if timesteps is None:
        timesteps = []
    n_nodes = rawdata['n_nodes']
    n_tstep = rawdata['n_tstep']
    if 'start' in rawdata:
        start = rawdata['start']
        interval = rawdata['interval']
    else:
        start, interval = 0, 1
    nodeids = rawdata['nodeids']
    data = rawdata['data']

    all_timesteps = range(start, (start + n_tstep) * interval, interval)

    df = pd.DataFrame({channel: [item for sublist in data for item in sublist],
                       'time': [item for sublist in [[x] * n_nodes for x in all_timesteps] for item in sublist],
                       'node': [item for sublist in [list(nodeids) * len(all_timesteps)] for item in sublist]})
    if not timesteps:
        return df

    timesteps = sorted(list(set(all_timesteps).intersection(timesteps)))
    return df[df['time'].isin(timesteps)]


import os
import pandas as pd

from idmtools.entities import IAnalyzer
from idmtools.entities.simulation import Simulation

class SpatialAnalyzer(IAnalyzer):
    # This analyzer can handled both unfiltered and filtered SpatialReports
    def __init__(self, dir_name, exp_id, spatial_channels, sweep_variables, working_dir='.', snapshot=None):
        super(SpatialAnalyzer, self).__init__(working_dir=working_dir,
                                                 filenames=[f'output/SpatialReportMalariaFiltered_all_ages_{x}.bin' for x in spatial_channels])

        self.dir_name = dir_name
        self.exp_id = exp_id
        # Once we fix idmtools, we should remove this
        self.parse = False
        self.sweep_variables = sweep_variables or ['Run_Number', 'x_Temporary_Larval_Habitat']
        self.spatial_channels = spatial_channels
        self.output_fname = os.path.join(self.working_dir, f"SpatialReportMalariaFiltered.csv")
        self.snapshot = snapshot
        

    def map(self, data, simulation: Simulation):
        # we have to parse our data first since it will be a raw set of binary data
        # Once we have this fixed within idmtools/emodpy, we will remove this bit of code
        for ch in self.spatial_channels:
            fn = f'output/SpatialReportMalariaFiltered_all_ages_{ch}.bin'
            data[fn] = SpatialOutput.from_bytes(data[fn], 'Filtered' in fn).to_dict()
        simdata = construct_spatial_output_df(data[f'output/SpatialReportMalariaFiltered_all_ages_{self.spatial_channels[0]}.bin'], self.spatial_channels[0])
        if len(self.spatial_channels) > 1:
            for ch in self.spatial_channels[1:]:
                simdata = pd.merge(left=simdata,
                                   right=construct_spatial_output_df(data[f'output/SpatialReportMalariaFiltered_all_ages_{ch}.bin'], ch),
                                   on=['time', 'node'])

        # simdata['time'] = simdata['time'] - 365
        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                simdata[sweep_var] = simulation.tags[sweep_var]
            else:
                simdata[sweep_var] = 0
        return simdata

    def reduce(self, all_data):
        data_sets_per_experiment = {}

        for simulation, associated_data in all_data.items():
            experiment_name = simulation.experiment.name
            if experiment_name not in data_sets_per_experiment:
                data_sets_per_experiment[experiment_name] = []

            data_sets_per_experiment[experiment_name].append(associated_data)

        for experiment_name, data_sets in data_sets_per_experiment.items():
            d = pd.concat(data_sets).reset_index(drop=True)
            d['experiment'] = self.exp_id
            # save full dataframe
            d.to_csv(self.output_fname, index=False)
            print("Reporting on", self.spatial_channels)
            print("Grouped by", self.sweep_variables)
            print("Full spatial report saved to", self.output_fname)
            # save snapshots
            if self.snapshot is not None:
                d_sub = d[d['time'] == self.snapshot[0]] 
                sub_fname = os.path.join(self.working_dir,f"SpatialReportMalariaFiltered_Snapshot.csv")
                if(len(self.snapshot)>1):
                    for snap in self.snapshot[1:]:
                        d_sub_add = d[d['time'] == snap]
                        d_sub = pd.concat([d_sub,d_sub_add])
                d_sub.to_csv(sub_fname)
                print("Snapshot from days",self.snapshot, "saved to",sub_fname) 
            


        
if __name__ == "__main__":

    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform
    
    
    expts = {#'indie_vector_test_burnin' : '79d809ab-a042-48f3-878a-6dde4f24000c',
             #'checkpoint_test' : '3e48d65c-391d-4916-89d4-a8bc774a78e1',
             #'habitat_test': '30665704-96a9-4245-b450-7d7994a73874',
             #'rainfall_shift': '67187c60-f3e5-4fa9-92a7-11afdc357330',
             #'30degree': 'cc51fc74-e79b-4c92-be34-f9f741fb4212',
             #'30degree_3x': '3bc9a766-555e-4319-9681-824911b17cad',
             #'30degree_10x':'f5ddee15-217d-4c9d-8920-196f3c890baa',
             #'30degree_50x': 'e8ac22a6-d7fb-49a0-9b3a-de69edb3cfe1',
             #'30degree_VarX': 'e4dec631-7f16-4b43-9cab-f0b4b6276cb7',
             '30degree_VarX_test': '8595ff17-40f2-46d2-b2f0-0208e4fe8b23'}
    
    jdir =  '/projects/b1139/indie_emodpy/experiments'
    wdir= '/projects/b1139/indie_emodpy/simulation_output'
    
    sweep_variables = ['Run_Number', 'x_Temporary_Larval_Habitat','HR'] # for burnins
    #sweep_variables = ['Run_Number', 'xTLH', 'cm_cov_u5','HR'] # for pickups
    spatial_channels = ['Population', 'PCR_Parasite_Prevalence','Daily_Bites_Per_Human','New_Clinical_Cases', 'Rainfall', 'Air_Temperature']
    dates = [213, 365, 516, 745]
    dates = np.array(dates)
    dates = 8*365 + dates
    if not os.path.exists(wdir):
                os.mkdir(wdir)
    with Platform('SLURM_LOCAL',job_directory=jdir) as platform:
        for expname, exp_id in expts.items():    
            if not os.path.exists(os.path.join(wdir,exp_id)):
                os.mkdir(os.path.join(wdir,exp_id))
                
            analyzer = [SpatialAnalyzer(dir_name=expname,
                                        exp_id = exp_id,
                                        spatial_channels=spatial_channels,
                                        sweep_variables=sweep_variables,
                                        working_dir=os.path.join(wdir,exp_id))]
            
            # Create AnalyzerManager with required parameters
            manager = AnalyzeManager(configuration={},ids=[(exp_id, ItemType.EXPERIMENT)],
                                     analyzers=analyzer, partial_analyze_ok=True)
            # Run analyze
            manager.analyze()
