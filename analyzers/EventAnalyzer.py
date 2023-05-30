import os
import re
import pandas as pd
import numpy as np

from idmtools.analysis.csv_analyzer import CSVAnalyzer, IAnalyzer
from idmtools.entities.simulation import Simulation


class EventCounterAnalyzer(IAnalyzer):
    def __init__(self, exp_name, exp_id, sweep_variables, nodes, events, working_dir=".", start_day=0):
        super(EventCounterAnalyzer, self).__init__(working_dir=working_dir, filenames=[f"output/ReportEventCounter_node_{n}.json" for n in nodes])
        self.exp_name = exp_name
        self.exp_id = exp_id
        self.nodes = nodes 
        self.sweep_variables = sweep_variables
        self.events = events 
        self.output_fname = os.path.join(self.working_dir,exp_id,'CountedEvents.csv')
        self.start_day = start_day
        
         # make sure output folder exists
        if not os.path.exists(os.path.join(self.working_dir,exp_id)):
            os.makedirs(os.path.join(self.working_dir,exp_id))
    
    def map(self, data, simulation: Simulation):
        node_id = re.findall(r'\d+', self.filenames[0])[0]
        simdata = pd.DataFrame({x: data[self.filenames[0]]['Channels'][x]['Data'] for x in self.events})
        simdata['Node'] = node_id
        simdata['Time'] = simdata.index + self.start_day
        for sweep_var in self.sweep_variables:
          if sweep_var in simulation.tags.keys():
            simdata[sweep_var] = simulation.tags[sweep_var]
          else:
            simdata[sweep_var] = 0
            
        data_files = [simdata]
                
        if len(self.filenames) > 1:
          for add_file in self.filenames[1:]:
            node_id = re.findall(r'\d+', add_file)[0]
            add_data = pd.DataFrame({x: data[add_file]['Channels'][x]['Data'] for x in self.events})
            add_data['Time'] = add_data.index + self.start_day
            add_data['Node'] = node_id
            
            for sweep_var in self.sweep_variables:
              if sweep_var in simulation.tags.keys():
                add_data[sweep_var] = simulation.tags[sweep_var]
              else:
                add_data[sweep_var] = 0
            
            data_files.append(add_data)
        
        all_data = pd.concat(data_files).reset_index(drop=True)
         
        return all_data
    
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
            print("Reporting on", self.events)
            print("Grouped by", self.sweep_variables)
            print("Full event counter report saved to", self.output_fname)
            




class EventRecorderAnalyzer(CSVAnalyzer):
# This analyzer can handle EventRecorder Reports
    def __init__(self, exp_name, exp_id, events, sweep_variables, working_dir='.'):
        super(EventRecorderAnalyzer, self).__init__(working_dir=working_dir,
                                                 filenames=['output/ReportEventRecorder.csv'])

        self.exp_name = exp_name
        self.exp_id = exp_id
        # Once we fix idmtools, we should remove this
        self.parse = False
        self.sweep_variables = sweep_variables or ['Run_Number', 'x_Temporary_Larval_Habitat', 'cm_cov_u5']
        self.events = events
        self.output_fname = os.path.join(self.working_dir, f"RecordedEvents.csv")

        # make sure output folder exists
        os.makedirs(os.path.join(self.working_dir, 'events'), exist_ok=True)

    def map(self, data, simulation: Simulation):
        # we have to parse our data first since it will be a raw set of binary data
        # Once we have this fixed within idmtools/emodpy, we will remove this bit of code
        
        simdata = pd.read_csv(self.filenames[0])
        counts= simdata.groupby(['Node_ID','Time','Event_Name'],as_index=False)['Individual_ID'].count().rename(columns={"Individual_ID":"Event_Count"})
        
        ## TO ADD: reshape so that each event is a column, and each output dataset has n_timesteps rows

        for sweep_var in self.sweep_variables:
            if sweep_var in simulation.tags.keys():
                counts[sweep_var] = simulation.tags[sweep_var]
            else:
                counts[sweep_var] = 0
        return counts

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
            print("Reporting on", self.events)
            print("Grouped by", self.sweep_variables)
            print("Full event recorder report saved to", self.output_fname)
            
     
     

if __name__ == "__main__":

    from idmtools.analysis.analyze_manager import AnalyzeManager
    from idmtools.core import ItemType
    from idmtools.core.platform_factory import Platform
    
    expts = {#'checkpoint_test': '3e48d65c-391d-4916-89d4-a8bc774a78e1',
             #'habitat_test': '30665704-96a9-4245-b450-7d7994a73874',
             #'rainfall_shift': '67187c60-f3e5-4fa9-92a7-11afdc357330',
             #'30degree_3x': '3bc9a766-555e-4319-9681-824911b17cad',
             #'30degree_10x': 'f5ddee15-217d-4c9d-8920-196f3c890baa',
             '30degree_50x': 'e8ac22a6-d7fb-49a0-9b3a-de69edb3cfe1'}
             
    clusters = ["1","2","3","4","5","6"]
    jdir =  '/projects/b1139/indie_emodpy/experiments'
    wdir= '/projects/b1139/indie_emodpy/simulation_output'
    sweep_variables = ['Run_Number','xTLH','cm_cov_u5']
    events = ['Received_ITN', 'Received_Treatment', 'Received_SMC']
    if not os.path.exists(wdir):
                os.mkdir(wdir)
    with Platform('SLURM_LOCAL',job_directory=jdir) as platform:
        for expname, exp_id in expts.items():  
            if not os.path.exists(os.path.join(wdir,exp_id)):
                os.mkdir(os.path.join(wdir,exp_id))
            analyzer = [EventCounterAnalyzer(exp_name = expname, 
                                             exp_id = exp_id, 
                                             sweep_variables = sweep_variables, 
                                             nodes = clusters, 
                                             events = events,
                                             working_dir = wdir,
                                             start_day = (10-3)*365)]
            
            # Create AnalyzerManager with required parameters
            manager = AnalyzeManager(configuration={},ids=[(exp_id, ItemType.EXPERIMENT)],
                                     analyzers=analyzer, partial_analyze_ok=True)
            # Run analyze
            manager.analyze()
   