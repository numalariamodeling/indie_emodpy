import os
import pandas as pd
import numpy as np
import sys
import re
import random
from idmtools.entities import IAnalyzer	
from idmtools.entities.simulation import Simulation
# Example AddAnalyzer for EMOD Experiment

# In this example, we will demonstrate how to create an AddAnalyzer to analyze an experiment's output file

# First, import some necessary system and idmtools packages.
from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.analysis.add_analyzer import AddAnalyzer
from idmtools.core import ItemType
from idmtools.core.platform_factory import Platform

if __name__ == '__main__':

    jdir =  '/projects/b1139/indie_emodpy/experiments'
    wdir=os.path.join(jdir, 'simulation_output')
    
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    
    sweep_variables = ['Run_Number', 'xTLH', 'cm_cov_u5'] 
    spatial_channels = ['Population']
    filenames = ['output/SpatialReport_Population.bin']
    exp_id = 'db957fdd-aa9b-4118-812b-2d3d9b32024b'
    with Platform('SLURM_LOCAL',job_directory=jdir) as platform:
        analyzer = [AddAnalyzer(filenames=filenames)]
        
        # Create AnalyzerManager with required parameters
        manager = AnalyzeManager(configuration={},ids=[(exp_id, ItemType.EXPERIMENT)],
                                 analyzers=analyzer, partial_analyze_ok=True)
        # Run analyze
        manager.analyze()
