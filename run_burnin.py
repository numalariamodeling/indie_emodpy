###################
# Import Packages #
###################

# generic packages
import pathlib
import os
import numpy as np
from functools import \
    partial 
import sys
 
# from idmtools   
from idmtools.assets import Asset, AssetCollection  
from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment
from idmtools.builders import SimulationBuilder

# from emodpy
from emodpy.emod_task import EMODTask
from emodpy.utils import EradicationBambooBuilds
from emodpy.bamboo import get_model_files

# from emodpy_api
import emod_api.config.default_from_schema_no_validation as dfs
import emod_api.campaign as camp
import emod_api.demographics.PreDefinedDistributions as Distributions

# from emodpy-malaria
from emodpy_malaria.reporters.builtin import *
import emodpy_malaria.demographics.MalariaDemographics as Demographics

# from manifest
import manifest

# from utils_slurm
sys.path.append('../')
from utils_slurm import build_burnin_df

#######################
# Sim. Specifications #
#######################

# Required 
user = os.getlogin()   # user specific tag
burnin_years = 30

# Optional
num_seeds = 1          # number stochastic realizations

min_xTLH = 0           # minimum x_Temporary_Larval_Habitat = 10^min
max_xTLH = 1.4         # maximum x_Temporary_Larval_Habitat = 10^max

num_hab_samples = 30   # number xTLH samples between min and max



#########################
# Set Config Parameters #
#########################

def set_param_fn(config):
    """
    This function is a callback that is passed to emod-api.config to set config parameters, including the malaria defaults.
    """
    import emodpy_malaria.malaria_config as conf
    # Setup config using team defaults
    config = conf.set_team_defaults(config, manifest)
    # Vectors
    conf.add_species(config, manifest, ["gambiae", "arabiensis", "funestus"])
    # Climate
    config.parameters.Air_Temperature_Filename = os.path.join('climate','dtk_15arcmin_air_temperature_daily.bin')
    config.parameters.Land_Temperature_Filename = os.path.join('climate','dtk_15arcmin_air_temperature_daily.bin')
    config.parameters.Rainfall_Filename = os.path.join('climate','dtk_15arcmin_rainfall_daily.bin')
    config.parameters.Relative_Humidity_Filename = os.path.join('climate', 'dtk_15arcmin_relative_humidity_daily.bin')
    # Serialization
    config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
    config.parameters.Serialization_Time_Steps = [365 * burnin_years]
    config.parameters.Serialization_Mask_Node_Write = 0
    config.parameters.Serialization_Precision = "REDUCED"
    # Other parameters
    config.parameters.Simulation_Duration = burnin_years*365
    

    return config

########################
# Set Sweep Parameters #
########################

def set_param(simulation, param, value):
    """
    Set specific parameter value
    Args:
        simulation: idmtools Simulation
        param: parameter
        value: new value
    Returns:
        dict
    """
    return simulation.task.set_parameter(param, value)

##################
# Build Campaign #
##################

def build_camp():
    """
    This function builds a campaign input file for the DTK using emod_api.
    """

    camp.schema_path = manifest.schema_file

    
    return camp

#################
# Serialization #
#################
def update_serialize_parameters(simulation, df, x: int):

    path = df["serialized_file_path"][x]
    seed = int(df["Run_Number"][x])
    
    simulation.task.set_parameter("Serialized_Population_Filenames", df["Serialized_Population_Filenames"][x])
    simulation.task.set_parameter("Serialized_Population_Path", os.path.join(path, "output"))
    simulation.task.set_parameter("Run_Number", seed) #match pickup simulation run number to burnin simulation

    return {"Run_Number":seed}


######################
# Build Demographics #
######################

def build_demog():
    """
    This function builds a demographics input file for the DTK using emod_api.
    """
    
    # From template node
#    demog = Demographics.from_template_node(lat=1, lon=2, pop=1000, name="Example_Site")
#    demog.SetEquilibriumVitalDynamics()
#    age_distribution = Distributions.AgeDistribution_SSAfrica
#    demog.SetAgeDistribution(age_distribution)

    # From input file csv
    demog = Demographics.from_csv(input_file = os.path.join(manifest.input_dir,"demographics","nodes.csv"), id_ref="indie_test", init_prev = 0.01, include_biting_heterogeneity = True)
    demog.SetEquilibriumVitalDynamics()
    age_distribution = Distributions.AgeDistribution_SSAfrica
    demog.SetAgeDistribution(age_distribution)
    
    return demog


#####################
# Build Simulations #
#####################

def general_sim(selected_platform):
    """
    This function is designed to be a parameterized version of the sequence of things we do 
    every time we run an emod experiment. 
    """
    # Platform #
    ############
    # Set platform and associated values, such as the maximum number of jobs to run at one time
    platform = Platform(selected_platform, job_directory=manifest.job_directory, partition='b1139', time='2:00:00',
                            account='b1139', modules=['singularity'], max_running_jobs=10)

    # Task #
    ########
    # create EMODTask 
    print("Creating EMODTask (from files)...")

    
    task = EMODTask.from_default2(
        config_path="config.json",
        eradication_path=manifest.eradication_path,
        campaign_builder=build_camp,
        schema_path=manifest.schema_file,
        param_custom_cb=set_param_fn,
        ep4_custom_cb=None,
        demog_builder=build_demog,
        plugin_report=None
    )
    
    
    # set the singularity image to be used when running this experiment
    task.set_sif(manifest.SIF_PATH, platform)
    
    # add weather directory as an asset
    task.common_assets.add_directory(os.path.join(manifest.input_dir, "climate"), relative_path="climate")

    # Builder #
    ########### 
    # add builder
    builder = SimulationBuilder()
    builder.add_sweep_definition(partial(set_param, param='x_Temporary_Larval_Habitat'), np.logspace(min_xTLH, max_xTLH, num_hab_samples)) # sweep over xTLH
    builder.add_sweep_definition(partial(set_param, param='Run_Number'), range(num_seeds)) # sweep over run_number
    
    # Reports #
    ###########
    # add report event recorder
    #add_event_recorder(task, event_list=["HappyBirthday", "Births"],
    #                   start_day=(sim_years-2)*365, end_day=sim_years*365, node_ids=[4,5], min_age_years=0,
    #                   max_age_years=100)
    
    # add malaria summary report
    add_malaria_summary_report(task, manifest, start_day=(burnin_years-2)*365, end_day=burnin_years*365, reporting_interval=30,
                               age_bins=[0.25, 5, 15, 115],
                               max_number_reports=30,
                               filename_suffix='monthly',
                               pretty_format=True)

    # create experiment from builder
    experiment = Experiment.from_builder(builder, task, name=f'{user}_indie_burnin')

    # Run Experiment #
    ##################
    # The last step is to call run() on the ExperimentManager to run the simulations.
    experiment.run(wait_until_done=True, platform=platform)


    # Check result
    if not experiment.succeeded:
        print(f"Experiment {experiment.uid} failed.\n")
        exit()

    print(f"Experiment {experiment.uid} succeeded.")



if __name__ == "__main__":
    import emod_malaria.bootstrap as dtk
    import pathlib
    import argparse

    dtk.setup(pathlib.Path(manifest.eradication_path).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--local', action='store_true', help='select slurm_local')
    args = parser.parse_args()
    if args.local:
        selected_platform = "SLURM_LOCAL"
    else:
        selected_platform = "SLURM_BRIDGED"
    
    general_sim(selected_platform)