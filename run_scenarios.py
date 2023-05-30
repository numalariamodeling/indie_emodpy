###################
# Import Packages #
###################

# generic packages
import pathlib
import os
import numpy as np
import pandas as pd
from functools import \
    partial 
import sys
 
# from idmtools   
from idmtools.assets import Asset, AssetCollection  
from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

# from emodpy
from emodpy.emod_task import EMODTask
from emodpy.utils import EradicationBambooBuilds
from emodpy.bamboo import get_model_files

# from emod_api
import emod_api.config.default_from_schema_no_validation as dfs
import emod_api.campaign as camp
import emod_api.demographics.PreDefinedDistributions as Distributions
import emod_api.migration.migration as migration

# from emodpy-malaria
from emodpy_malaria.reporters.builtin import *
import emodpy_malaria.demographics.MalariaDemographics as Demographics
import emodpy_malaria.interventions.treatment_seeking as cm
import emodpy_malaria.interventions.bednet as itn
import emodpy_malaria.interventions.drug_campaign as dc
from emodpy_malaria.interventions.scale_larval_habitats import add_scale_larval_habitats
import emodpy_malaria.malaria_config as malaria_config

# from manifest
import manifest

# from utils_slurm
sys.path.append('../')
from utils_slurm import build_burnin_df

#######################
# Sim. Specifications #
#######################
user = os.getlogin()                                     # username for paths & naming files
tag = '30degree_HabRatio'                                  # label for experiment
phase = 'burnin'                                         # burnin or pickup?
burnin_id = 'e4dec631-7f16-4b43-9cab-f0b4b6276cb7'       # REQ. for pickup
checkpoint_id = ''   # checkpoint experiment from calibration
checkpoint = None    


# Defaults - if no checkpoint is provided
burnin_years = 30    # 1981-2010      
pickup_years = 10    # 2011-2020  
if(phase=="burnin"):
    num_seeds = 1         # number stochastic realizations
    # Vary Habitat Scale Factors
    num_xTLH_samples = 20
    min_xTLH = 0.5
    max_xTLH = 1.5 
    # Vary Habitat Ratios
    min_HR = 10
    max_HR = 100
    num_HR_samples = 10
if(phase=="pickup"): 
    num_seeds = 5        # number stochastic realizations
    # Vary Case Management
    min_CM = 0.8
    max_CM = 0.8
    num_CM_samples = 1
  
if checkpoint is not None:
    burnin_years = 37      # 1981-2017
    pickup_years = 3       # 2018-2020
    if(phase=="burnin"):
      num_seeds = 1         # number stochastic realizations
      num_samples = 10      # include top # ranked parameter sets from checkpoint
    if(phase=="pickup"): 
      num_seeds = 1        # number stochastic realizations per scenario
      

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
    conf.set_species_param(config, 'arabiensis', 'Anthropophily', 0.65, overwrite=True)
    conf.set_species_param(config, 'arabiensis', 'Indoor_Feeding_Fraction', 0.5, overwrite=True)
    conf.set_species_param(config, 'arabiensis', 'Days_Between_Feeds', 3.1, overwrite=True)
    conf.set_species_param(config, 'funestus', 'Anthropophily', 0.35, overwrite=True)
    conf.set_species_param(config, 'funestus', 'Indoor_Feeding_Fraction', 0.92, overwrite=True)
    conf.set_species_param(config, 'funestus', 'Days_Between_Feeds', 3.1, overwrite=True)
    conf.set_species_param(config, 'gambiae', 'Anthropophily', 0.85, overwrite=True)
    conf.set_species_param(config, 'gambiae', 'Indoor_Feeding_Fraction', 0.95, overwrite=True)
    conf.set_species_param(config, 'gambiae', 'Days_Between_Feeds', 3.1, overwrite=True)
    
    
    # Max Habitats  
    scale_arabiensis = 0.01
    scale_gambiae = 0.96
    scale_funestus = 0.03
    
    # Habitat Ratio (TR:Constant)
    constant = 5000000.0
    temp_rain_ratio = 1.0
    
    conf.set_species_param(config, 
                           species= 'gambiae', 
                           parameter="Habitats", 
                           value= [{"Habitat_Type": "CONSTANT", 
                                    "Max_Larval_Capacity": constant*scale_gambiae},
                                   {"Habitat_Type": "TEMPORARY_RAINFALL", 
                                    "Max_Larval_Capacity":constant*scale_gambiae*temp_rain_ratio}], 
                           overwrite=True) # delete previous habitat types 
    conf.set_species_param(config, 
                           species= 'arabiensis', 
                           parameter="Habitats", 
                           value= [{"Habitat_Type": "CONSTANT", 
                                    "Max_Larval_Capacity": constant*scale_arabiensis},
                                   {"Habitat_Type": "TEMPORARY_RAINFALL", 
                                    "Max_Larval_Capacity":constant*scale_arabiensis*temp_rain_ratio}], 
                           overwrite=True) # delete previous habitat types 
    conf.set_species_param(config, 
                           species= 'funestus', 
                           parameter="Habitats", 
                           value= [{"Habitat_Type": "CONSTANT", 
                                    "Max_Larval_Capacity": constant*scale_funestus},
                                   {"Habitat_Type": "TEMPORARY_RAINFALL", 
                                    "Max_Larval_Capacity":constant*scale_funestus*temp_rain_ratio}], 
                           overwrite=True) # delete previous habitat types 
    
    
    # Climate
    climate_root = os.path.join('climate','2011-2020','30.0degrees')
    if checkpoint is not None and phase=="pickup":
        climate_root = os.path.join('climate','2018-2020','30.0degrees')
    config.parameters.Air_Temperature_Filename = os.path.join(climate_root,'dtk_15arcmin_air_temperature_daily.bin')
    config.parameters.Land_Temperature_Filename = os.path.join(climate_root, 'dtk_15arcmin_air_temperature_daily.bin')
    config.parameters.Rainfall_Filename = os.path.join(climate_root, 'dtk_15arcmin_rainfall_daily.bin')
    config.parameters.Relative_Humidity_Filename = os.path.join(climate_root, 'dtk_15arcmin_relative_humidity_daily.bin')
        
    # Serialization
    if(phase=="burnin"):
      config.parameters.Serialized_Population_Writing_Type = "TIMESTEP"
      config.parameters.Serialization_Time_Steps = [365 * burnin_years]
      config.parameters.Serialization_Mask_Node_Write = 0
      config.parameters.Serialization_Precision = "REDUCED"
      config.parameters.Simulation_Duration = burnin_years*365
    if(phase=="pickup"):
      config.parameters.Serialized_Population_Reading_Type = "READ"
      config.parameters.Serialization_Time_Steps = [365 * burnin_years]
      config.parameters.Serialization_Mask_Node_Read = 0
      config.parameters.Simulation_Duration = pickup_years*365
    # Output
      config.parameters.Custom_Individual_Events = ["Received_ITN", "Received_SMC", "Received_Treatment"]
      
    # Migration
    config.parameters.Enable_Migration_Heterogeneity = 0

    
    return config

###########################
# Sweep Config Parameters #
###########################

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

def build_camp(scale_start_day = 1, habitat_ratio = 50.0, cm_cov_u5 =0.8):
    """
    This function builds a campaign input file for the DTK using emod_api.
    """
    camp.set_schema(manifest.schema_file)
    df = pd.read_csv(os.path.join(manifest.input_dir,"demographics","clusters.csv"))
    # Scale Temporary Rainfall relative to Constant habitat
    scales=[habitat_ratio] * len(df.index)
    lh_scales = pd.DataFrame({'NodeID': list(df['node_id']),
                              'TEMPORARY_RAINFALL':scales})
    add_scale_larval_habitats(camp, lh_scales, start_day=scale_start_day)
    
    # Scale larval habitats by node population
    
    pop_scales=[pop / 1000 for pop in list(df['pop'])]
    lh_pop_scales = pd.DataFrame({'NodeID': list(df['node_id']),
                              'CONSTANT': pop_scales,
                              'TEMPORARY_RAINFALL': pop_scales})
    
    add_scale_larval_habitats(camp, df=lh_pop_scales, start_day=scale_start_day)
    
    
    ### Calibration (no checkpoint)
    ###############################
    # Note: no interventions during burnin of calibration framework
    if(phase =="pickup"):
      ### ITN Distributions ###
      itn.add_itn_scheduled(camp, 
                            start_day = 165, 
                            demographic_coverage = 0.9, 
                            repetitions = 4, 
                            timesteps_between_repetitions = 365*3, 
                            receiving_itn_broadcast_event= "Received_ITN", 
                            # ITN parameters from malaria-bf-hbhi/simulation/setup_inputs/set_up_planned_scenarios.py
                            killing_initial_effect = 0.7,
                            killing_box_duration = 180,
                            killing_decay_time_constant = 90)
                            
      ### Case Management ###
      # Treatment-Seeking Rates by age #
      cm_coverage_by_age = [{'trigger': 'NewClinicalCase',      ## For uncomplicated symptomatic cases < 5
                                        'coverage': cm_cov_u5,
                                        'agemin': 0,
                                        'agemax': 5,
                                        'seek': 1,
                                        'rate': 0.3},
                            {'trigger': 'NewClinicalCase',      ## For uncomplicated symptomatic cases 5-15
                                        'coverage': cm_cov_u5*0.6,
                                        'agemin': 5,
                                        'agemax': 15,
                                        'seek': 1,
                                        'rate': 0.3},
                            {'trigger': 'NewClinicalCase',      ## For uncomplicated symptomatic cases 15+
                                        'coverage': cm_cov_u5*0.4,
                                        'agemin': 15,
                                        'agemax': 115,
                                        'seek': 1,
                                        'rate': 0.3},
                            {'trigger': 'NewSevereCase',        ## For severe clinical cases, all-ages
                                        'coverage': 0.8,
                                        'agemin': 0,
                                        'agemax': 115,
                                        'seek': 1,
                                        'rate': 0.5}]
      # Treatment #                                 
      cm.add_treatment_seeking(camp, 
                               start_day = 1, 
                               drug=['Artemether','Lumefantrine'],
                               targets=cm_coverage_by_age,
                               broadcast_event_name="Received_Treatment")
                               
                               
      ### SMC ###
      smc_dates = [2394, 2765, 3122, 3489]   # 4 rounds in each cycle, 1 month between rounds. Beginning in July each year 2016-2019
      # Modeled as a simple MDA
      dc.add_drug_campaign(camp, campaign_type="MDA", drug_code="SPA", 
                           start_days=smc_dates,
                           repetitions=4, 
                           tsteps_btwn_repetitions=30, 
                           coverage=0.95,
                           target_group={'agemin': 0.25, 'agemax': 5},
                           receiving_drugs_event_name="Received_SMC")

       
    return camp

#############################
# Sweep Campaign Parameters #
#############################
def update_campaign_multiple_parameters(simulation, scale_start_day, habitat_ratio):
    """
        This is a callback function that updates several parameters in the build_campaign function.
        the sweep is achieved by the itertools creating a an array of inputs with all the possible combinations
        see builder.add_sweep_definition(update_campaign_multiple_parameters function below
    Args:
        simulation: simulation object to which we will attach the callback function
        cm_cov_u5: U5 case management coverage
    Returns:
        a dictionary of tags for the simulation to use in COMPS
    """
    build_campaign_partial = partial(build_camp, scale_start_day = scale_start_day, habitat_ratio = habitat_ratio)
    simulation.task.create_campaign_from_callback(build_campaign_partial)
    return {"scale_start_day":scale_start_day, "HR": habitat_ratio}


#################
# Serialization #
#################
def update_serialize_parameters(simulation, df, x: int):
    # Serialized file path:
    path = df["serialized_file_path"][x]    
    # Other parameters from burnin that need to be carried over:
    xTLH = df["x_Temporary_Larval_Habitat"][x]
    HR = df["HR"][x]
    # Set Parameters
    simulation.task.set_parameter("Serialized_Population_Filenames", df["Serialized_Population_Filenames"][x])  # Set serialized population filename
    simulation.task.set_parameter("Serialized_Population_Path", os.path.join(path, "output"))                   # Set serialized population path
    simulation.task.set_parameter("x_Temporary_Larval_Habitat", xTLH)                                           # Grab xTLH from burnin simulation
    
    
    
    return {"xTLH":xTLH}      # Return serialized parameters as tags


######################
# Build Demographics #
######################

def build_demog():
    """
    This function builds a demographics input file for the DTK using emod_api.
    """
    
    # From template node #
    ######################
#    demog = Demographics.from_template_node(lat=1, lon=2, pop=1000, name="Example_Site")
#    demog.SetEquilibriumVitalDynamics()
#    age_distribution = Distributions.AgeDistribution_SSAfrica
#    demog.SetAgeDistribution(age_distribution)

    # From input file csv #
    #######################
    demog = Demographics.from_csv(input_file = os.path.join(manifest.input_dir,"demographics","clusters.csv"), id_ref="indie_clusters", init_prev = 0.01, include_biting_heterogeneity = True)
    demog.SetEquilibriumVitalDynamics()
    # Age Distribution
    age_distribution = Distributions.AgeDistribution_SSAfrica
    demog.SetAgeDistribution(age_distribution)
    # Individual Properties
    # Study Group (for scenarios only)
    #demog.AddIndividualPropertyAndHINT(Property="study_arm", 
    #                                   Values = ["1","2","3"],
    #                                   Initial_Distribution = [0.8,0.1,0.1]) # simple test, should pull % by cluster from demographics input file .csv
    
   
    # Waiting for fix to from_csv error... 
    # TypeError: unexpected keyword argument 'demographics_file_path'
    #migration_partial = partial(migration.from_csv, file_name = os.path.join(manifest.input_dir,"migration","local_migration.csv"), id_ref="indie_clusters")
    
    # Using this gravity parameters set from Monique's MMC work 
    #migration_partial = partial(migration.from_demog_and_param_gravity, gravity_params=[7.50395776e-06, 9.65648371e-01, 9.65648371e-01, -1.10305489e+00], id_ref='indie_clusters', migration_type=migration.Migration.REGIONAL)
    return demog#, migration_partial


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
    if(phase=="burnin"):
      # Use b1139 for longer simulations (do not exveed 100 max_running_jobs)
      platform = Platform(selected_platform, 
                          job_directory=manifest.job_directory, 
                          partition='b1139', 
                          time='6:00:00',
                          account='b1139', 
                          modules=['singularity'], 
                          max_running_jobs=100)
    if(phase=="pickup"):
      # Use p30781 for a high # of relatively short simulations
      platform = Platform(selected_platform, 
                          job_directory=manifest.job_directory, 
                          partition='short', 
                          time='4:00:00',
                          account='p30781', 
                          modules=['singularity'], 
                          max_running_jobs=1000,
                          mem = 40000)

    # Task #
    ########
    # create EMODTask #
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
    # set the singularity image to be used when running this experiment #
    task.set_sif(manifest.SIF_PATH, platform)
    # add weather directory as an asset #
    task.common_assets.add_directory(os.path.join(manifest.input_dir, "climate"), relative_path="climate")
    # Builder #
    ########### 
    # add builder #
    builder = SimulationBuilder()

    if(phase=="burnin"):
      ### Parameters to sweep over in burnin ###
      # Run number
      builder.add_sweep_definition(partial(set_param, param='Run_Number'), range(num_seeds))
      builder.add_multiple_parameter_sweep_definition(update_campaign_multiple_parameters, dict(scale_start_day=[1], habitat_ratio=np.linspace(min_HR,max_HR,num_HR_samples)))
      # x_Temporary_Larval_Habitat
      builder.add_sweep_definition(partial(set_param, param='x_Temporary_Larval_Habitat'), np.logspace(min_xTLH, max_xTLH, num_xTLH_samples))
      #builder.add_sweep_definition(partial(update_campaign_single_parameter), np.linspace(min_HR,max_HR,num_HR_samples))
    if(phase=="pickup"):
      ### Connect to burnin ###
      ## Read in serialized data
      burnin_df = build_burnin_df(burnin_id, platform, burnin_years*365) 
      ## Pick up parameters
      # x_Temporary_Larval_Habitat
      builder.add_sweep_definition(partial(update_serialize_parameters, df=burnin_df), range(len(burnin_df.index)))
      
      ### New parameters to sweep over in pickup ###
      # Run number
      builder.add_sweep_definition(partial(set_param, param='Run_Number'), range(num_seeds))
      # Case mangement coverage
      #builder.add_sweep_definition(partial(update_campaign_single_parameter), np.linspace(min_CM, max_CM, num_CM_samples))
   
    
    # Reporting #
    #############
    if(phase =="burnin"):
      # Report over last year
      start_report = (burnin_years-1)*365
      end_report = burnin_years*365
    if(phase =="pickup"):
      # Report over last 3 years
      start_report = (pickup_years-3)*365
      end_report = pickup_years*365
      # add report event counter #
      add_event_recorder(task, 
                         event_list=["Received_ITN", "Received_Treatment", "Received_SMC"],
                         start_day=start_report,
                         end_day=end_report,
                         min_age_years=0,
                         max_age_years=100)
      demo_df = pd.read_csv(os.path.join(manifest.input_dir, "demographics", "clusters.csv"))
      for node in demo_df['node_id']:
          add_report_event_counter(task, manifest,
                                   start_day = start_report,
                                   end_day = end_report,
                                   node_ids = [node],
                                   min_age_years = 0,
                                   max_age_years = 100,
                                   event_trigger_list = ["Received_ITN", "Received_Treatment", "Received_SMC"],
                                   filename_suffix = "_".join(("node",str(node))))
    
    ## Reports that are always on: 
    ##############################
    # monthly malaria summary report #
#    add_malaria_summary_report(task, manifest, start_day=start_report, end_day=end_report, reporting_interval=30,
#                               age_bins=[0.25, 5, 15, 115],
#                               max_number_reports=30,
#                               filename_suffix='monthly',
#                               pretty_format=True)
   # filtered spatial malaria report
    add_spatial_report_malaria_filtered(task, manifest, start_day = start_report, end_day = end_report, reporting_interval = 1,
                                        node_ids =None, min_age_years = 0.25, max_age_years = 100,
                                        spatial_output_channels = ["Population", "Daily_Bites_Per_Human","PCR_Parasite_Prevalence","New_Clinical_Cases", "Air_Temperature", "Rainfall"] ,
                                        filename_suffix = "all_ages")
    

    # create experiment from builder #
    ##################################
    experiment = Experiment.from_builder(builder, task, name=f'{user}_{tag}_{phase}')

    # Run Experiment #
    ##################
    # The last step is to call run() on the ExperimentManager to run the simulations. #
    experiment.run(wait_until_done=True, platform=platform)
    # Check result #
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