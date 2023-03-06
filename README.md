# indie_emodpy
Setting up INDIE modeling workflow using emodpy


## Directory Structure

### generic files
*get_climate.py* - generates climate files from csv using ERA5    
*idmtools.ini*  -   
*manifest.py*  - 
*utils_slurm.py* -  

### simulation_inputs/  
- **Climate/**  -  
- **Demographics/**  
    - *nodes.csv*

### analyzers/ 
- *InsetChartAnalyzer.py* 
- *MonthlyPfPRAnalyzer.py*  

### experiments/simulation_outputs/  
- *All_Age_InsetChart.csv*
- *PfPR_ClinicalIncidence_monthly.csv*

### simulation files
*run_burnin.py* - vary xTLH and case management rates 
*run_pickup.py* 

