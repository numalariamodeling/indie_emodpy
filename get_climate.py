from emodpy_malaria.weather import *
import os

def get_climate(tag = "default", start_year="2015", start_day="001", end_year="2016", end_day="365", demo_fname="demographics.csv", fix_temp=None):
    # Specifications #
    ##################
    # Date Range
    start = "".join((start_year,start_day))  
    end = "".join((end_year,end_day))     
    
    # Demographics
    demo = "".join(("simulation_inputs/demographics/",demo_fname))
    
    # Output folder to store climate files
    dir1 = "/".join(("simulation_inputs/climate",tag,"-".join((start,end))))
    
    if os.path.exists(dir1):
        print("Path already exists. Please check for existing climate files.")
        return
    else:
        print("Generating climate files from {} for day {} of {} to day {} of {}".format(demo,start_day,start_year,end_day,end_year))
        os.makedirs(dir1)
        csv_file=os.path.join(dir1,"weather.csv")
        # Request weather files
        wa = WeatherArgs(site_file= demo,
                         start_date=int(start),
                         end_date=int(end),
                         node_column="node_id",
                         id_reference=tag)
        
        wr: WeatherRequest = WeatherRequest(platform="Calculon")
        wr.generate(weather_args=wa, request_name=tag)
        wr.download(local_dir=dir1)
        
        print(f"Original files are downloaded in: {dir1}") 
        
        df, wa = weather_to_csv(weather_dir = dir1, csv_file=csv_file)
        df.to_csv(csv_file)
    
        if fix_temp is not None:
            df['airtemp'] = fix_temp
            df['landtemp'] = fix_temp
            weather_columns = {WeatherVariable.AIR_TEMPERATURE: "airtemp",
                               WeatherVariable.LAND_TEMPERATURE: "landtemp",
                               WeatherVariable.RELATIVE_HUMIDITY: "humidity",
                               WeatherVariable.RAINFALL: "rainfall"}
            weather_filenames = {WeatherVariable.AIR_TEMPERATURE: 'dtk_15arcmin_air_temperature_daily.bin',
                                 WeatherVariable.LAND_TEMPERATURE: "dtk_15arcmin_land_temperature_daily.bin",
                                 WeatherVariable.RELATIVE_HUMIDITY: "dtk_15arcmin_relative_humidity_daily.bin",
                                 WeatherVariable.RAINFALL: "dtk_15arcmin_rainfall_daily.bin"}
            
            ws2 = csv_to_weather(csv_data=df, attributes=wa, weather_dir=dir1, weather_columns=weather_columns, weather_file_names = weather_filenames)
            ws2.to_files(dir_path=dir1)
            
            print(f"Fixed-Temperature ({fix_temp} degrees) files are downloaded in: {dir1}")  # same as out_dir

if __name__ == "__main__":
    #get_climate(tag="indie_clusters", start_year="2011", end_year = "2020", demo_fname="clusters.csv")
    get_climate(tag="FE_example", start_year="2019", end_year="2019", demo_fname="FE_example_nodes.csv")