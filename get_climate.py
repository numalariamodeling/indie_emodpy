from emodpy_malaria.weather import *
import os

tag = 'indie_test'

start_year = "2018"
end_year = "2020"

start = "".join((start_year,"001"))
end = "".join((end_year,"365"))

demo_csv = 'nodes.csv'

dir1 = "/".join(("simulation_inputs/climate","-".join((start_year,end_year))))

if not os.path.exists(dir1):
    os.makedirs(dir1)

dir2 = "/".join((dir1,tag))

if not os.path.exists(dir2):
    os.makedirs(dir2)

csv_file=os.path.join(dir1,"".join((tag,"weather.csv")))

# Request weather files
wa = WeatherArgs(site_file= "".join(("simulation_inputs/demographics/",demo_csv)),
                 start_date=int(start),
                 end_date=int(end),
                 node_column="node_id",
                 id_reference=tag)

wr: WeatherRequest = WeatherRequest(platform="COMPS")
wr.generate(weather_args=wa, request_name=tag)
wr.download(local_dir=dir1)

print(f"Original files are downloaded in dir: {dir1}") 

df, wa = weather_to_csv(weather_dir = dir1, csv_file=csv_file)
df.to_csv(csv_file)

#
## Update air temperate values and save
#df["airtemp"] = 30
#df["landtemp"]= 30
#
## Shift weather (rain) left 120 days
#max_step = df['steps'].max()
#df['steps']= df['steps'] - 120
#df.loc[(df['steps']<=0),"steps"] = df.loc[(df['steps']<=0),"steps"] + max_step
#
#weather_columns = {WeatherVariable.AIR_TEMPERATURE: "airtemp",
#                   WeatherVariable.LAND_TEMPERATURE: "landtemp",
#                   WeatherVariable.RELATIVE_HUMIDITY: "humidity",
#                   WeatherVariable.RAINFALL: "rainfall"}
#
#weather_filenames = {WeatherVariable.AIR_TEMPERATURE: "_".join((tag,"air_temperature_daily.bin")),
#                     WeatherVariable.LAND_TEMPERATURE: "_".join((tag,"land_temperature_daily.bin")),
#                     WeatherVariable.RELATIVE_HUMIDITY: "_".join((tag,"relative_humidity_daily.bin")),
#                     WeatherVariable.RAINFALL: "_".join((tag,"rainfall_daily.bin"))}
#
#
#ws2 = csv_to_weather(csv_data=df, attributes=wa, weather_dir=dir2, weather_columns=weather_columns, weather_file_names = weather_filenames)
#ws2.to_files(dir_path=dir2)
#print(f"Adjusted Files are downloaded in dir: {dir2}")  # same as out_dir