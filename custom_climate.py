import tempfile
from pathlib import Path
import os

from emodpy_malaria.weather import *

start = 2011
end = 2020
span = "-".join((str(start),str(end)))
root = os.path.join("simulation_inputs","climate",span)
os.makedirs(root, exist_ok=True)
temp = 30.0


# ---| Request weather files |--- 

# Request weather time series from 2015 to 2016, for nodes listed in a .csv file
wr = generate_weather(platform="Calculon",
                      request_name = "indie_clusters",
                      id_reference = "indie_clusters",
                      site_file="simulation_inputs/demographics/clusters.csv",
                      start_date=2018,
                      end_date=2020,
                      node_column="node_id",
                      local_dir=os.path.join(root,"base"))   

# Generated weather files are downloaded into a local dir, if specified. 
print("\n".join(wr.files))


# ---| Convert to a dataframe |--- 

# Convert weather files to a dataframe and weather attributes
df, wa = weather_to_csv(weather_dir=wr.local_dir)     
print(df.head())


# ---| Modify and save |---

# Modify weather data (for example) 
df["landtemp"] *= 0.000001
df["landtemp"] += temp
df["airtemp"] *= 0.000001
df["airtemp"] += temp 
print(df.head())

# Generate modified weather files (see "Weather Objects" section
specs = "".join((str(temp),"degrees"))
weather_dir = os.path.join(root,specs)
ws = csv_to_weather(csv_data=df, attributes=wa, weather_dir=weather_dir)
print(list(Path(weather_dir).glob("*")))