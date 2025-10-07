import pandas as pd
import folium

# Load CSV
df = pd.read_csv(r"E:\00Studies\Aalto\2ndYear\ML_Project\utd19_splits\luzern_detectors_meta.csv")

# Select unique sensors by 'detid' along with their longitude and latitude
unique_sensors = df[['detid', 'long', 'lat']].drop_duplicates()

# Reset index for cleanliness
unique_sensors.reset_index(drop=True, inplace=True)

# Display result
print(unique_sensors)



# Create a world map centered roughly at the mean of coordinates
map_center = [unique_sensors['lat'].mean(), unique_sensors['long'].mean()]
m = folium.Map(location=map_center, zoom_start=2)

# Add markers for each sensor
for _, row in unique_sensors.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=2,  # smaller size
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.7,
        popup=row['detid']
    ).add_to(m)

# Save to HTML
m.save(r"E:\00Studies\Aalto\2ndYear\ML_Project\sensors_map_smaller.html")

