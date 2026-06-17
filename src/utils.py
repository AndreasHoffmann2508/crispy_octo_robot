import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_scatterplot(df, x_col, y_col, title, xlabel, ylabel):
    """
    This function creates a scatter plot with a linear regression line from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x_col (str): The column in the DataFrame to use for the x-axis.
    y_col (str): The column in the DataFrame to use for the y-axis.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """

    # Create the plot
    plt.figure(figsize=(7, 7))
    sns.regplot(x=df[x_col], y=df[y_col], scatter_kws={"alpha": 0.3})

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show the plot
    plt.show()


def get_a_random_chunk_property(data):
    """
    This function only serves an example of fetching some of the properties
    from the data.
    Indeed, all the content in "data" may be useful for your project!
    """

    chunk_index = np.random.choice(len(data))

    date_list = list(data[chunk_index]["near_earth_objects"].keys())

    date = np.random.choice(date_list)

    objects_data = data[chunk_index]["near_earth_objects"][date]

    object_index = np.random.choice(len(objects_data))

    object = objects_data[object_index]

    properties = list(object.keys())
    property = np.random.choice(properties)

    print("date:", date)
    print("NEO name:", object["name"])
    print(f"{property}:", object[property])


def load_data_from_google_drive(url):
    url_processed='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url_processed)
    return df






# Function that cleans the dataframe for different outliers and other irrelevant data
def clean_taxi_data(df, remove_outliers=True):
    """
    Clean taxi trip data by removing invalid records.
    
    Parameters:
    df (pandas.DataFrame): The raw taxi data DataFrame
    remove_outliers (bool): If True, also removes extreme outliers
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame with invalid records removed
    """
    
    df_clean = df.copy()
    rows_before = len(df_clean)
    
    original_rows = len(df_clean)
    print(f"Starting with {original_rows:,} records")
    
    # Remove negative fare amounts
    df_clean = df_clean[df_clean["fare_amount"] > 0]
    print(f"  After removing negative fares: {len(df_clean):,} records (-{original_rows - len(df_clean)})")
    
    rows_before = len(df_clean)
    
    # Remove zero passenger counts
    df_clean = df_clean[df_clean["passenger_count"] > 0]
    print(f"  After removing zero passengers: {len(df_clean):,} records (-{rows_before - len(df_clean)})")
    
    rows_before = len(df_clean)

    # Remove excessive passenger counts (more than 6 seems unlikely for a taxi)
    df_clean = df_clean[df_clean["passenger_count"] <= 6]
    print(f"  After removing excessive passengers: {len(df_clean):,} records (-{rows_before - len(df_clean)})")
    
    rows_before = len(df_clean)
    
    # Remove zero trip distances
    df_clean = df_clean[df_clean["trip_distance"] > 0]
    print(f"  After removing zero distance: {len(df_clean):,} records (-{rows_before - len(df_clean)})")
    
    if remove_outliers:
        rows_before = len(df_clean)
        
        # Remove extreme fares (> 300)
        df_clean = df_clean[df_clean["fare_amount"] <= 300]
        print(f"  After removing extreme fares (>300): {len(df_clean):,} records (-{rows_before - len(df_clean)})")
        
        rows_before = len(df_clean)
        
        # Remove extreme trip distances (> 100 km seems reasonable)
        df_clean = df_clean[df_clean["trip_distance"] <= 100]
        print(f"  After removing extreme distances (>100): {len(df_clean):,} records (-{rows_before - len(df_clean)})")
    
    total_removed = original_rows - len(df_clean)
    removed_pct = 100 * total_removed / original_rows if original_rows else 0
    print(f"Final cleaned dataset: {len(df_clean):,} records")
    print(f"Removed {total_removed:,} records total ({removed_pct:.1f}%)")
    
    return df_clean




import matplotlib.pyplot as plt
import seaborn as sns


# Visualizing the different distributions for the trip distance, fare amounts and passenger counts from the dataframe
def plot_trip_distributions(df, taxi_type="Taxi"):
    """
    Plots the distribution of trip distance, fare amount, 
    and passenger count for a given taxi dataset.
    
    Parameters:
    df (pandas.DataFrame): The cleaned taxi data.
    taxi_type (str): The name of the taxi type (e.g., 'Yellow' or 'Green') for the titles.
    """
    # Creates a clean white background (eliminates clutter)
    sns.set_theme(style="whitegrid")

    # Creates a figure with three graphs to create a nice overview
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Trip Distance (Histogram)
    sns.histplot(data=df, x='trip_distance', bins=80, ax=axes[0], color='skyblue', kde=False)
    axes[0].set_title(f'Distribution of Trip Distance ({taxi_type} Taxis)', fontsize=14)
    axes[0].set_xlabel('Trip Distance (miles)', fontsize=12)
    axes[0].set_ylabel('Number of Trips', fontsize=12)
    axes[0].set_xlim(0, 20) 

    # 2. Fare Amount (Histogram)
    sns.histplot(data=df, x='fare_amount', bins=80, ax=axes[1], color='lightgreen', kde=False)
    axes[1].set_title(f'Distribution of Fare Amount ({taxi_type} Taxis)', fontsize=14)
    axes[1].set_xlabel('Fare Amount ($)', fontsize=12)
    axes[1].set_ylabel('')
    axes[1].set_xlim(0, 80)

    # 3. Passenger Count (Bar chart / Countplot)
    # Added hue='passenger_count' and legend=False to avoid warnings i newer Seaborn-versions
    sns.countplot(data=df, x='passenger_count', ax=axes[2], palette="viridis", hue='passenger_count', legend=False)
    axes[2].set_title(f'Passenger Count Distribution ({taxi_type} Taxis)', fontsize=14)
    axes[2].set_xlabel('Number of Passengers', fontsize=12)
    axes[2].set_ylabel('')

    # Adjust the layout
    plt.tight_layout()
    plt.show()


# The same as before but percentage instead of actual numbers
def plot_trip_distributions_percent(df, taxi_type="Taxi"):
    """
    Plots the distribution of trip distance, fare amount, 
    and passenger count as percentages (%) for a given taxi dataset.
    
    Parameters:
    df (pandas.DataFrame): The cleaned taxi data.
    taxi_type (str): The name of the taxi type (e.g., 'Yellow' or 'Green') for the titles.
    """
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Trip Distance (Histogram - nu med procenter)
    # stat='percent' makes sure the y-axis shows the percent of the total 
    sns.histplot(data=df, x='trip_distance', bins=80, stat='percent', ax=axes[0], color='skyblue', kde=False)
    axes[0].set_title(f'Distribution of Trip Distance ({taxi_type} Taxis)', fontsize=14)
    axes[0].set_xlabel('Trip Distance (miles)', fontsize=12)
    axes[0].set_ylabel('Percentage of Trips (%)', fontsize=12)
    axes[0].set_xlim(0, 20) 

    # 2. Fare Amount (Histogram - in percentage)
    sns.histplot(data=df, x='fare_amount', bins=80, stat='percent', ax=axes[1], color='lightgreen', kde=False)
    axes[1].set_title(f'Distribution of Fare Amount ({taxi_type} Taxis)', fontsize=14)
    axes[1].set_xlabel('Fare Amount ($)', fontsize=12)
    axes[1].set_ylabel('Percentage of Trips (%)', fontsize=12)
    axes[1].set_xlim(0, 80)

    # 3. Passenger Count (Bar chart - in percentage)
    # First the percentage is calculated to make a proper bar figure
    pct_data = df['passenger_count'].value_counts(normalize=True).sort_index() * 100
    
    sns.barplot(x=pct_data.index, y=pct_data.values, ax=axes[2], palette="viridis", hue=pct_data.index, legend=False)
    axes[2].set_title(f'Passenger Count Distribution ({taxi_type} Taxis)', fontsize=14)
    axes[2].set_xlabel('Number of Passengers', fontsize=12)
    axes[2].set_ylabel('Percentage of Trips (%)', fontsize=12)

    plt.tight_layout()
    plt.show()




# This functions visualizes the correlations between the different variables trip fare, distance and passenger count
def plot_trip_relationships(df, taxi_type="Taxi", sample_size=10000):
    """
    Plots the relationships between trip distance, fare amount, and passenger count.
    Generates a scatter plot, a box plot, and a correlation heatmap.
    
    Parameters:
    df (pandas.DataFrame): The cleaned taxi data.
    taxi_type (str): The name of the taxi type (e.g., 'Yellow' or 'Green') for the titles.
    sample_size (int): Number of rows to sample for the scatterplot to prevent overplotting.
    """
    sns.set_theme(style="whitegrid")

    # Sure sample: We use 10.000 datapoints or the maximum amount if there is less than 10.000 datapoints
    n_samples = min(len(df), sample_size)
    df_sample = df.sample(n=n_samples, random_state=42)

    # Makes a figure with two plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Scatterplot: Trip Distance vs. Fare Amount
    sns.scatterplot(data=df_sample, x='trip_distance', y='fare_amount', 
                    alpha=0.3, s=15, ax=axes[0], color='#1f77b4')
    axes[0].set_title(f'Trip Distance vs. Fare Amount ({taxi_type} - {n_samples} trips)', fontsize=14)
    axes[0].set_xlabel('Trip Distance (miles)', fontsize=12)
    axes[0].set_ylabel('Fare Amount ($)', fontsize=12)

    # 2. Boxplot: Passenger Count vs. Fare Amount
    sns.boxplot(data=df, x='passenger_count', y='fare_amount', 
                ax=axes[1], palette='Set2', hue='passenger_count', legend=False)
    axes[1].set_title(f'Fare Distribution by Passenger Count ({taxi_type})', fontsize=14)
    axes[1].set_xlabel('Number of Passengers', fontsize=12)
    axes[1].set_ylabel('')
    axes[1].set_ylim(-2, 25) # Zoomes in on the main part of the data

    plt.tight_layout()
    plt.show()

    # 3. Correlations-Heatmap (Matematical correlation)
    plt.figure(figsize=(8, 5))
    
    # Calculate Pearsons variable between the three different variables
    corr_matrix = df[['trip_distance', 'fare_amount', 'passenger_count']].corr()

    # Creates a heatmap with annotation.
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
    plt.title(f'Correlation Heatmap ({taxi_type} Taxis)', fontsize=14)
    plt.show()



# Function to plot the temporal trends for the taxi rides
def plot_temporal_trends(df, taxi_type="Taxi"):
    """
    Plots temporal trends for taxi rides, including:
    - Volume of rides by time of day
    - Volume of rides by day of the week
    - Average fare amount by time of day
    - Average trip distance by time of day
    
    Parameters:
    df (pandas.DataFrame): The cleaned taxi data containing 'pickup_hour' and 'pickup_dayofweek'.
    taxi_type (str): The name of the taxi type (e.g., 'Yellow' or 'Green') for the titles.
    """
    sns.set_theme(style="whitegrid")

    # Creates a 2x2 grid for the 4 graphs
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))


    # Row 1: The volume of rides (When are people driving)

    
    # 1. Rides distributed at a time of day (Hours)
    sns.histplot(data=df, x='pickup_hour', discrete=True, stat='percent', ax=axes[0,0], color='coral')
    axes[0,0].set_title(f'Rides by Time of Day ({taxi_type} Taxis)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Hour of Day (0-23)', fontsize=12)
    axes[0,0].set_ylabel('Percentage of Rides (%)', fontsize=12)
    axes[0,0].set_xticks(range(0, 24, 2)) # Viser kun hver anden time


    # Rides distributed by weekday (Day of the week)
    sns.histplot(data=df, x='pickup_dayofweek', discrete=True, stat='percent', ax=axes[0,1], color='mediumaquamarine')
    axes[0,1].set_title(f'Rides by Day of the Week ({taxi_type} Taxis)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Day of Week', fontsize=12)
    axes[0,1].set_ylabel('Percentage of Rides (%)', fontsize=12)
    axes[0,1].set_xticks(range(0, 7))
    axes[0,1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    

    # Row 2: Correlations between time and fare/distance)

    # 3. Average prices distributed over the time of day
    sns.lineplot(data=df, x='pickup_hour', y='fare_amount', estimator='mean', errorbar=None, ax=axes[1,0], color='darkred', marker='o')
    axes[1,0].set_title(f'Average Fare by Time of Day ({taxi_type} Taxis)', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Hour of Day (0-23)', fontsize=12)
    axes[1,0].set_ylabel('Average Fare ($)', fontsize=12)
    axes[1,0].set_xticks(range(0, 24, 2))

    # 4. Average distance distributed over the time of dat
    sns.lineplot(data=df, x='pickup_hour', y='trip_distance', estimator='mean', errorbar=None, ax=axes[1,1], color='navy', marker='o')
    axes[1,1].set_title(f'Average Trip Distance by Time of Day ({taxi_type} Taxis)', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Hour of Day (0-23)', fontsize=12)
    axes[1,1].set_ylabel('Average Distance (miles)', fontsize=12)
    axes[1,1].set_xticks(range(0, 24, 2))

    # Makes a tight nice layout
    plt.tight_layout()
    plt.show()





# Function to prepare the datafram so it fits with the prophet functions
def prepare_prophet_data(taxi_type='yellow', years=None, months=None):
    """
    Downloads, aggregates, and prepares daily ride counts for Prophet forecasting.
    Processes data month-by-month to remain memory-efficient.
    
    Parameters:
    taxi_type (str): 'yellow' or 'green'
    years (list): List of year strings, e.g., ['2018', '2019']
    months (list): List of month strings, e.g., ['01', '02', ..., '12']
    
    Returns:
    pandas.DataFrame: Aggregated daily counts ready for Prophet.
    """
    if years is None:
        years = ['2018', '2019']
    if months is None:
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    base_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{{year}}-{{month}}.parquet"

    # Define the correct datetime column based on taxi type
    if taxi_type.lower() == 'yellow':
        dt_col = 'tpep_pickup_datetime'
    elif taxi_type.lower() == 'green':
        dt_col = 'lpep_pickup_datetime'
    else:
        raise ValueError("taxi_type must be either 'yellow' or 'green'")

    daily_list = []

    print(f"Starting memory-efficient download for {taxi_type.capitalize()} Taxis ({years[0]}-{years[-1]})...\n")

    for year in years:
        for month in months:
            url = base_url.format(year=year, month=month)
            try:
                print(f"Fetching and aggregating data for {year}-{month}...")
                df_temp = pd.read_parquet(url)
                
                df_temp[dt_col] = pd.to_datetime(df_temp[dt_col])
                df_temp['ds'] = df_temp[dt_col].dt.date
                
                daily_counts = df_temp.groupby('ds').size().reset_index(name='y')
                daily_list.append(daily_counts)
                
                print(f"  ✓ Condensed to {len(daily_counts)} days. Freeing memory...")
                del df_temp
                
            except Exception as e:
                print(f"  ✗ Error fetching {year}-{month}: {e}")

    if daily_list:
        df_prophet = pd.concat(daily_list, ignore_index=True)
        
        # Filter to ensure we only have data strictly within the specified years
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        start_date = f"{years[0]}-01-01"
        end_date = f"{years[-1]}-12-31"
        
        df_prophet = df_prophet[(df_prophet['ds'] >= start_date) & 
                                (df_prophet['ds'] <= end_date)]
        
        # Group by 'ds' again to sum overlapping days across files (if any)
        df_prophet = df_prophet.groupby('ds')['y'].sum().reset_index()
        df_prophet = df_prophet.sort_values('ds')
        
        expected_days = len(years) * 365
        print(f"\nSuccess! Dataset is ready. Rows: {len(df_prophet)} (Expected approx. {expected_days})")
        
        return df_prophet
    else:
        print("\nFailed to create dataset. No data was downloaded.")
        return None








from keplergl import KeplerGl
import webbrowser
import os


def create_taxi_hotspot_map(data, df_zones, dataset_name):
    # Building the filename inside the function by an f-string
    output_file = f'{dataset_name}_3D.html'
    
    df_sample = data.copy()
    """
    Takes cleaned taxi data and zone data, merges them, 
    identifies top 10 hotspots, and generates a 3D Kepler.gl map.
    """
  

    # Merge PICKUP - Remember to include 'zone' from df_zones!
    df_spatial = pd.merge(df_sample, 
                          df_zones[['LocationID', 'lat', 'lng', 'zone']], 
                          left_on='PULocationID', 
                          right_on='LocationID', 
                          how='inner')
    df_spatial = df_spatial.rename(columns={'lat': 'pickup_lat', 
                                            'lng': 'pickup_lng',
                                            'zone': 'pickup_zone'})
    df_spatial = df_spatial.drop(columns=['LocationID'])

    # Merge DROPOFF - Remember to include 'zone' from df_zones!
    df_spatial = pd.merge(df_spatial, 
                          df_zones[['LocationID', 'lat', 'lng', 'zone']], 
                          left_on='DOLocationID', 
                          right_on='LocationID', 
                          how='inner')
    df_spatial = df_spatial.rename(columns={'lat': 'dropoff_lat', 
                                            'lng': 'dropoff_lng',
                                            'zone': 'dropoff_zone'})
    df_spatial = df_spatial.drop(columns=['LocationID'])


    # Analysis of PICKUPS
    hotspot_pickup = df_spatial.groupby(['PULocationID', 'pickup_zone']).agg({
        'fare_amount': 'count',  # Number of rides
        'pickup_lat': 'first',
        'pickup_lng': 'first'
    }).rename(columns={'fare_amount': 'ride_count'}).reset_index()

    hotspot_pickup = hotspot_pickup.sort_values('ride_count', ascending=False)

    print("\nTOP 10 PICKUP HOTSPOTS:")
    # Print only the zone name and the count for easy readability:
    print(hotspot_pickup[['pickup_zone', 'ride_count']].head(10).to_string(index=False))

    # Analysis of DROPOFFS
    hotspot_dropoff = df_spatial.groupby(['DOLocationID', 'dropoff_zone']).agg({
        'fare_amount': 'count',  # Number of rides
        'dropoff_lat': 'first',
        'dropoff_lng': 'first'
    }).rename(columns={'fare_amount': 'ride_count'}).reset_index()

    hotspot_dropoff = hotspot_dropoff.sort_values('ride_count', ascending=False)

    print("\nTOP 10 DROPOFF HOTSPOTS:")
    print(hotspot_dropoff[['dropoff_zone', 'ride_count']].head(10).to_string(index=False))


    ## KEPLER MAP (3D COLUMNS)
    pickup_agg = hotspot_pickup[['pickup_lat', 'pickup_lng', 'ride_count']].rename(columns={'pickup_lat': 'lat', 'pickup_lng': 'lng'})
    dropoff_agg = hotspot_dropoff[['dropoff_lat', 'dropoff_lng', 'ride_count']].rename(columns={'dropoff_lat': 'lat', 'dropoff_lng': 'lng'})

    nyc_3d_config = {
        "version": "v1",
        "config": {
            "visState": {
                "layers": [
                    {
                        "id": "pickup_layer",
                        "type": "hexagon",
                        "config": {
                            "dataId": "Pickup Hotspots",
                            "label": "Pickup Hotspots",
                            "columns": {"lat": "lat", "lng": "lng"},
                            "isVisible": True,
                            "visConfig": {
                                "opacity": 0.8,
                                "worldUnitSize": 0.4, 
                                "enable3d": True,
                                "elevationScale": 25, 
                                "colorRange": {
                                    "name": "Global Warming",
                                    "type": "sequential",
                                    "category": "Uber",
                                    "colors": ["#FFC300", "#F1920E", "#E3611C", "#C70039", "#900C3F", "#5A1846"]
                                }
                            }
                        },
                        "visualChannels": {
                            # colorField and sizeField are set to 'ride_count' to control color + height
                            "colorField": {"name": "ride_count", "type": "integer"},
                            "colorScale": "quantize",
                            "sizeField": {"name": "ride_count", "type": "integer"},
                            "sizeScale": "linear"
                        }
                    }
                ]
            },
            "mapState": {
                "bearing": 25,     
                "dragRotate": True,
                "latitude": 40.730610,
                "longitude": -73.935242,
                "pitch": 50,       
                "zoom": 10
            }
        }
    }

    map_kepler = KeplerGl(height=800, config=nyc_3d_config)
    map_kepler.add_data(data=pickup_agg, name='Pickup Hotspots')
    map_kepler.add_data(data=dropoff_agg, name='Dropoff Hotspots')

    map_kepler.save_to_html(file_name=output_file)
    print(f"\nMap successfully saved as: {output_file}")


    
    # Finds the full pathfile on the computer
    full_path = os.path.abspath(output_file)
    # Opens the file in the standard browser
    webbrowser.open(f"file://{full_path}")

    return map_kepler


def create_top_routes_map(df_taxi, df_zones, top_n, output_file):
    """
    Finds the most popular routes from zone to zone and visualizes them 
    with arcs in Kepler.gl.
    """
    print(f"Calculating the top {top_n} most popular routes...")
    
    # 1. Count the number of trips for each unique route (PULocationID -> DOLocationID)
    route_counts = df_taxi.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name='trip_count')
    
    # 2. Find the top N most popular routes
    top_routes = route_counts.sort_values('trip_count', ascending=False).head(top_n)
    
    # 3. Merge start coordinates and zone names (Pickup)
    df_routes_mapped = pd.merge(
        top_routes, 
        df_zones[['LocationID', 'lat', 'lng', 'zone']], 
        left_on='PULocationID', 
        right_on='LocationID', 
        how='inner'
    )
    df_routes_mapped = df_routes_mapped.rename(columns={
        'lat': 'start_lat', 
        'lng': 'start_lng', 
        'zone': 'start_zone'
    }).drop(columns=['LocationID'])
    
    # 4. Merge end coordinates and zone names (Dropoff)
    df_routes_mapped = pd.merge(
        df_routes_mapped, 
        df_zones[['LocationID', 'lat', 'lng', 'zone']], 
        left_on='DOLocationID', 
        right_on='LocationID', 
        how='inner'
    )
    df_routes_mapped = df_routes_mapped.rename(columns={
        'lat': 'end_lat', 
        'lng': 'end_lng', 
        'zone': 'end_zone'
    }).drop(columns=['LocationID'])

    print("\nHere is a sneak peek at the most popular routes:")
    print(df_routes_mapped[['start_zone', 'end_zone', 'trip_count']].head(10).to_string(index=False))

    # 5. Kepler.gl configuration to create "Arcs"
    arc_config = {
        "version": "v1",
        "config": {
            "visState": {
                "layers": [
                    {
                        "id": "route_arcs",
                        "type": "arc",
                        "config": {
                            "dataId": "Top Routes",
                            "label": "Popular Taxi Routes",
                            "columns": {
                                "lat0": "start_lat",
                                "lng0": "start_lng",
                                "lat1": "end_lat",
                                "lng1": "end_lng"
                            },
                            "isVisible": True,
                            "visConfig": {
                                "opacity": 0.8,
                                "thickness": 2,
                                "colorRange": {
                                    "name": "ColorBrewer YlOrRd-6",
                                    "type": "sequential",
                                    "category": "ColorBrewer",
                                    "colors": ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]
                                },
                                "sizeRange": [1, 10],
                                "targetColor": [150, 0, 90] # Gives the dropoff a slightly different tone
                            }
                        },
                        "visualChannels": {
                            "colorField": {"name": "trip_count", "type": "integer"},
                            "colorScale": "quantize",
                            "sizeField": {"name": "trip_count", "type": "integer"},
                            "sizeScale": "linear"
                        }
                    }
                ]
            },
            "mapState": {
                "bearing": 24,
                "dragRotate": True,
                "latitude": 40.75,
                "longitude": -73.95,
                "pitch": 45,
                "zoom": 10
            }
        }
    }

    # 6. Generate the map
    route_map = KeplerGl(height=800, config=arc_config)
    route_map.add_data(data=df_routes_mapped, name='Top Routes')
    
    # Save and open automatically
    route_map.save_to_html(file_name=output_file)
    print(f"\nMap successfully saved as: {output_file}")
    
    # Finds the full file-path on the computer
    full_path = os.path.abspath(output_file)
    
    # Directly opens the file in the standard browser
    webbrowser.open(f"file://{full_path}")
    
    return route_map





from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Creating and evaluating the prophet forecast
def evaluate_prophet_forecast(df, taxi_type="Yellow", split_date='2019-11-01', zoom_date='2019-10-01'):
    """
    Trains a Prophet model on the provided time-series data, evaluates its accuracy 
    using MAE, RMSE, and MAPE, and visualizes the forecast.
    
    Parameters:
    df (pandas.DataFrame): The prepared daily ride counts.
    taxi_type (str): The name of the taxi type (e.g., 'Yellow' or 'Green').
    split_date (str): The date separating training and testing data.
    zoom_date (str): The start date for the zoomed-in plot view.
    
    Returns:
    model (Prophet): The trained Prophet model.
    forecast (pandas.DataFrame): The generated future forecast.
    """
    split_dt = pd.to_datetime(split_date)
    zoom_dt = pd.to_datetime(zoom_date)
    
    # 1. Train / Test Split – The time based splitting approach
    train = df[df['ds'] < split_dt].copy()
    test = df[df['ds'] >= split_dt].copy()

    # 2. Train the Prophet Model with the training data
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name='US')
    model.fit(train)

    # 3. Generate forecast for test period 
    future = model.make_future_dataframe(periods=len(test), freq='D')
    forecast = model.predict(future)

    # 4. Evaluation Metrics (MAE, RSME and MAPE)
    forecast_test = forecast[forecast['ds'] >= test['ds'].min()]
    y_pred = forecast_test['yhat'].values
    y_true = test['y'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"--- Evaluation of {taxi_type} Taxi Forecast ---")
    print(f"MAE  (Mean Absolute Error): {mae:,.0f} rides/day")
    print(f"RMSE (Root Mean Squared Error): {rmse:,.0f} rides/day")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f} %\n")

    # Determine colors dynamically based on taxi type
    if taxi_type.lower() == 'green':
        color_actual = 'darkgreen'
        color_forecast = 'limegreen'
    else:
        color_actual = 'navy'
        color_forecast = 'darkorange'

    # 5. Visualization:
    sns.set_theme(style="white")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # CHART 1: Full Context
    ax1.plot(train['ds'], train['y'], label='Historical (Train)', color='#d3d3d3', linewidth=1.5)
    ax1.plot(test['ds'], test['y'], label='Actual (Test)', color=color_actual, linewidth=2)
    ax1.plot(forecast_test['ds'], forecast_test['yhat'], label='Prophet Forecast', color=color_forecast, linewidth=2, linestyle='--')
    ax1.fill_between(forecast_test['ds'], forecast_test['yhat_lower'], forecast_test['yhat_upper'], color=color_forecast, alpha=0.2, label='Confidence Interval')

    ax1.set_title(f'Full Context: Prophet Forecast vs Actual {taxi_type} Taxi Rides (2018-2019)', fontsize=16, fontweight='bold', loc='left')
    ax1.set_ylabel('Number of Daily Rides', fontsize=12)
    ax1.legend(frameon=False, loc='lower left')

    # CHART 2: Zoomed View (Focus)
    train_zoom = train[train['ds'] >= zoom_dt]

    ax2.plot(train_zoom['ds'], train_zoom['y'], label='Historical (Train)', color='#d3d3d3', linewidth=2)
    ax2.plot(test['ds'], test['y'], label='Actual (Test)', color=color_actual, linewidth=2, marker='o', markersize=4)
    ax2.plot(forecast_test['ds'], forecast_test['yhat'], label='Prophet Forecast', color=color_forecast, linewidth=2, linestyle='--', marker='o', markersize=4)
    ax2.fill_between(forecast_test['ds'], forecast_test['yhat_lower'], forecast_test['yhat_upper'], color=color_forecast, alpha=0.2)

    ax2.set_title(f'Zoomed View: Evaluating Forecast Accuracy (Oct-Dec 2019) - {taxi_type} Taxis', fontsize=16, fontweight='bold', loc='left')
    ax2.set_ylabel('Number of Daily Rides', fontsize=12)

    # Eliminate clutter
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)

    plt.tight_layout(pad=3.0)


    plt.show()
    
    return model, forecast