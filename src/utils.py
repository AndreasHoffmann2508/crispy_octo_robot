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

    print("\nHere is the most popular routes:")
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
                "zoom": 10.3
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



import random
def get_a_random_chunk_property(data):
    """Returnerer en tilfældig egenskab fra en tilfældig asteroide i datasættet."""
    chunk = random.choice(data)
    date  = random.choice(list(chunk['near_earth_objects'].keys()))
    neo   = random.choice(chunk['near_earth_objects'][date])
    prop  = random.choice(list(neo.keys()))
    return {prop: neo[prop]}


def build_neo_dataframe(data):
    """
    Bygger et Pandas DataFrame fra rådata fra NASA\'s NEO API.

    Returnerer én række per (asteroide, dato) med:
        id, name, date, size_km, is_hazardous,
        distance_km, velocity_kmh, Status, week

    NEOs uden close_approach_data springes over.
    """
    records = []
    for chunk in data:
        if 'near_earth_objects' not in chunk:
            continue
        for date, neos in chunk['near_earth_objects'].items():
            for neo in neos:
                if not neo['close_approach_data']:
                    continue
                min_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_min']
                max_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_max']
                records.append({
                    'id':           neo['id'],
                    'name':         neo['name'],
                    'date':         pd.to_datetime(date),
                    'size_km':      (min_dia + max_dia) / 2,
                    'is_hazardous': neo['is_potentially_hazardous_asteroid'],
                    'distance_km':  float(neo['close_approach_data'][0]['miss_distance']['kilometers']),
                    'velocity_kmh': float(neo['close_approach_data'][0]['relative_velocity']['kilometers_per_hour']),
                })

    df = pd.DataFrame(records).drop_duplicates(subset=['id', 'date'])
    df['Status'] = df['is_hazardous'].map({True: 'Hazardous (PHA)', False: 'Harmless'})
    df['week']   = df['date'].dt.to_period('W').apply(lambda p: p.start_time)
    return df


def get_daily_sizes(data):
    """
    Returnerer en dict { 'YYYY-MM-DD': [størrelser_km] } for alle NEOs.
    Bruges til at beregne daglige gennemsnit, median mm.
    """
    daily = {}
    for chunk in data:
        if 'near_earth_objects' not in chunk:
            continue
        for date, neos in chunk['near_earth_objects'].items():
            if date not in daily:
                daily[date] = []
            for neo in neos:
                min_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_min']
                max_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_max']
                daily[date].append((min_dia + max_dia) / 2)
    return daily


def get_all_sizes(data):
    """
    Returnerer en flad liste med gennemsnitsstørrelsen (km) for ALLE NEOs.
    Bruges til statistisk analyse på tværs af hele datasættet.
    """
    sizes = []
    for chunk in data:
        if 'near_earth_objects' not in chunk:
            continue
        for neos in chunk['near_earth_objects'].values():
            for neo in neos:
                min_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_min']
                max_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_max']
                sizes.append((min_dia + max_dia) / 2)
    return sizes


def load_data_from_google_drive(url):
    """
    Loader en CSV-fil direkte fra Google Drive.
    Konverterer den delte link-URL til en download-URL automatisk.
    """
    import pandas as pd
    url_processed = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url_processed)
    return df





import numpy as np

def get_daily_averages(data):
    """
    Processes the raw NASA data chunks and calculates the average size of NEOs for each day.
    Returns a dictionary with dates as keys and average sizes (km) as values.
    """
    # 1. Create an empty dictionary to keep track of dates and their corresponding sizes
    daily_sizes = {}

    # 2. Loop through each week/chunk in your 'data' list
    for chunk in data:
        # The NASA API stores the actual date data under the key 'near_earth_objects'
        if 'near_earth_objects' in chunk:
            for date, neos in chunk['near_earth_objects'].items():
                
                # If the date does not exist in our dictionary yet, create it with an empty list
                if date not in daily_sizes:
                    daily_sizes[date] = []
                    
                # Loop through all asteroids (NEOs) for that specific date
                for neo in neos:
                    # We take the average of the min and max estimated diameter in kilometers
                    min_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_min']
                    max_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_max']
                    avg_neo_size = (min_dia + max_dia) / 2
                    
                    # Store this asteroid's size under the correct date
                    daily_sizes[date].append(avg_neo_size)

    # 3. Calculate the final average (mean) for each day
    daily_averages = {}
    for date, sizes in daily_sizes.items():
        if sizes: # Ensures that there actually is data for the day
            daily_averages[date] = np.mean(sizes)
            
    #Return data
    return daily_averages





def calculate_hazardous_proportion(data):
    """
    Calculates the total number of NEOs, the number of potentially hazardous NEOs,
    and their overall percentage proportion across all data chunks.
    Returns a dictionary with the results.
    """
    # Start counters
    total_neos = 0
    hazardous_neos = 0

    # Loop through the data
    for chunk in data:
        if 'near_earth_objects' in chunk:
            for date, neos in chunk['near_earth_objects'].items():
                for neo in neos:
                    
                    # 1. Add 1 to the total counter for each asteroid we encounter
                    total_neos += 1
                    
                    # 2. Check if NASA has marked it as hazardous
                    if neo['is_potentially_hazardous_asteroid'] == True:
                        hazardous_neos += 1

    # Calculate the proportion in percent if data exists
    if total_neos > 0:
        proportion = (hazardous_neos / total_neos) * 100
    else:
        proportion = 0.0

    return {
        'total_neos': total_neos,
        'hazardous_neos': hazardous_neos,
        'proportion_percent': proportion
    }



def get_closest_neos_per_day(data):
    """
    Finds the closest asteroid to Earth for each specific day in the dataset.
    Returns a dictionary structured with dates as keys, containing the asteroid name and distance.
    """
    # 1. A dictionary to store the closest asteroid for each day
    closest_neos_per_day = {}

    # 2. Loop through the data layer by layer
    for chunk in data:
        if 'near_earth_objects' in chunk:
            for date, neos in chunk['near_earth_objects'].items():
                
                # Starting values for this specific day
                closest_distance = float('inf')  # Set to "infinity" initially so any number will be smaller
                closest_neo_name = None
                
                for neo in neos:
                    # Check if there actually is close_approach_data available
                    if neo['close_approach_data']:
                        # Get the distance in kilometers and convert it to a number (float)
                        distance_str = neo['close_approach_data'][0]['miss_distance']['kilometers']
                        distance = float(distance_str)
                        
                        # If this distance is closer (smaller) than the previous closest, we store it
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_neo_name = neo['name']
                
                # Store the result for the day in our dictionary (if an asteroid was found)
                if closest_neo_name is not None:
                    closest_neos_per_day[date] = {
                        'name': closest_neo_name,
                        'distance_km': closest_distance
                    }
                    
    # Return the dictionary to the main script
    return closest_neos_per_day



import pandas as pd

def calculate_size_statistics(data):
    """
    Extracts all asteroid sizes and calculates key statistical measures:
    Mean, Median, Mode, Standard Deviation, Range, and the 95th Percentile.
    Returns a dictionary containing all computed metrics.
    """
    # 1. Extract ALL asteroid sizes into a flat list
    all_sizes = []

    for chunk in data:
        if 'near_earth_objects' in chunk:
            for date, neos in chunk['near_earth_objects'].items():
                for neo in neos:
                    # We use the average of min and max diameter in km
                    min_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_min']
                    max_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_max']
                    avg_size = (min_dia + max_dia) / 2
                    all_sizes.append(avg_size)

    # 2. Convert the list to a Pandas Series for easy statistics
    df_sizes = pd.Series(all_sizes)

    # Check if we have data to avoid runtime errors
    if df_sizes.empty:
        return {}

    # 3. Calculate the required statistical measures
    mean_val = df_sizes.mean()
    median_val = df_sizes.median()
    std_val = df_sizes.std()
    
    # Mode: most frequent value (returns the first one if all are unique)
    mode_val = df_sizes.mode()[0]

    # --- Two extra statistical methods for deeper analysis ---
    # Extra 1: Range = Max - Min
    min_val = df_sizes.min()
    max_val = df_sizes.max()
    range_val = max_val - min_val

    # Extra 2: 95th Percentile (The top 5% largest asteroids)
    percentile_95 = df_sizes.quantile(0.95)

    # Return all metrics structured in a dictionary
    return {
        'total_count': len(df_sizes),
        'mean': mean_val,
        'median': median_val,
        'mode': mode_val,
        'std_dev': std_val,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'percentile_95': percentile_95
    }



import pandas as pd
from scipy import stats

def analyze_size_hazardous_correlation(data):
    """
    Extracts asteroid sizes and hazardous statuses, then calculates the 
    Point-Biserial correlation coefficient, p-value, and group mean sizes.
    Returns a dictionary containing all statistical results.
    """
    # 1. Extract both size and hazardous status for ALL asteroids
    sizes = []
    is_hazardous = []

    for chunk in data:
        if 'near_earth_objects' in chunk:
            for date, neos in chunk['near_earth_objects'].items():
                for neo in neos:
                    # Size (Average of min and max in km)
                    min_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_min']
                    max_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_max']
                    avg_size = (min_dia + max_dia) / 2
                    
                    # Hazardous (True/False -> converted to 1 or 0 for statistics)
                    hazardous_status = 1 if neo['is_potentially_hazardous_asteroid'] else 0
                    
                    sizes.append(avg_size)
                    is_hazardous.append(hazardous_status)

    # 2. Load data into a Pandas DataFrame
    df_corr = pd.DataFrame({
        'Size_km': sizes,
        'Is_Hazardous': is_hazardous
    })

    # Check if we have data to avoid calculation errors
    if df_corr.empty:
        return {}

    # 3. Calculate Point-Biserial Correlation
    correlation, p_value = stats.pointbiserialr(df_corr['Is_Hazardous'], df_corr['Size_km'])

    # 4. Calculate the average size for the two groups to provide context
    mean_safe = df_corr[df_corr['Is_Hazardous'] == 0]['Size_km'].mean()
    mean_hazardous = df_corr[df_corr['Is_Hazardous'] == 1]['Size_km'].mean()

    # Return results inside a dictionary
    return {
        'correlation': correlation,
        'p_value': p_value,
        'mean_safe': mean_safe,
        'mean_hazardous': mean_hazardous
    }



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def plot_asteroid_size_comparison(data):
    """
    Creates a boxplot comparing the sizes of potentially hazardous asteroids (PHA) 
    against harmless ones, using a clean and designer-friendly approach.
    """
# 1. Prepare the data
    sizes = []
    is_hazardous = []

    for chunk in data:
        if 'near_earth_objects' in chunk:
            for date, neos in chunk['near_earth_objects'].items():
                for neo in neos:
                    min_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_min']
                    max_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_max']
                    sizes.append((min_dia + max_dia) / 2)
                    # Labels translated to English
                    is_hazardous.append('Hazardous (PHA)' if neo['is_potentially_hazardous_asteroid'] else 'Harmless')

    df_plot = pd.DataFrame({'Size (km)': sizes, 'Status': is_hazardous})

    # 2. Design the plot ("Think like a designer" & "Eliminate clutter")
    sns.set_theme(style="white") # Removes the dark grey background and heavy gridlines
    plt.figure(figsize=(9, 5))

    # Strategic color palette (Muted grey vs. Attention-grabbing red)
    color_palette = {'Harmless': '#95a5a6', 'Hazardous (PHA)': '#e74c3c'}

    # Draw the boxplot without outliers for better readability
    ax = sns.boxplot(
        x='Size (km)', 
        y='Status', 
        data=df_plot, 
        palette=color_palette, 
        showfliers=False, 
        width=0.5
    )

    # 3. Clean up the chart ("Eliminate clutter")
    sns.despine(left=True, bottom=True)

    # 4. English titles and labels
    plt.title('Potentially Hazardous Asteroids are Markedly Larger Than Harmless Ones', fontsize=14, fontweight='bold', pad=20, loc='left')
    plt.xlabel('Estimated Diameter (kilometers)', fontsize=11, color='#2c3e50')
    plt.ylabel('') # Removed because 'Harmless' and 'Hazardous' labels are self-explanatory

    plt.tight_layout()
    plt.show()




#________________________________________________________________________________________________________
#Nasa

#Task 3: Data Visualization Part A

# Default farve-konfiguration
BLUE = '#2980b9'


def build_neo_viz_dataframe(data):
    """Træk dato, id, navn og gennemsnitlig størrelse ud for ALLE asteroider.

    Parameters
    ----------
    data : list
        The raw NASA NEO data: a list of chunks, each containing a
        ``near_earth_objects`` dict that maps date strings to lists of NEOs.

    Returns
    -------
    pandas.DataFrame
        Columns: date, id, name, size_km, week. Duplicates (same date + id
        across overlapping chunks) are removed.
    """
    records = []
    for chunk in data:
        if 'near_earth_objects' in chunk:
            for date, neos in chunk['near_earth_objects'].items():
                for neo in neos:
                    min_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_min']
                    max_dia = neo['estimated_diameter']['kilometers']['estimated_diameter_max']
                    avg_size = (min_dia + max_dia) / 2
                    records.append({
                        'date': pd.to_datetime(date),
                        'id': neo['id'],
                        'name': neo['name'],
                        'size_km': avg_size,
                    })

    df_viz = pd.DataFrame(records)

    # Samme dato kan optræde i to chunks (hvor ugerne overlapper) -> fjern dubletter
    df_viz = df_viz.drop_duplicates(subset=['date', 'id'])

    # Uge-kolonne (mandag i ugen) til de ugentlige plots
    df_viz['week'] = df_viz['date'].dt.to_period('W').apply(lambda p: p.start_time)

    return df_viz


def plot_neos_per_week(df_viz, color=BLUE):
    """(a) Line plot: antal NEOs pr. uge."""
    sns.set_theme(style="white")

    weekly_counts = df_viz.groupby('week').size().reset_index(drop=True)
    weekly_counts.index = weekly_counts.index + 1   # uge 1, 2, 3, ...

    plt.figure(figsize=(11, 4))
    sns.lineplot(x=weekly_counts.index, y=weekly_counts.values, color=color, linewidth=2)
    sns.despine()
    plt.title('Number of NEOs per Week', fontsize=14, fontweight='bold', loc='left', pad=15)
    plt.xlabel('Week number'); plt.ylabel('Number of NEOs')
    plt.ylim(bottom=0)
    plt.margins(x=0)
    plt.xticks(weekly_counts.index[::3], fontsize=11)   # hver 3. uge vises
    plt.tight_layout(); plt.show()


def plot_neo_size_distribution(df_viz, color=BLUE):
    """(b) Histogram: distribution af NEO-størrelser."""
    sns.set_theme(style="white")

    plt.figure(figsize=(9, 4))
    sns.histplot(df_viz['size_km'], bins=50, color=color, log_scale=True)  # log: størrelser er meget skæve
    sns.despine()
    plt.title('Distribution of NEO Sizes', fontsize=14, fontweight='bold', loc='left', pad=15)
    plt.xlabel('Estimated Diameter (km, log scale)'); plt.ylabel('Number of NEOs')
    plt.tight_layout(); plt.show()


def plot_avg_neo_size_per_week(df_viz, color=BLUE):
    """(c) Bar plot: gennemsnitlig NEO-størrelse pr. uge."""
    sns.set_theme(style="white")

    weekly_avg = df_viz.groupby('week')['size_km'].mean()

    plt.figure(figsize=(12, 4))
    sns.barplot(x=list(range(len(weekly_avg))), y=weekly_avg.values, color=color)
    sns.despine()
    plt.title('Average NEO Size per Week', fontsize=14, fontweight='bold', loc='left', pad=15)
    plt.xlabel('Week index (1 = first week)'); plt.ylabel('Average Diameter (km)')
    plt.xticks([])  # for mange uger til at vise alle labels
    plt.tight_layout(); plt.show()


def plot_neo_weekday_heatmap(df_viz):
    """(d) Seaborn heatmap: antal NEOs pr. ugedag x uge."""
    sns.set_theme(style="white")

    # Arbejd på en kopi, så df_viz ikke muteres af de ekstra kolonner
    df = df_viz.copy()
    df['weekday'] = pd.Categorical(
        df['date'].dt.day_name(),
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                    'Friday', 'Saturday', 'Sunday'], ordered=True)
    df['iso_week'] = df['date'].dt.isocalendar().week
    pivot = df.pivot_table(index='weekday', columns='iso_week',
                           values='id', aggfunc='count', fill_value=0)

    plt.figure(figsize=(14, 3.2))
    sns.heatmap(pivot, cmap='rocket', cbar_kws={'label': 'Number of NEOs'})
    plt.title('NEOs: Weekday x Week of Year', fontsize=14, fontweight='bold', loc='left', pad=15)
    plt.xlabel('Week of year'); plt.ylabel('')
    plt.tight_layout(); plt.show()


def run_task3_visualizations(data):
    """Convenience wrapper: build the dataframe and draw all four plots.

    Returns the dataframe in case you want to inspect it afterwards.
    """
    df_viz = build_neo_viz_dataframe(data)
    plot_neos_per_week(df_viz)
    plot_neo_size_distribution(df_viz)
    plot_avg_neo_size_per_week(df_viz)
    plot_neo_weekday_heatmap(df_viz)
    return df_viz
