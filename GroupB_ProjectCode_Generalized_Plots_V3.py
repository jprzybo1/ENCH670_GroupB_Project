import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import pandas as pd 
    import os
    import numpy as np
    import requests
    from io import StringIO

    # This code takes hourly point source emission data (SO2 lbs/hr) for a set period of time and finds the surrounding NCore pollutant
    # monitoring network sites and ASOS weather stations within a specified radius. The most and least impact NCore sensors are found based
    # on the number of surrounding point sources (neglecting sensors with no point sources in the surrounding radius). If a file 
    # containing data for the localized sites is not found, it is retrieved using API calls and saved. The Gaussian Plume equation can then be solved 
    # for each point source at the location of the NCore sensor to predict the contributions of each point source to the sensor measurements. 


    # Define functions that will be used throughout the code

    # Define haversine function (great circle distance between two point on a sphere)
    def haversine(lat1, lon1, lat2, lon2):
        """
        Function created by gemini with the prompt: "I have data frames with longitude and latitudes 
        of point source locations and SO2 sensors, I need a function to find the SO2 sensor with the most 
        surrounding point sources and the sensor that is the most isolated." 

        Calculates the shortest distance between two points on a sphere ("as-the-crow-flies")
        Determines the great-circle distance between two points using the formula
        d = 2rsin^-1*(sin^2(lat1-lat2/2)+cos(lat1)*cos(lat2)*sin^2((lon1-lon2)/2))^(1/2)

        """
        # Convert decimal degrees to radians 
        lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
        # Haversine formula 
        dlat = lat2 - lat1 
        dlon = lon2 - lon1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    # Define function used to find point sources and weather stations in the region surrounding specified SO2 sensor
    # This will be used to determine all point sources impacting a sensor and to identify weather stations in the region
    def get_nearby_assets(sensor_lat, sensor_lon, asset_df, lat_col, lon_col, radius_km):

        """
        Returns rows from asset_df that are within radius of the sensor
        Uses the Haversine formula to convert lat and lon to distance in km
        sensor_lat and sensor_lon should be the location of the sensor used as the ground truth reference
        lat_col and lon_col come from the asset_df which can be the weather stations or point sources impacting the sensor
        """

        dists = haversine(sensor_lat, sensor_lon, asset_df[lat_col], asset_df[lon_col])
        return asset_df[dists <= radius_km]

    # Define function used to find a specific type of asset closest to another asset of a different type
    # This will be used to select the weather station nearest to a point source
    def get_nearest_asset(sensor_lat, sensor_lon, asset_df, lat_col, lon_col):

        """
        Returns only the single row from asset_df closest to the sensor.
        sensor_lat and sensor_lon should be the location of the point source
        lat_col and lon_col should be from the asset_df 
        """

        dists = haversine(sensor_lat, sensor_lon, asset_df[lat_col], asset_df[lon_col])
        # index the minimum distance
        nearest_idx = dists.idxmin()
        return asset_df.loc[[nearest_idx]] # return a single-row data frame

    # Generalized API call functions
    # Need to specifiy sensor/weather station, start date, and end date
    # That the data may take a significant time to load, even when saved files are found and the API call isn't needed

    # API call for SO2 sensor data 
    # Since SO2 sensors are the basis of this analysis, only two SO2 sensors are used each run (most and least surrounded)
    def fetch_so2_data(sensor_row, start_date, end_date, email, api_key,label):

        """ 
        Downloads SO2 data for a specific AQS ID 
        AQS ID = state code - county code - site code
        AQS ID comes from ncore_sites reference file that must be downloaded prior to running this code
        site code must be 4 digits (zeros added to beginning)
         """

        # retrieve the AQS id from the NCore Sensor reference data frame (must be downloaded to run)
        aqs_id = str(sensor_row['AQS ID']).split('-')
        if len(aqs_id) < 3:
            print(f"Invalid AQS ID format: {sensor_row['AQS ID']}")
            return pd.DataFrame()

        # Pull out the state, county, and site codes from the AQS id
        state, county, site = aqs_id[0], aqs_id[1], aqs_id[2].zfill(4)
        filename = f"so2_data_{label}_{state}_{county}_{site}_{start_date}.csv"

        # Check if data for that site/start date is downloaded
        # If file is found, import .csv to dataframe
        # If file is not found, use API call to retrieve the data and save file
        if os.path.exists(filename):
            print(f"Loading {filename} from local storage...")
            return pd.read_csv(filename)

        print(f"Fetching SO2 data for Site {site}...")

        # API call for SO2 data
        url = "https://aqs.epa.gov/data/api/sampleData/bySite"
        params = {
            "email": email, "key": api_key, "param": "42401",
            "bdate": start_date.replace("-", ""), "edate": end_date.replace("-", ""),
            "state": state, "county": county, "site": site
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'Data' in data and data['Data']:
                df = pd.DataFrame(data['Data'])
                df.to_csv(filename, index=False)
                return df
        except Exception as e:
            print(f"Error fetching SO2: {e}")
        return pd.DataFrame()

    # Weather Station Data (Part 1, single API Call)
    # This function contains the API call used to get weather data from a single station
    # To run this API call for multiple weather stations at one, this function is used in 
    # fetch_weather_data to call all the data needed within the specified radius
    def get_historical_weather(station_id, start_date, end_date):
        """
        Fetches historical METAR data from IEM for a given station and date range
        Dates should be in 'YYYY-MM-DD' format
        Station_id from ASOS_sites reference file that must be downloaded prior to running this code
        """
        # The IEM "ASOS-download" service URL
        base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

        # Parameters for the request
        # Pulls the entire data sheet, will need to be filtered

        params = {
            "station": station_id,
            "data": "all",
            "year1": start_date.split('-')[0],
            "month1": start_date.split('-')[1],
            "day1": start_date.split('-')[2],
            "year2": end_date.split('-')[0],
            "month2": end_date.split('-')[1],
            "day2": end_date.split('-')[2],
            "tz": "Etc/UTC",
            "format": "comma",
            "latlon": "yes"
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:

            # IEM returns data as a CSV-formatted string
            df = pd.read_csv(StringIO(response.text), skiprows=5, na_values=['M'])

            # Time intervals of the weather data are inconsistent and usually much higher time resolution then other data sources
            # use clean_weather_hourly to match time resolution of df to others
            df = clean_weather_hourly(df)
            return df
        else:
            print("Failed to retrieve data")
            return None

    # Define function to correct weather station time resolution to hourly
    # Used within get_historical_weather
    def clean_weather_hourly(df):
        """
        Groups weather data by hour and takes first observation per hour
        First observation used rather than average because some rows are not numerical
        Ensures 'valid' is datetime
        """
        df = df.copy()

        df['valid'] = pd.to_datetime(df['valid'], errors='coerce')
        df = df.dropna(subset=['valid'])

        df = df.set_index('valid')

        hourly_df = df.resample('h').first().reset_index()

        return hourly_df

    # Weather Station Part 2 (API call applied to all weather stations in the radius)
    def fetch_weather_data(station_ids, start_date, end_date, label):

        """ 
        Downloads and combines weather data for a list of ASOS stations (using get_historical_weather)
        Pulls data from all stations within designated radius (identified with get_nearby_assets)
        Resulting data frame can be used to select the station closest to each point source
        Start and end date should be in YYYY-MM-DD format
         """

        # Check if data for that weather station is downloaded
        # If file is found, import .csv to dataframe
        # If file is not found, use API call to retrieve the data and save file
        filename = f"weather_data_source_{label}.csv"
        if os.path.exists(filename):
            print(f"Loading {filename} from local storage...")
            return pd.read_csv(filename)

        combined_dfs = []
        for sid in station_ids:
            clean_sid = str(sid).strip()
            if not clean_sid or clean_sid == 'nan': continue

            print(f"Fetching ASOS data for: {clean_sid}")

            # Use get_historical_weather for each station id 
            df = get_historical_weather(clean_sid, start_date, end_date) 
            if df is not None and not df.empty:
                df['source_station'] = clean_sid
                combined_dfs.append(df)

        if combined_dfs:
            full_df = pd.concat(combined_dfs, ignore_index=True)
            full_df.to_csv(filename, index=False)
            return full_df
        return pd.DataFrame()

    # Generalized combination of api calls for most and least surrounded sensors based on any input point source .csv
    # Input source .csv must be pre-saved to the folder to run the code
    # This data can be retrieved using the API call in the file "emissions_data_demo.py" in the cam-api-examples github
    # repository created by USEPA and saved for use in this code by adding the following lines to the function before return:
            # filtered_df = streamingResponse_df[
            # ['facilityId', 'facilityName', 'stateCode', 'so2Mass','so2Rate']
            # ]
            # filtered_df = filtered_df.dropna(subset=['so2Mass'])
            # print(filtered_df)
            #  num_unique = filtered_df['facilityId'].nunique()
            # print(num_unique)
            # filtered_df.to_csv(f"PointSources-_{beginDate}_{endDate}.csv")

    def run_analysis_pipeline(pointsource_csv, start_date, end_date, radius_km, email, api_key):
        """
        Uses any point source .csv file to find the corresponding most and least impact sensors
        and runs the previously defined API call functions to get the associated data
        Define radius- 25 km for local impact, 50 km for regional impact, 100 km for long range impact
        """
        # load reference excel sheets containing the latitude and longitudes of 
        # point source locations based on ORIS ids, NCORE SO2 sensor locations based on
        # AQS ids, and ASOS weather stations based on nws IDs, and the effective stack height reference (created by gemini)
        # All these files can be found in the github repository ENCH670_groupb_project

        # ORIS facitility ids and coordinates for each point source
        pointsource_codes_df = pd.read_excel('2___Plant_Y2024.xlsx',usecols='c,j,k',skiprows=1)

        # NCore sites (SO2 monitoring) with AQS ids and their coordinates
        ncore_df = pd.read_excel("ncore_sites.xlsx")

        # Abbreviated list of ASOS weather stations with codes and corresponding coordinates
        # This is abbreviated to improve processing time due to the massive quantity of weather stations available
        asos_df = pd.read_excel('ASOS_sites.xlsx')

        # Load the point source .csv containing SO2 emissions in lb/hr 
        # Created using API call file referenced above
        ps_df = pd.read_csv(pointsource_csv)

        # Find coordinates corresponding to each point source based on site codes
        # Freate new data frame containing emissions rates and point source lat/long
        pointsource_total_df = pd.merge(
        ps_df, 
        pointsource_codes_df, 
        left_on='facilityId', 
        right_on='Plant Code', 
        how='inner'
        )
        pointsource_total_df = pointsource_total_df.drop(columns=['Plant Code'])  

        eff_stack_df = pd.read_csv('eff_stack_heights.csv', skiprows=2)
        eff_stack_df = eff_stack_df[['Facility ID', 'Effective Height (m)']]
        eff_stack_df = eff_stack_df.rename(columns={'Effective Height (m)': 'H'})

        # Create separate data frame containing only the coordinates and SO2 lb/hr emissions of each facility
        pointsource_coordinates_df = pointsource_total_df[['facilityId', 'Latitude', 'Longitude','so2Mass']].drop_duplicates()
        #print(f"Columns in merged DF: {pointsource_total_df.columns}")

        # merge stack heights from reference sheet with pointsource_coordinates_df based on facility ids
        pointsource_coordinates_df = pd.merge(
        eff_stack_df,
        pointsource_coordinates_df, 
        left_on='Facility ID',
        right_on = 'facilityId')

        # Clean dfs by forcing coordinates to be numeric and removing NANs
        for df in [ncore_df]:
            df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
            df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
            df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

        for df in [pointsource_coordinates_df]:
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

        # Use previously defined haversine function to determine the most and least impacted sensors
        # based on number of surrounding point sources in a specified radius 
        # Iterate through each SO2 sensor in ncore_df

        # First check of distances for this time frame have been previously calculated and saved 
        # This is an effort to improve run time of the code by avoiding repetitive calculations
        cache_filename = "ncore_sites_with_distances.csv"
        if os.path.exists(cache_filename):
            print(f"Loading pre-calculated distances from {cache_filename} ")
            ncore_df = pd.read_csv(cache_filename)
        else:
            print("calculating distances for all sensors...")
            surrounding_counts = [] # will be increased every time a point source if found within the radius
            min_distances = [] # saves the minimum distance each sensor is to a point source 

            for idx, sensor in ncore_df.iterrows():
            # Calculate distances from this sensor to every point source
                distances = haversine(
                    sensor['LATITUDE'], sensor['LONGITUDE'], 
                    pointsource_coordinates_df['Latitude'], pointsource_coordinates_df['Longitude']
                )
                surrounding_counts.append(np.sum(distances <= radius_km))
                min_distances.append(np.min(distances))

            ncore_df['surrounding_count'] = surrounding_counts
            ncore_df['dist_to_nearest_source'] = min_distances
            ncore_df.to_csv(cache_filename, index=False)


        # identify targets (most and least surrounded SO2 sensor)
        # the most impacted sensor is identifed as the one with the largest surrounding count
        # the least impacted sensor is identified as the one with the smallest surrounding count, > 0
        impacted_sensors = ncore_df[ncore_df['surrounding_count'] > 0]
        targets = {
            "most": ncore_df.loc[ncore_df['surrounding_count'].idxmax()],
            "least": impacted_sensors.loc[impacted_sensors['surrounding_count'].idxmin()]
        }

        results = {}

        # Dictionary unpacker to loop the Gaussian Plume equation through the most and least impacted sensors
        for key, sensor in targets.items():
            #print(f"\n--- processing {key.upper()} surrounded sensor---")

            # Pull so2 sensor data for that key
            so2_data = fetch_so2_data(sensor, start_date, end_date, email, api_key,key)


            # Identify all point sources in the specified radius around the selected sensor (most or least) 
            sources_in_radius = get_nearby_assets( sensor['LATITUDE'],
            sensor['LONGITUDE'], 
            pointsource_coordinates_df, 
            'Latitude', 'Longitude',  
            radius_km
            )


            source_weather_mapping = []

            for idx, source in sources_in_radius.iterrows():

                # Convert hourly emissions from lbs SO2/hr ug SO2/hr
                q_value = source['so2Mass']*126000 

                # print(f"Source {source['facilityId']} has an emission rate of: {q_value}")

                # define stack height as H (merged previously from reference to pointsource_coordinates_df)
                H = source['H']

                # Find the weather station nearest to each point source
                nearest_asos = get_nearest_asset(
                    source['Latitude'], source['Longitude'], 
                    asos_df, ' Latitude', ' Longitude'
                )

                station_id = nearest_asos['Station ID'].iloc[0]

                # Use fetch_weather_data to get the data for the nearest station
                weather_df = fetch_weather_data(
                    [station_id], start_date, end_date, f"source_{source['facilityId']}"
                )
                if weather_df is None or weather_df.empty:

                    continue

                # Important columns in the weather_df for GP inputs are
                # sknt - wind direction in knots
                # drct- - wind direction in degrees
                # skyc1 - cloud cover description (used in atmospheric stability)
                # Convert the windspeed from knots to mps and save to new column
                weather_df['wind_speed_mps'] = weather_df['sknt'] * 0.514444

                # Save wind direction to new column
                weather_df['wind_from'] = weather_df['drct']

                # Use get_plume_coords function to solve for x and y in km 
                plume_vals = weather_df.apply(
                    lambda row: get_plume_coords(
                        source['Latitude'],
                        source['Longitude'],
                        sensor['LATITUDE'],
                        sensor['LONGITUDE'],
                        row['wind_from']
                    ),
                    axis=1
                )

                # Save x and y values to weather__df
                weather_df['x_m'] = plume_vals.apply(lambda v: v[0])
                weather_df['y_m'] = plume_vals.apply(lambda v: v[1])

                # Discard times when x_m < 0 
                # This means the wind is blowing the plume of the point source away from the sensor
                # and should not be accounted for as impacting the sensor
                weather_df = weather_df[weather_df['x_m'] > 0]



                # Define atmospheric stability constants (disance > 1 km)
                # format: 'Class': [a, b, c, d, f]
                STABILITY_COEFFS = {
                    'A': [213, 0.894, 459.7, 2.094, -9.6],
                    'B': [156, 0.894, 108.2, 1.098, 2.0],
                    'C': [104, 0.894, 61.0, 0.911, 0],
                    'D': [68, 0.894, 44.5, 0.516, -13.0],
                    'E': [50.5, 0.894, 55.4, 0.305, -34.0],
                    'F': [34, 0.894, 62.6, 0.180, -48.6]
                }

                # Define function to identify atmospheric stability class
                def assign_coefficients(row):
                    """ 
                    Uses time of day, wind speed, and cloud coverage to identify atmospheric
                    stability classifications
                    """
                    u = row['wind_speed_mps']
                    sky = row['skyc1']

                    # Day time
                    if 7 <= row['valid'].hour <= 18:
                        if u < 2:
                            class_val = 'A'

                        elif u < 5:
                            class_val = 'B'

                        else:
                            class_val = 'C'

                        # Cloud cover stabilizes atmosphere
                        if sky in ['BKN', 'OVC']:
                            class_val = chr(ord(class_val) + 1)

                    # Night time
                    else:

                        if u < 2:
                            class_val = 'F'

                        elif u < 5:
                            class_val = 'E'

                        else:
                            class_val = 'D'

                    coeffs = STABILITY_COEFFS.get(
                        class_val,
                        STABILITY_COEFFS['D']
                    )

                    return pd.Series(
                        coeffs,
                        index=['a', 'b', 'c', 'd', 'f']
                    )

                weather_df['valid'] = pd.to_datetime(weather_df['valid'], errors='coerce')

                weather_df[['a', 'b', 'c', 'd', 'f']] = (
                weather_df.apply(assign_coefficients, axis=1)
                )

                # Calculate dispersion parameters (sigma_y and sigma_z, based on atmospheric stability)
                weather_df['sigma_y'] = (weather_df['a'] * (weather_df['x_m'] ** weather_df['b'])
                )

                weather_df['sigma_z'] = (weather_df['c'] * (weather_df['x_m'] ** weather_df['d']) + weather_df['f']
                )

                # Avoid zero wind speeds
                weather_df['wind_speedsafe'] = (
                    weather_df['wind_speed_mps'].clip(lower=0.5)
                )

                # Gaussian plume equation
                # Broken into 3 terms 
                # Solving at z=0 (ground level)
                term1 = (q_value /(2 * np.pi * weather_df['wind_speedsafe'] * weather_df['sigma_y'] * weather_df['sigma_z'])
                )

                term2 = np.exp(-0.5 *(weather_df['y_m']**2 /weather_df['sigma_y']**2)
                )

                term3 = (np.exp(-0.5 *((0 - H)**2 /weather_df['sigma_z']**2)) + np.exp( -0.5 * ((0 + H)**2 /weather_df['sigma_z']**2))
                )

                weather_df['concentration'] = (term1 * term2 * term3)



                # Combine point source data with corresponding weather data
                source_weather_mapping.append({
                    "facilityId": source['facilityId'],
                    "source_lat": source['Latitude'],
                    "source_lon": source['Longitude'],
                    "Q_SO2": q_value,
                    "nearest_station_id": station_id,
                    "weather_data": weather_df,
                })

            # Store everything for each scenario
            results[key] = {
                "sensor_info": sensor,
                "sensor_so2": so2_data,
                "sources_and_weather": source_weather_mapping 
            }
        return results

    # Define functions to orient plume coordinates from lat and lon to x and y
    def get_plume_coords(source_lat, source_lon, sensor_lat, sensor_lon, wind_dir_from):
        """
        Converts latitude and longitude values to x and y values in km
        Defines the location of the point source as the origin
        Defines x as the downwind direction from the point source
        """
        # Use haversine equation to find the distance between the source and the sensor in meters
        r = haversine(source_lat, source_lon, sensor_lat, sensor_lon) * 1000 

        # Calculate the bearing from point source to sensor (0-360 degrees)
        # Using a simplified atan2 approach for local distances (50 km)
        d_lon = np.radians(sensor_lon - source_lon)
        lat1, lat2 = np.radians(source_lat), np.radians(sensor_lat)

        y_b = np.sin(d_lon) * np.cos(lat2)
        x_b = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
        bearing = (np.degrees(np.arctan2(y_b, x_b)) + 360) % 360

        # Define the plume travel direction based on wind direction
        # Convert 'wind_from' to 'wind_to' 
        wind_to = (wind_dir_from + 180) % 360

        # Find theta as the angle between the plume centerline and the sensor
        # Angle difference (theta) between plume centerline and sensor
        theta = np.radians(bearing - wind_to)

        # Calculate x (downwind) and y (crosswind)
        x = (r * np.cos(theta))/1000
        y = (r * np.sin(theta))/1000

        return x, y


    # Example execution (July 1 2024-July 14 2024)

    # api key inputs
    # API key can be obtained by signing up with email using the following link
    # https://aqs.epa.gov/aqsweb/documents/data_api.html
    EMAIL = "email"
    API_KEY = "API Key" 

    #
    data_package = run_analysis_pipeline(
        "PointSources_2024-07-01_2024-07-14.csv", 
        "2024-07-01", "2024-07-15", 
        50, EMAIL, API_KEY
    )

    # print summary (make sure data_package exists after code runs)
    if data_package is None or len(data_package) ==0:
        print("\n ERROR: data_package is empty! Check run_analysis_pipeline returns.", flush=True)
    else:
        print(f"\n SUCCESS: data_package contains scenarios: {list(data_package.keys())}", flush=True)

    # Check that keys include ['most', 'least']
    print(f"Data Package Keys: {data_package.keys()}") 

    # Inspect the first point source for the 'most' scenario (used to monitor progress as code was created)
    first_source = data_package['most']['sources_and_weather'][0]

    print(f"\n--- GP Equation Inputs for Plant {first_source['facilityId']} ---")
    print(f"Emission Rate (Q): {first_source['Q_SO2']} µg/s")
    #print(f"Weather Data (u, direction) Columns: {first_source['weather_data']}")
    print(f"Sensor SO2 Data (C) Rows: {len(data_package['most']['sensor_so2'])}")





    return data_package, pd


@app.cell
def _(data_package, pd):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from windrose import WindroseAxes

    # Create visualizations to compare predicted and measured concentrations

    for scenario in ['most', 'least']:
        if scenario not in data_package:
            continue

        # Prepare dataframe for sensor SO2 data
        # This will be used as the timeline basis (all data is in local time)
        df_plot = data_package[scenario]['sensor_so2'].copy()
        df_plot['valid'] = pd.to_datetime(
        df_plot['date_local'] + ' ' + df_plot['time_local'],
        errors='coerce')
        df_plot = df_plot.rename(columns={'sample_measurement': 'actual_ppb'})

        # Source columns will contain hourly calculated values 
        # Should be aligned with sensor SO2 timeline
        source_columns = []


        # Extract the 'concentration' calculated for each source every hour
        for source_entry in data_package[scenario]['sources_and_weather']:

            s_id = str(source_entry['facilityId']).replace('.', '_')
            col_name = f"Plant_{s_id}"

            if col_name in df_plot.columns:
                continue

            w_df = source_entry['weather_data'][['valid', 'concentration', 'drct']].copy()
            w_df['valid'] = pd.to_datetime(w_df['valid'])


            # Rename and merge concentration values into data frame
            w_df = w_df.rename(columns={'concentration': col_name})

            df_plot = pd.merge(
                df_plot,
                w_df[['valid', col_name]],
                on='valid',
                how='left'
            )

            source_columns.append(col_name)

        # Fill NaNs with 0 
        # NaNs show up when the plume direction was opposite the sensor (treating this as a negligible impact)
        df_plot[source_columns] = df_plot[source_columns].fillna(0)
        df_plot['modeled_total'] = df_plot[source_columns].sum(axis=1)

        if 'drct' not in df_plot.columns:
            df_plot = df_plot.merge(data_package[scenario]['sources_and_weather'][0]['weather_data'][['valid', 'drct']],
                on='valid',
                how='left'
            )

        # Plot 1
        # Stacked time series showing the predicted SO2 concentration from each sensor as colored bars
        # Line shows the SO2 concentrations at those times
        plt.figure(figsize=(14, 6))
        plt.stackplot(df_plot['valid'], [df_plot[c] for c in source_columns], 
                      labels=source_columns, alpha=0.7)
        plt.plot(df_plot['valid'], df_plot['actual_ppb'], color='black', 
                 linewidth=2, label='Actual Sensor Reading', linestyle='--')

        plt.title(f"Source Attribution Analysis: {scenario.upper()} Surrounded Sensor")
        plt.ylabel("SO2 Concentration (ppb / µg/m³)")
        plt.xlabel("Date/Time")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'timeseries_{scenario}.jpg')
        plt.show()
        plt.close()

        # Plot 2
        # Actual vs modeled pollution roses
        # Shows the intenstiy of SO2 concentration with respect to the wind directions
        # Compare actual and modelled roses
        rose_data = df_plot.dropna(subset=['drct'])

        fig = plt.figure(figsize=(12, 6))
        # Actual Rose
        ax1 = fig.add_subplot(121, projection='windrose')
        ax1.bar(rose_data['drct'], rose_data['actual_ppb'], normed=True, opening=0.8, edgecolor='white')
        ax1.set_title(f"Actual SO2 Distribution ({scenario})")

        # Modeled Rose
        ax2 = fig.add_subplot(122, projection='windrose')
        ax2.bar(rose_data['drct'], rose_data['modeled_total'], normed=True, opening=0.8, edgecolor='white')
        ax2.set_title(f"GP Modeled SO2 Distribution ({scenario})")
        plt.savefig(f'roses_{scenario}.jpg')
        plt.show()
        plt.close()

        # Plot 3
        # Residuals (actual-modeled) vs (modeled)
        plt.figure(figsize=(10, 4))
        residual = df_plot['actual_ppb'] - df_plot['modeled_total']
        sns.scatterplot(x=df_plot['modeled_total'], y=residual, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f"Model Residuals (Actual - Modeled): {scenario.upper()}")
        plt.xlabel("Modeled Concentration")
        plt.ylabel("Error")
        plt.savefig(f'residuals_{scenario}.jpg')
        plt.show()
        plt.close()
    return


if __name__ == "__main__":
    app.run()
