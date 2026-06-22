# Imputation script to add for missing data
import pickle 
from geopy.geocoders import Nominatim
from ratelimiter import RateLimiter
import pandas as pd

# County imputation --> use lat/long and geospandas to fill in missing county data
  
def countyImpute(df, pkl_filepth=None):
    """
    Fills missing county data via reverse geocoding. Loads from pickle if available
    and fully converted; otherwise runs the geocoding and saves the result.
 
    Parameters
    df          : DataFrame with 'gm_gis_county', 'latitude', 'longitude' columns
    pkl_filepth : optional path to a pickled version of the converted DataFrame
    """
    print("YEARS SINCE 2017 IN PER COUNTY: ", df['years_since_2016'].head())

    if pkl_filepth: 
        with open(pkl_filepth, 'rb') as file:
            df = pickle.load(file)
            unknown_countyAFTER = df[df['gm_gis_county']=='unknown']
            no_countyAFTER = df[df['gm_gis_county']=='NO COUNTY FOUND']
            # Check that there are no unknowns or no county left.
            if len(unknown_countyAFTER)==0 and len(no_countyAFTER)==0:
                return df
            else:
                print("There are still unknowns or no counties left in df after conversion.")

    # Counties were not converted. Execute county convert and pickle the final df.
    no_county = df[df['gm_gis_county']=='NO COUNTY FOUND']
    unknown_county  = df[df['gm_gis_county']=='unknown']
    null_county = pd.concat([no_county, unknown_county], axis=0)
    null_county = null_county[['latitude', 'longitude']]

    # Drop San Benito because only 1 data sample
    rural_counties = ['SAN BENITO'] # Do not drop the rural counties.
    # Get county for the unknown/no county rows.
    geolocator = Nominatim(user_agent='my_county_finder_app')
    geocode_reverse = RateLimiter(geolocator.reverse, min_delay_seconds=2)

    print("LATITUDE AND LONG: ", null_county[['latitude', 'longitude']])
    print("Columns: ", null_county.columns)

    print("NO CONVERTED DF WAS FOUND — running reverse geocoding")
    for row in null_county.itertuples():
        lat = row.latitude
        long = row.longitude
        try:
            location = geocode_reverse((lat, long))
        except Exception as e:
            print(f"  Geocoding failed for ({lat}, {long}): {e} — skipping row")
            continue

        if not location:
            print(f"  No location returned for ({lat}, {long})")
            continue

        address = location.raw.get('address', {})
        # Nominatim uses 'county' for most CA counties but falls back to
        # 'city', 'town', or 'municipality' for some unincorporated areas.
        county_name = (
            address.get('county')
            or address.get('city')
            or address.get('town')
            or address.get('municipality')
            or address.get('state_district')
        )

        if county_name:
            # Normalize to uppercase to match the rest of your county values.
            df.loc[row.Index, 'gm_gis_county'] = county_name.upper().replace(' COUNTY', '').strip()
            print(f"  ({lat}, {long}) → {df.loc[row.Index, 'gm_gis_county']}")
        else:
            print(f"  No county field in address for ({lat}, {long}). Full: {location.address}")
            # Leave as-is — don't overwrite with None

    remaining_unknowns = df[df['gm_gis_county'].isin(['unknown', 'NO COUNTY FOUND'])]
    if len(remaining_unknowns) == 0:
        save_path = pkl_filepth if pkl_filepth else 'convertedCounties.pkl'
        df.to_pickle(save_path)
        print(f"Pickled converted counties to {save_path}")
        return df
    else:
        print(f"Still {len(remaining_unknowns)} rows unconverted after geocoding:")
        print(remaining_unknowns[['latitude', 'longitude', 'gm_gis_county']])
        # Return df anyway — don't crash the pipeline over a few unconverted rows
        return df