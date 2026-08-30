import pandas as pd

def preprocess_crime():
    df = pd.read_csv("crime_data.csv")

    violent_types = ['BATTERY', 'ASSAULT', 'ROBBERY', 'HOMICIDE', 'CRIM SEXUAL ASSAULT']
    df = df[df['Primary Type'].isin(violent_types)]

    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Extract hour (VERY IMPORTANT for routines)
    df['Hour'] = pd.to_datetime(df['Date']).dt.hour

    # Normalize coordinates
    df['lat_norm'] = (df['Latitude'] - df['Latitude'].min()) / (df['Latitude'].max() - df['Latitude'].min())
    df['lon_norm'] = (df['Longitude'] - df['Longitude'].min()) / (df['Longitude'].max() - df['Longitude'].min())

    df.to_csv("processed_crime.csv", index=False)
    print("✅ Crime data processed")


def preprocess_census():
    df = pd.read_csv("census_data.csv")

    df['unemployment'] = df['PERCENT AGED 16+ UNEMPLOYED'] / 100
    df['hardship'] = df['HARDSHIP INDEX'] / 100

    df.to_csv("processed_census.csv", index=False)
    print("✅ Census data processed")


if __name__ == "__main__":
    preprocess_crime()
    preprocess_census()