import pandas as pd
from datetime import datetime


def combine():
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:00")
    solar_data = pd.read_csv("../solar_data.csv", index_col=0, usecols=[1, 2, 3], parse_dates=["TIME"], date_parser=date_parser)
    weather_data = pd.read_csv("../weather_data.csv", index_col=0, parse_dates=["date_time"])

    # only minute 59 to match weather data
    solar_data = solar_data[solar_data.index.minute == 59]

    # remove duplicate hours
    df = solar_data.reset_index()
    df['TIME'] = df['TIME'].apply(lambda x: x.replace(minute=0, second=0))
    df = df.drop_duplicates(subset='TIME')
    # set 59 minutes to match weather data
    df['TIME'] = df['TIME'].apply(lambda x: x.replace(minute=59, second=0))
    solar_data = df.set_index('TIME')

    # removing extra rows in each dataset
    solar_mask = solar_data.index.isin(weather_data.index)
    weather_mask = weather_data.index.isin(solar_data.index)
    solar_result = solar_data.loc[solar_mask]
    weather_result = weather_data.loc[weather_mask]

    final_data: pd.DataFrame = weather_result
    final_data.insert(0, "Solar_power", solar_result["Solar_power"])

    final_data.drop("station_id", axis=1, inplace=True)

    final_data.to_csv("../combined_data.csv")


if __name__ == '__main__':
    # uses ./solar_data.csv and ./weather_data.csv to combine them based on data to combined_data.csv
    combine()
