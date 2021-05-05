import talib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def manipulate(mode, df, COL, MINMAX=True, MIN=None, MAX=None, SORT=False, PLOT=False, PRINT=False, *args, **kwargs):
    if PRINT:
        print(df[COL])

    # drop all NaN values
    df = df.dropna(how='any', subset=[COL])
    length = len(df)

    if MINMAX:
        # Filter Values outside of MIN and MAX
        if MIN is None and MAX is not None:
            df = df[(df[COL] <= MAX)]
        elif MAX is None and MIN is not None:
            df = df[(df[COL] >= MIN)]
        elif MAX is None and MIN is None:
            df = df
        else:
            df = df[(df[COL] >= MIN) & (df[COL] <= MAX)]
        if PRINT:
            print('MIN MAX:\n', df)

    if SORT:
        # Sort the Dataframe by that Column
        df = df.sort_values(by=[COL], ignore_index=True)
        if PRINT:
            print('Sorted:\n', df[COL])

    if PLOT:
        # plot the ordered column to see the outliers
        print(df[COL], COL)

        plt.title(mode.upper() + " - Trimmed & Sorted")

        plt.ylabel("Solar Power Generated")
        plt.xlabel("Index")

        plt.plot(df[COL], 'go', label=COL)
        plt.legend()
        plt.show()

    removed_rows = length - len(df)
    print(COL, removed_rows)
    return df


def mov_ave(df, col, ave):
    """ Replaces the col with the moving average
    Then removes NaN """
    """ Be carefull, if you loop through the columns and 
    remove the mov ave every time it removes to much"""

    print('df:', len(df))
    df[col] = talib.SMA(df[col], timeperiod=ave)
    df = df.dropna(how='any', subset=[col])
    print('df_:', len(df))
    return df


def insert_shifted_up(df: pd.DataFrame, column):
    new_col_name = f"{column}_Shifted"
    df.insert(0, new_col_name, df[column].shift(-1))
    return df.dropna(), new_col_name


def insert_percentage(df: pd.DataFrame, column):
    df, new_shifted_col_name = insert_shifted_up(df, column)
    scaled_col_name = f"{new_shifted_col_name}_Percentage"
    _max = max(df[new_shifted_col_name])
    _min = min(df[new_shifted_col_name])
    scalar = preprocessing.MinMaxScaler()

    scaled_data = scalar.fit_transform(df[[new_shifted_col_name]])
    scaled_data = scaled_data * 100
    scaled_data = scaled_data.round(0)
    df.insert(0, scaled_col_name, scaled_data)
    df = df.drop(new_shifted_col_name, axis=1)
    return df, scaled_col_name


def insert_sections(df: pd.DataFrame, column):

    df, new_scaled_col_name = insert_percentage(df, column)

    count_of_sections_used = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
    }

    def apply_sections(row):
        for k, pair in sections.items():
            if pair[0] <= row <= pair[1]:
                count_of_sections_used[k] += 1
                return k

    df.insert(0, f"{new_scaled_col_name}_Sections", df[new_scaled_col_name].apply(apply_sections))
    df = df.drop(new_scaled_col_name, axis=1)
    print(count_of_sections_used)
    return df


def run(cols, mode, pred_mode=None, save=False, *args, **kwargs):
    df = pd.read_csv('../combined_data.csv', index_col=0, parse_dates=["date_time"])

    for idx, _min, _max in cols[mode]:
        df = manipulate(mode=mode, df=df, COL=idx, MIN=_min, MAX=_max, *args, **kwargs)

        print(df.iloc[:, :4])

        if pred_mode == 'solar':
            df, _ = insert_shifted_up(df, idx)
        if pred_mode == 'sections':
            df = insert_sections(df, idx)
        if pred_mode == 'percentage':
            df, _ = insert_percentage(df, idx)

        print(df.iloc[:, :3])

    df.drop(['rh_tmn', 'rh_tmx', 'td_tmn', 'td_tmx', 'airt_tmn', 'airt_tmx', 'winds_tmx'], axis=1, inplace=True)
    df = df.dropna()
    if kwargs.get("SS"):
        df = df.to_numpy()
        np_arr = preprocessing.scale(df)

        # scalar = StandardScaler()
        # scalar.fit(df)
        # np_arr = scalar.transform(df)

        df = pd.DataFrame(np_arr, columns=df.columns)
    # print(df)

    if save:
        scale_ext = "-ss" if kwargs.get("SS") else ""
        sort_ext = "-sort" if kwargs.get("SORT") else ""
        df.to_csv(f'../data/{mode}-{pred_mode}{scale_ext}{sort_ext}-{len(df)}.csv', index=None, header=None)


sections = {
    1: (0, 39),
    2: (40, 69),
    3: (70, 84),
    4: (85, 94),
    5: (95, 100)
    # {1: 1553, 2: 624, 3: 128, 4: 117, 5: 7}

    # 20 - not bad
    # 1: (0, 19),
    # 2: (20, 39),
    # 3: (40, 59),
    # 4: (60, 79),
    # 5: (80, 100)
    # {1: 1044, 2: 509, 3: 411, 4: 312, 5: 153}
}

cols = {
    "tight": [
        ("Solar_power", 2000, 65000),
        # ("Meter_value", -80, 20)
    ],
    "cautious": [
        ("Solar_power", 0, None),
        # ("Meter_value", -100, None)
    ],
    "norm": [
        ("Solar_power", None, None),
        # ("Meter_value", None, None)
    ]
}

run(cols, mode="tight", save=False, SORT=True, PLOT=True)
run(cols, mode="cautious", save=False, SORT=True, PLOT=True)
run(cols, mode="norm", save=False, SORT=True, PLOT=True)

SAVE = True
# run(cols, mode="tight", pred_mode='none', save=SAVE, MINMAX=True)
# run(cols, mode="tight", pred_mode='solar', save=SAVE, MINMAX=True)
# run(cols, mode="tight", pred_mode='percentage', save=SAVE, MINMAX=True)
# run(cols, mode="tight", pred_mode='sections', save=SAVE, MINMAX=True)
#
# run(cols, mode="cautious", pred_mode='none', save=SAVE, MINMAX=True)
# run(cols, mode="cautious", pred_mode='solar', save=SAVE, MINMAX=True)
# run(cols, mode="cautious", pred_mode='percentage', save=SAVE, MINMAX=True)
# run(cols, mode="cautious", pred_mode='sections', save=SAVE, MINMAX=True)
#
# run(cols, mode="norm", pred_mode='none', save=SAVE, MINMAX=True)
# run(cols, mode="norm", pred_mode='solar', save=SAVE, MINMAX=True)
# run(cols, mode="norm", pred_mode='percentage', save=SAVE, MINMAX=True)
# run(cols, mode="norm", pred_mode='sections', save=SAVE, MINMAX=True)

