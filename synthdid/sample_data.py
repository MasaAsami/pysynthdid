import pandas as pd


def fetch_CaliforniaSmoking() -> pd.DataFrame:
    """
    This data is from https://web.stanford.edu/~jhain/synthpage.html
    [Retun]
    pd.DataFrame
    """
    _raw = pd.read_csv("../sample_data/MLAB_data.txt", sep="\t", header=None)

    _raw.columns = [
        "Alabama",
        "Arkansas",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Georgia",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Mexico",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
        "California",
    ]

    _raw.index = [i for i in range(1962, 2001)]

    return _raw.loc[1970:]

