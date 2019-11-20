import pandas as pd


def export_dataframe_to_h5(df, file):
    with pd.HDFStore(file) as store:
        df.to_hdf(store, 'df', append=False, format='table')
