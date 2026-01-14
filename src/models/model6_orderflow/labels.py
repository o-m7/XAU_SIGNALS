import pandas as pd
import numpy as np

def create_labels(df):
    from src.features_complete import add_triple_barrier_labels
    df_labeled = add_triple_barrier_labels(
        df,
        h_max=30,
        tp_mult=1.5,
        sl_mult=1.0
    )
    return df_labeled
