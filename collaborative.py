import numpy as np

def collaborative_score(df):
    """
    Since we do not have users,
    we simulate collaborative signals using ratings + helpfulness
    """

    df['collab_score'] = (
        0.6 * df['rating_norm'] +
        0.4 * df['helpfulness_norm']
    )

    return df
