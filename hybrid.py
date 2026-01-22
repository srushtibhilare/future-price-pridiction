def hybrid_recommendation(df):
    """
    Final business score combining:
    - Content similarity (implicit via filtering)
    - Rating quality
    - Helpfulness trust
    """

    df['final_score'] = (
        0.5 * df['rating_norm'] +
        0.3 * df['helpfulness_norm'] +
        0.2 * df['collab_score']
    )

    return df.sort_values('final_score', ascending=False)
