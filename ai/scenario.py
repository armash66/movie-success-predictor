import pandas as pd

def genre_counterfactual_analysis(model, input_df, genres):
    """
    Tests adding each genre and measures change in success probability.
    """

    base_prob = model.predict_proba(input_df)[0][1]
    results = []

    for genre in genres:
        if input_df.loc[0, genre] == 0:
            modified_df = input_df.copy()
            modified_df.loc[0, genre] = 1

            new_prob = model.predict_proba(modified_df)[0][1]
            delta = new_prob - base_prob

            results.append({
                "Genre": genre,
                "New Probability": new_prob,
                "Change": delta
            })

    return (
        base_prob,
        pd.DataFrame(results)
        .sort_values("Change", ascending=False)
    )
