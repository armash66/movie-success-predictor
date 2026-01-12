import pandas as pd

def counterfactual_analysis(model, base_input_df, feature_columns, genre_features):
    base_prob = model.predict_proba(base_input_df)[0][1]
    results = []

    for g in genre_features:
        if g not in feature_columns:
            continue

        modified = base_input_df.copy()
        modified[g] = 1 - modified[g].iloc[0]

        new_prob = model.predict_proba(modified)[0][1]

        results.append({
            "feature": g,
            "change": new_prob - base_prob,
            "new_probability": new_prob
        })

    return base_prob, pd.DataFrame(results).sort_values("change", ascending=False)
