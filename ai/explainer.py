def generate_insight(shap_df, probability):
    """
    Converts SHAP feature impacts into a human-readable explanation.
    """

    top_features = shap_df.head(5)

    positive = top_features[top_features["Impact"] > 0]["Feature"].tolist()
    negative = top_features[top_features["Impact"] < 0]["Feature"].tolist()

    insight_parts = []

    if probability >= 0.6:
        insight_parts.append("The model predicts a strong chance of success.")
    elif probability >= 0.4:
        insight_parts.append("The model predicts a moderate chance of success.")
    else:
        insight_parts.append("The model predicts a low chance of success.")

    if positive:
        insight_parts.append(
            "Key positive contributors include: " + ", ".join(positive) + "."
        )

    if negative:
        insight_parts.append(
            "Factors reducing the success probability include: " + ", ".join(negative) + "."
        )

    return " ".join(insight_parts)
