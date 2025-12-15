import pandas as pd
import shap
import matplotlib.pyplot as plt


def plot_feature_importance(model, X, top_n=15):
    """
    Global feature importance (Random Forest)
    """
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Features Driving Churn:")
    print(importance_df.head(top_n))

    importance_df.head(top_n).plot(
        x="Feature",
        y="Importance",
        kind="barh",
        figsize=(8, 6),
        title="Top Features Driving Churn"
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def shap_summary(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle both old & new SHAP formats
    if isinstance(shap_values, list):
        # Old format: list of arrays (binary classification)
        shap_vals = shap_values[1]
    else:
        # New format: 3D array (samples, features, classes)
        shap_vals = shap_values[:, :, 1]

    shap.summary_plot(shap_vals, X_sample)

