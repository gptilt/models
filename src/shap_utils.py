import comet_ml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap

def run(
    model,
    df_X: pd.DataFrame,
    df_y: pd.Series,
    n_of_top_features: int,
    comet_experiment: comet_ml.CometExperiment
):
    """
    Args:
        model (any): Model to explain with SHAP.
        df_X (pd.DataFrame): Model independent variables.
        df_y (pd.Series): Model dependent variable.
        n_of_top_features (int): Number of top features to include in the analysis.
        comet_experiment (comet_ml.CometExperiment): The running experiment.
    """
    print("Running SHAP Explainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_X)

    print("Creating Explanation object...")
    # Extract base values (expected_value) and ensure shape matches.
    # Base values are the values the model would predict with no features.
    base_values = explainer.expected_value
    if isinstance(base_values, list):  # For multiclass or binary with list
        base_values = np.array(base_values[0])

    # Compute absolute mean SHAP values per feature
    shap_abs_mean = np.abs(shap_values).mean(axis=(0, 2))
    df_feature_importance = pd.DataFrame({
        'feature': df_X.columns,
        'importance': shap_abs_mean
    }).sort_values(by='importance', ascending=False)

    for row in df_feature_importance.itertuples():
        comet_experiment.log_metric(f"shap_importance_{row.feature}", row.importance)

    # Select top-k features
    top_features = df_feature_importance['feature'].head(n_of_top_features).tolist()
    top_feature_indices = [df_X.columns.get_loc(f) for f in top_features]

    # Filter SHAP values and data for top features
    shap_values_top = shap_values[:, top_feature_indices, :]
    data_top = df_X.iloc[:, top_feature_indices]

    # Create Explanation with top features
    explanation = shap.Explanation(
        # Computed SHAP values
        values=shap_values_top,
        # Model's base value(s)
        base_values=base_values,
        # Original input data
        data=data_top.values,
        # Feature names
        feature_names=top_features
    )

    print("Plotting...")

    # 2️⃣ Per-Output SHAP Plots
    n_outputs = shap_values.shape[2]
    print(f"There are {n_outputs} possible labels in the dataset.")
    print("Generating beeswarm plots for each...")
    for i in range(n_outputs):
        # Full feature set
        plt.figure()
        shap.plots.beeswarm(
            explanation[:, :, i],
            max_display=n_of_top_features,
            show=False,
        )
        plt.title(f"SHAP Summary - Output {i}")
        plt.subplots_adjust(left=0.3)
        file_path = f"output/shap_summary_output_{i}.png"
        plt.savefig(file_path)
        plt.close()
        comet_experiment.log_image(file_path)
    
    # 3️⃣ Averaged SHAP Plot (Across Outputs)
    print("Plotting SHAP averaged across outputs...")
    explanation_averaged = shap.Explanation(
        values=shap_values_top.mean(axis=2),
        base_values=base_values.mean(),
        data=data_top.values,
        feature_names=top_features
    )
    plt.figure()
    shap.plots.beeswarm(
        explanation_averaged,
        max_display=n_of_top_features,
        show=False,
    )
    plt.title("SHAP Summary (Mean Over Outputs)")
    plt.subplots_adjust(left=0.3)
    file_path = "output/shap_summary_mean.png"
    plt.savefig(file_path)
    plt.close()
    comet_experiment.log_image(file_path)

    # 4️⃣ Bar Plot for Feature Importance
    plt.figure()
    shap.plots.bar(
        explanation_averaged.cohorts(2).abs.mean(0),
        max_display=n_of_top_features,
        clustering=shap.utils.hclust(data_top, df_y),
        clustering_cutoff=0.9,
        show=False
    )
    plt.title("SHAP Importances (Mean Over Outputs)")
    plt.subplots_adjust(left=0.3)
    file_path = "output/shap_importances_bar.png"
    plt.savefig(file_path)
    plt.close()
    comet_experiment.log_image(file_path)
