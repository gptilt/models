import experiments
import matplotlib.pyplot as plt
import polars as pl
from sklearn import calibration
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
import shap_utils


def run(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    parameters: dict
):
    exp = experiments.start()
    
    print("Creating train/val/test splits...")
    X_train = df_train.drop('winningTeam', 'matchId')
    y_train = (df_train['winningTeam'] / 100 - 1).cast(pl.Int8)

    X_test = df_test.drop('winningTeam', 'matchId')
    y_test = (df_test['winningTeam'] / 100 - 1).cast(pl.Int8)

    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42
    )
    exp.log_metric("size_train", len(X_train))
    exp.log_metric("size_val", len(X_val))
    exp.log_metric("size_test", len(X_test))

    # Random Forest Model
    model = ensemble.RandomForestClassifier(
        random_state=42,
        oob_score=True,
        **parameters['best_params'],
        n_jobs=-1
    )

    print("Conducting grid search...")
    # Hyperparameter Tuning
    if parameters['flag_optimize_hyperparameters']:
        exp.log_parameters(parameters['parameter_grid'])
        grid_search = model_selection.GridSearchCV(
            model,
            parameters['parameter_grid'],
            cv=3,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Best model
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    exp.log_parameters(best_model.get_params())

    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_val_pred = best_model.predict(X_val)
    val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
    exp.log_metric("val_accuracy", val_accuracy)

    # Retrain best model on full train+val set
    print("Retraining on full train+val set...")
    # best_model.fit(pl.concat([X_train, X_val]), pl.concat([y_train, y_val]))

    # Evaluate on test set
    print("Evaluating on test set...")
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    # Log metrics
    exp.log_metric("oob_score", best_model.oob_score_)
    exp.log_metric("test_accuracy", metrics.accuracy_score(y_test, y_test_pred))
    exp.log_metric("test_precision", metrics.precision_score(y_test, y_test_pred))
    exp.log_metric("test_recall", metrics.recall_score(y_test, y_test_pred))
    exp.log_metric("test_f1", metrics.f1_score(y_test, y_test_pred))
    if len(set(y_test)) == 2:
        exp.log_metric("test_roc_auc", metrics.roc_auc_score(y_test, y_test_prob))
    exp.log_confusion_matrix(
        y_true=y_test,
        y_predicted=y_test_pred,
        labels=[0, 1]
    )
    # Calibration metrics
    exp.log_metric("test_brier_score_loss", metrics.brier_score_loss(y_test, y_test_prob))
    fraction_of_positives, mean_predicted_value = calibration.calibration_curve(y_test, y_test_prob, n_bins=10)
    # x-axis: mean predicted probabilities
    # y-axis: fraction of positives
    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (After Calibration)")
    plt.legend()
    plt.grid()
    file_path = "output/calibration_curve.png"
    plt.savefig(file_path)
    plt.close()
    exp.log_image(file_path)

    # 1️⃣ SHAP Importance
    shap_utils.run(
        best_model,
        X_test.to_pandas(),
        y_test.to_pandas(),
        parameters['feature_importance_n'],
        comet_experiment=exp
    )
    # End Comet experiment
    exp.end()
