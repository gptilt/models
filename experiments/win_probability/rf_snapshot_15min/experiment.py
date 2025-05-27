import experiments as exp
import matplotlib.pyplot as plt
import polars as pl
from sklearn import metrics
from sklearn import ensemble
from sklearn import model_selection


def run(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    parameters: dict
):
    # experiment = exp.start()
    
    X_train = df_train.drop('winningTeam', 'matchId')
    y_train = df_train['winningTeam']

    X_test = df_test.drop('winningTeam', 'matchId')
    y_test = df_test['winningTeam']

    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42
    )

    # Random Forest Model
    model = ensemble.RandomForestClassifier(random_state=42, oob_score=True)

    # Hyperparameter Tuning
    # exp.log_parameters(parameters['parameter_grid'])
    grid_search = model_selection.GridSearchCV(
        model,
        parameters['parameter_grid'], cv=3, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    # exp.log_parameters(grid_search.best_params_)

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
    # exp.log_metric("val_accuracy", val_accuracy)

    # Retrain best model on full train+val set
    best_model.fit(pl.concat([X_train, X_val]), pl.concat([y_train, y_val]))

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    # Log metrics
    # exp.log_metric("oob_score", best_model.oob_score_)
    # exp.log_metric("test_accuracy", metrics.accuracy_score(y_test, y_test_pred))
    # exp.log_metric("test_precision", metrics.precision_score(y_test, y_test_pred))
    # exp.log_metric("test_recall", metrics.recall_score(y_test, y_test_pred))
    # exp.log_metric("test_f1", metrics.f1_score(y_test, y_test_pred))
    # if len(set(y_test)) == 2:
    #     exp.log_metric("test_roc_auc", metrics.roc_auc_score(y_test, y_test_prob))
    conf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
    # exp.log_confusion_matrix(matrix=conf_matrix, labels=[0, 1])

    # Feature importances are biased towards high cardinality.
    # Hence, we use permutation importance.
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_importances_idx = result.importances_mean.argsort()
    importances = (
        pl.DataFrame(
            result.importances[sorted_importances_idx].T,
            columns=X_train.columns[sorted_importances_idx],
            name='features'
        )
        .sort(descending=True)
        .filter(pl.col('features') > 1e-5)
        .to_pandas()
    )

    # Plot Permutation Importances
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    exp.log_figure(figure_name="feature_importances_plot", figure=plt.gcf())
    plt.close()

    # End Comet experiment
    # exp.end()