import datasets as ds
from experiment import run
import polars as pl


def preprocessing(
    df: pl.DataFrame
):
    list_of_columns_to_explode = [
        f"{prefix}_{i}"
        for i in range(1, 11)
        for prefix in [
            "inventoryIds",
            "inventoryCounts"
        ]
    ]

    df_exploded = df.with_columns([
        pl.col(col).list.get(j).alias(f"{col}_{j}")
        for col in list_of_columns_to_explode
        for j in range(8)
    ]).drop(list_of_columns_to_explode)

    return df_exploded.to_dummies(["platformId", "gameVersion", "region"])


def main():
    dict_of_parameters = {
        'parameter_grid': {
            'n_estimators': [300, 400, 500, 600],
            'max_depth': [25, 50, 100, None],
            'min_samples_split': [2, 5, 10]
        },
        'best_params': {
            'n_estimators': 400,
            'max_depth': 50,
            'min_samples_split': 2,
        },
        'flag_optimize_hyperparameters': False,
        'feature_importance_n': 30
    }

    repo_id = "gptilt/lol-ultimate-snapshot-challenger-15min"
    dataset = ds.load_dataset(repo_id, "snapshot")

    df_train = pl.concat([
        split.to_polars()
        for split_name, split in dataset.items()
        if split_name.startswith('train')
    ])
    df_test = pl.concat([
        split.to_polars()
        for split_name, split in dataset.items()
        if split_name.startswith('test')
    ])

    run(preprocessing(df_train), preprocessing(df_test), dict_of_parameters)

if __name__ == "__main__":
    main()