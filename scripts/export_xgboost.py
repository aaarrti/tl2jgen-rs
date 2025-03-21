import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import treelite
import tl2cgen

import json
import tempfile


def main():

    X, y = make_regression(n_features=3, n_samples=100_000, random_state=22)

    categorical_column = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=[100_000, 1])

    X = np.concatenate([X, categorical_column], axis=-1)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.01, random_state=22
    )

    model = XGBRegressor(
        enable_categorical=True,
        feature_types=["q", "q", "q", "c"],
        n_estimators=20
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    with tempfile.TemporaryDirectory() as d:
        temp_file_name = f"{d}/temp.json"
        model.save_model(temp_file_name)
        treelite_model = treelite.frontend.load_xgboost_model(temp_file_name)

    with open("data/xgboost/model.json", "w+") as f:
        f.write(treelite_model.dump_as_json())

    with open("example/src/test/resources/xgboost.json", "w+") as f:

        json.dump(
            {"X": X_test[:100].tolist(), "y_pred": y_pred[:100].tolist()}, f, indent=4
        )

    tl2cgen.generate_c_code(
        treelite_model, dirpath="data/xgboost/c", params={"parallel_comp": 20}
    )


if __name__ == "__main__":
    main()
