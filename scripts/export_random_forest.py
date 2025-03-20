from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import treelite.sklearn
import tl2cgen

import json


def main():
    X, y = make_regression(n_features=20, n_samples=100_000, random_state=22)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.01, random_state=22
    )

    model = RandomForestRegressor(
        n_estimators=20, max_depth=10, random_state=22, verbose=2, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    treelite_model = treelite.sklearn.import_model(model)

    with open("data/random_forest/model.json", "w+") as f:
        f.write(treelite_model.dump_as_json())

    with open("test/src/test/resources/random_forest.json", "w+") as f:

        json.dump({"X": X[:100].tolist(), "y_pred": y_pred.tolist()}, f, indent=4)

    tl2cgen.generate_c_code(
        treelite_model, dirpath="data/random_forest/c", params={"parallel_comp": 20}
    )


if __name__ == "__main__":
    main()
