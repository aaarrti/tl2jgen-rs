from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

import treelite.sklearn


def main():
    X, y = make_regression(n_features=20, n_samples=10_000, random_state=22)
    model = RandomForestRegressor(
        n_estimators=20, max_depth=20, random_state=22, verbose=2
    )
    model.fit(X, y)
    treelite_model = treelite.sklearn.import_model(model)
    with open("data/model.json", "w+") as f:
        f.write(treelite_model.dump_as_json())


if __name__ == "__main__":
    main()
