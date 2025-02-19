import numpy as np

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import treelite.sklearn


def main():
    X, y = make_regression(n_features=20, n_samples=100_000, random_state=22)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.01, random_state=22
    )

    model = RandomForestRegressor(
        n_estimators=40, max_depth=10, random_state=22, verbose=2, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    np.save("data/X_test.npy", X_test, allow_pickle=False)
    np.save("data/y_pred.npy", y_pred, allow_pickle=False)

    treelite_model = treelite.sklearn.import_model(model)
    with open("data/model.json", "w+") as f:
        f.write(treelite_model.dump_as_json())


if __name__ == "__main__":
    main()
