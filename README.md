# Tree Lite 2 Java Generator (tl2jgen)

Like [tl2cgen](https://github.com/dmlc/tl2cgen) ¯\\_(ツ)_/¯, but generates java code.

Supported regression models:

- sklearn random/extreme forest
- xgboost with categorical features :construction:

--- 

### Running examples

```shell
uv venv --python 3.12
uv run scripts/export_random_forest.py

./scripts/generate_random_forest.sh

cd example
./gradlew test
```