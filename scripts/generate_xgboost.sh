#!/bin/zsh

cargo run -- \
  --source data/xgboost/model.json \
  --package com.github.aaarrti.tl2jgen.xgboost \
  --destination example/src/main/java
