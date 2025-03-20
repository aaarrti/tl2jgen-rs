#!/bin/zsh

cargo run -- \
  --source data/random_forest/model.json \
  --destination example/src/main/java \
  --package com.github.aaarrti.tl2jgen.randomforest