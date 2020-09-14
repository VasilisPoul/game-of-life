#!/bin/sh

# shellcheck disable=SC2045
for file in $(ls generations/row/)
do
  python3 scripts/boxes.py 32 generations/row/$file generations/boxes/$file
done
