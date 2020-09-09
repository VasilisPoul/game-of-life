#!/bin/sh
rm -f generations/row/*
rm -f generations/boxes/*
python3 scripts/block.py 960 generations/row/input.txt
