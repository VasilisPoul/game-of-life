#!/bin/sh
rm -f generations/row/*
rm -f generations/boxes/*
python3 scripts/block.py 32 generations/row/input.txt
