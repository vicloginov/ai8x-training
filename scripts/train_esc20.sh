#!/bin/sh
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule_esc20.yaml --model ai85kws20netv3 --dataset ESC_20 --confusion --device MAX78000 "$@"
