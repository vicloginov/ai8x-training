#!/bin/sh
./train.py --epochs 500 --optimizer Adam --lr 0.001 --deterministic --compress schedule_eme30.yaml --model ai85esc20netv3 --dataset EME30 --confusion --device MAX78000 "$@"
