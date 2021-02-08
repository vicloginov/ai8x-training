#!/bin/sh
./train.py --model ai85esc20netv3 --dataset ESC_20 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-esc20_v3-qat8.pth.tar -8 --device MAX78000 "$@"
