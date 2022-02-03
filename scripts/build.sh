#!/bin/bash

rm -fr ./output/*
# rm -fr ./results/*.csv

mkdir -p ./output/
mdkir -p ./results/

now=$(date +"%m-%d-%Y_%H:%M:%S");

gcc \
k_means_compression.c \
./utils/utils.c \
-O2 \
-w \
-lm -lOpenCL \
-fopenmp \
-Wl,-rpath,./ -L./lib/ -l:"libfreeimage.so.3" \
-o dist/k_means;

if [[ $? -ne 0 ]]; then
    exit
fi


img="cube";
colors=(16 32 64 128 256)
iterations=(50)
threads=(2 4 8 6 12);
resolutions=("6016x3384" "5120x2880" "3840x2160" "1920x1080" "1280x720" "900x600" "640x480" "320x240")


for res in ${resolutions[@]}; do
    ./dist/k_means 16 50 12 "img/${img}_${res}.png" "gpu";
done