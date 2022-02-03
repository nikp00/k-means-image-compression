#!/bin/bash

rm -fr ./output/*
# rm -fr ./results/*.csv

mkdir -p ./output/
mkdir -p ./results/

now=$(date +"%m-%d-%Y_%H:%M:%S")

module load CUDA

gcc \
k_means_compression.c \
./utils/utils.c \
-O2 \
-w \
-lm -lOpenCL \
-fopenmp \
-Wl,-rpath,./ -L./lib/ -l:"libfreeimage.so.3" \
-o dist/k_means

if [[ $? -ne 0 ]]; then
    exit
fi

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/
export LD_LIBRARY_PATH

img="cube";
colors=(16 32 64 128 256)
iterations=(50)
threads=(2 4 8 16 32 64);
resolutions=("6016x3384" "5120x2880" "3840x2160" "1920x1080" "1280x720" "900x600" "640x480" "320x240")



echo "Starting CPU benchmark...";
echo "k,n,width,height,time" >> ./results/cpu.csv
for res in ${resolutions[@]}; do
    for k in ${colors[@]}; do
        for i in ${iterations[@]}; do
            echo "Compressing ${img}_${res}.png  |  K: ${k}  |  Iter: ${i}"
            srun -n1 --reservation=fri ./dist/k_means ${k} ${i} 0 "img/${img}_${res}.png" "cpu" "benchmark" 2>&1 | tee -a "./results/cpu.csv"
            srun -n1 --reservation=fri ./dist/k_means ${k} ${i} 0 "img/${img}_${res}.png" "cpu" "benchmark" 2>&1 | tee -a "./results/cpu.csv"
        done
    done
done

sed -i '/srun.*/d' ./results/cpu.csv

echo "Starting CPU parallel benchmark...";
echo "k,n,t,width,height,time" >> ./results/cpup.csv
for res in ${resolutions[@]}; do
    for t in ${threads[@]}; do
        for k in ${colors[@]}; do
            for i in ${iterations[@]}; do
                echo "Compressing ${img}_${res}.png  |  T: ${i}  |  K: ${k}  |  Iter: ${i}";
                srun -n1 --cpus-per-task=64 --reservation=fri ./dist/k_means $k $i $t "img/${img}_${res}.png" "cpup" "benchmark" 2>&1 | tee -a "./results/cpup.csv";
                srun -n1 --cpus-per-task=64 --reservation=fri ./dist/k_means $k $i $t "img/${img}_${res}.png" "cpup" "benchmark" 2>&1 | tee -a "./results/cpup.csv";
            done
        done
    done
done

sed -i '/srun.*/d' ./results/cpup.csv

echo "Starting CPU benchmark...";
echo "k,n,width,height,time" >> ./results/gpu.csv
for res in ${resolutions[@]}; do
    for k in ${colors[@]}; do
        for i in ${iterations[@]}; do
            echo "Compressing ${img}_${res}.png  |  K: ${k}  |  Iter: ${i}";
            srun -n1 -G1 --reservation=fri ./dist/k_means $k $i 0 "img/${img}_${res}.png" "gpu" "benchmark" 2>&1 | tee -a "./results/gpu.csv";
            srun -n1 -G1 --reservation=fri ./dist/k_means $k $i 0 "img/${img}_${res}.png" "gpu" "benchmark" 2>&1 | tee -a "./results/gpu.csv";
        done
    done
done

sed -i '/srun.*/d' ./results/gpu.csv