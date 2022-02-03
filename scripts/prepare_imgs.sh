#!/bin/bash

img_dir=$(pwd)/img

resolutions=("12288x6480" "7680x4320" "6016x3384" "5120x2880" "3840x2160" "1920x1080" "1280x720" "900x600" "640x480" "320x240")

cd img

for file in "${img_dir}"/*; do
    name="$(basename "$file")"
    for res in ${resolutions[@]}; do
        convert $name -resize ${res}\! ${name%.png}_${res}.png
    done
done