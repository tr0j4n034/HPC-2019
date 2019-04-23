#!/bin/bash

for ((i = $1; i <= $2; ++i))
do
    p=$(printf "%05d" $i)
    convert "dump_"$i".ppm" -quality 0 "it_${p}.png"
done

### old and not worked ### convert "it_"*".png" -quality 0 -delay 300 movie.gif
ffmpeg -framerate 10 -pattern_type glob -i "it_*.png" -c:v libx264 -r 30 \
    movie.mp4

### -framerate 1/2: This sets the framerate to one-half FPS, or 2 seconds per frame.
### -i img%04d.png: This tells ffmpeg to read the files img0000.png though img9999.png.
### -c:v libx264:   Use video codec libx264.
### -r 30:          Set the output framerate to 30 FPS.
###                 Each of the input images will be duplicated to make
###                 the output what you specify here. You can leave this
###                 parameter off, and the output file will be at the input
###                 framerate.
### movie.mp4:      Output filename.
