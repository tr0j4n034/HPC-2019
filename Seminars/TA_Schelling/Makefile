include Makefile.in

all: clean compile run convert

rm_ppm:
	rm -rf *.ppm

rm_png:
	rm -rf *.png

clean: rm_ppm rm_png
	rm -rf *.out *.mp4

compile:
	mpicxx -std=c++11 -O3 schelling.cpp -o schelling.out

run:
	mpirun -n $(PROC) ./schelling.out $(ITER) $(SIZE) $(THRESH) $(FRAC) $(EMPTY)

convert:
	./convert.sh 0 $(ITER)

