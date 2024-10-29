HIPCCFLAGS=--save-temps -O2
all:

	hipcc $(HIPCCFLAGS) main.cpp -o rocbench
