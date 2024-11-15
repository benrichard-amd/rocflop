HIPCCFLAGS=
all:

	hipcc $(HIPCCFLAGS) main.cpp -o rocflop
