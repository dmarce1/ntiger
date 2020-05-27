#include <ntiger/ewald.hpp>
#include <cstdio>

int main() {
	const auto dr = 0.001;
	for (real r = 1.0; r > 1.0e-10; r /= 2.0) {
		vect x(0);
		x[0] = r;
		vect f;
		real phi;
		printf( "%e %.14e\n", r, EW(x));
	}

}
