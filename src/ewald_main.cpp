#include <ntiger/ewald.hpp>
#include <cstdio>

int main() {
	const auto dr = 0.001;
	for (real r = dr / 2.0; r < .5; r += dr) {
		const auto ep = ewald_potential(r + dr * 0.00001);
		const auto em = ewald_potential(r - dr * 0.00001);
//		printf("%e %e\n", r.get(), ((ep - em) / (0.00002 * dr)).get());
	}

}
