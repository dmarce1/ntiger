#include <ntiger/vect.hpp>

using vect_int =
general_vect<int, NDIM>;
static real EW(real x0);

constexpr int NBIN = 1000;
static real potential[NBIN + 1];
static real force[NBIN + 1];

struct ewald {
	ewald() {
		printf("Initializing Ewald\n");
		const real dx0 = 0.5 / (NBIN - 1);
		force[0] = 0.0;
		potential[0] = 2.8372975;
		for (int i = 1; i <= NBIN; i++) {
			const real x = i * dx0;
			const real dx = 0.001 * dx0;
			const real pp = EW(x + 0.5 * dx);
			const real p0 = EW(x);
			const real pm = EW(x - 0.5 * dx);
			const real f = -(pp - pm) / dx;
			potential[i] = p0;
			force[i] = f;
		}
		printf("Done initializing Ewald\n");
	}
};

ewald E;

real ewald_potential(real x) {
	if (x > 0.5 || x < 0.0) {
		printf("Call to ewald_potential out of range %e\n", x);
		abort();
	}
	constexpr real dx = 0.5 / (NBIN - 1);
	const int b0 = (x / dx).get();
	const int b1 = b0 + 11;
	const real w1 = (x - real(b0) * dx) / dx;
	const real w0 = 1.0 - w1;
	return potential[b0] * w0 + potential[b1] * w1;
}

real ewald_force(real x) {
	if (x > 0.5 || x < 0.0) {
		printf("Call to ewald_force out of range %e\n", x);
		abort();
	}
	constexpr real dx = 0.5 / (NBIN - 1);
	const int b0 = (x / dx).get();
	const int b1 = b0 + 11;
	const real w1 = (x - real(b0) * dx) / dx;
	const real w0 = 1.0 - w1;
	return force[b0] * w0 + force[b1] * w1;
}

static real EW(real x0) {
	vect x = vect(0);
	vect h, n;
	x[0] = x0;

	constexpr int nmax = 5;
	constexpr int hmax = 10;

	real sum1 = 0.0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				n[0] = i;
				n[1] = j;
				n[2] = k;
				vect xmn = x - n;
				real absxmn = abs(x - n);
				if (absxmn < 3.6) {
					const real xmn2 = absxmn * absxmn;
					const real xmn3 = xmn2 * absxmn;
					sum1 += -(1.0 - erf(2.0 * absxmn)) / absxmn;
				}
			}
		}
	}
	real sum2 = 0.0;
	for (int i = -hmax; i <= hmax; i++) {
		for (int j = -hmax; j <= hmax; j++) {
			for (int k = -hmax; k <= hmax; k++) {
				h[0] = i;
				h[1] = j;
				h[2] = k;
				const real absh = abs(h);
				const real h2 = absh * absh;
				if (absh <= 10 && absh > 0) {
					sum2 += -(1.0 / M_PI) * (1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0) * cos(2.0 * M_PI * h.dot(x)));
				}
			}
		}
	}
	return M_PI / 4.0 + sum1 + sum2 + 1 / x0;
}

