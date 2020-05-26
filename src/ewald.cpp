#include <ntiger/ewald.hpp>

using vect_int =
general_vect<int, NDIM>;
static real EW(real x0);

constexpr int NBIN = 10000;
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

real ewald_potential(vect x) {
	for (int dim = 0; dim < NDIM; dim++) {
		if (x[dim] > 1.0 || x[dim] < -1.0) {
			printf("Call to ewald_potential out of range %i %e\n", dim, x[dim].get());
			abort();
		}
		if (x[dim] > 0.5) {
			x[dim] = -(1.0 - x[dim]);
		} else if( x[dim] < -0.5) {
			x[dim] = -(1.0 + x[dim]);
		}
	}
	const real r = abs(x);
	constexpr real dx = 0.5 / (NBIN - 1);
	const int b0 = (r / dx).get();
	const int b1 = b0 + 11;
	const real w1 = (r - real(b0) * dx) / dx;
	const real w0 = 1.0 - w1;
	return (-1.0 / r) + potential[b0] * w0 + potential[b1] * w1;
}

real ewald_separation(vect x) {
	real d = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		const real d1 = abs(x[dim]);
		const real d2 = min(d1, abs(x[dim] + 1.0));
		const real d3 = min(d2, abs(x[dim] - 1.0));
		d += d3 * d3;
	}
	return sqrt(d);
}

vect ewald_location(vect x) {
	vect y;
	for (int dim = 0; dim < NDIM; dim++) {
		while (x[dim] < -0.5) {
			x[dim] += 1.0;
		}
		while (x[dim] > +0.5) {
			x[dim] -= 1.0;
		}
	}
	return x;
}

vect ewald_force(vect x) {
	for (int dim = 0; dim < NDIM; dim++) {
		if (x[dim] > 1.0 || x[dim] < -1.0) {
			printf("Call to ewald_force out of range %i %e\n", dim, x[dim].get());
			abort();
		}
		if (x[dim] > 0.5) {
			x[dim] = -(1.0 - x[dim]);
		} else if( x[dim] < -0.5) {
			x[dim] = -(1.0 + x[dim]);
		}
	}
	const auto r = abs(x);
	constexpr real dx = 0.5 / (NBIN - 1);
	const int b0 = (r / dx).get();
	const int b1 = b0 + 11;
	const real w1 = (r - real(b0) * dx) / dx;
	const real w0 = 1.0 - w1;
	real fc = (force[b0] * w0 + force[b1] * w1);
//	printf( "!!!%e %e\n", fc, r);
	return -x / pow(r, 3) + x * fc / r;
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

