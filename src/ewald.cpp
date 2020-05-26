#include <ntiger/ewald.hpp>
#include <cassert>

using vect_int =
general_vect<int, NDIM>;
static real EW(vect);

constexpr int NBIN = 16;
static std::array<std::array<std::array<real, NBIN + 1>, NBIN + 1>, NBIN + 1> potential;
static std::array<std::array<std::array<vect, NBIN + 1>, NBIN + 1>, NBIN + 1> force;

struct ewald {
	ewald() {
		FILE *fp = fopen("ewald.dat", "rb");
		if (true) {
			printf( "Found ewald.dat\n");
			const int sz = (NBIN + 1) * (NBIN + 1) * (NBIN + 1);
			fread(&potential, sizeof(real), sz, fp);
			fread(&force, sizeof(real), sz * NDIM, fp);
			fclose(fp);
		} else {
			printf("ewald.dat not found\n");
			printf("Initializing Ewald\n");

			const real dx0 = 0.5 / (NBIN - 1);
			for (int dim = 0; dim < NDIM; dim++) {
				force[0][0][0][dim] = 0.0;
			}
			potential[0][0][0] = 2.8372975;
			for (int i = 0; i <= NBIN; i++) {
				for (int j = 0; j <= i; j++) {
					for (int k = 0; k <= j; k++) {
						vect x;
						x[0] = i * dx0;
						x[1] = j * dx0;
						x[2] = k * dx0;
						if (x.dot(x) == 0.0) {
							continue;
						}
						const real dx = 0.01 * dx0;
						for (int dim = 0; dim < NDIM; dim++) {
							vect ym = x;
							vect yp = x;
							yp[dim] += 0.5 * dx;
							ym[dim] -= 0.5 * dx;
							force[i][j][k][dim] = -(EW(yp) - EW(ym)) / dx;
							force[j][k][i][dim] = -(EW(yp) - EW(ym)) / dx;
							force[k][i][j][dim] = -(EW(yp) - EW(ym)) / dx;
							force[j][i][k][dim] = -(EW(yp) - EW(ym)) / dx;
							force[i][k][j][dim] = -(EW(yp) - EW(ym)) / dx;
							force[k][j][i][dim] = -(EW(yp) - EW(ym)) / dx;
						}
						potential[i][j][k] = EW(x);
						potential[j][k][i] = EW(x);
						potential[k][i][j] = EW(x);
						potential[j][i][k] = EW(x);
						potential[i][k][j] = EW(x);
						potential[k][j][i] = EW(x);
					}
				}
			}
			printf("Done initializing Ewald\n");
			fp = fopen("ewald.dat", "wb");
			const int sz = (NBIN + 1) * (NBIN + 1) * (NBIN + 1);
			fwrite(&potential, sizeof(real), sz, fp);
			fwrite(&force, sizeof(real), sz * NDIM, fp);
			fclose(fp);
		}

	}
};

ewald E;

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

void ewald_force_and_pot(vect x, vect &f, real& phi) {
	real sgn = 1.0;
	for (int dim = 0; dim < NDIM; dim++) {
		if (x[dim] > 1.0 || x[dim] < -1.0) {
			printf("Call to ewald_force out of range %i %e\n", dim, x[dim].get());
			abort();
		}
		if (x[dim] > 0.5) {
			x[dim] = (1.0 - x[dim]);
			sgn *= -1.0;
		} else if (x[dim] < -0.5) {
			x[dim] = 1.0 + x[dim];
			sgn *= -1.0;
		}
		if (x[dim] < 0.0) {
			x[dim] *= -1.0;
			sgn *= -1.0;
		}
	}
	general_vect<int, NDIM> I;
	general_vect<real, NDIM> w;
	constexpr real dx = 0.5 / (NBIN - 1);
	for (int dim = 0; dim < NDIM; dim++) {
		I[dim] = (x[dim] / dx).get();
		w[dim] = x[dim] - real(I[dim]) * dx;
	}
	const auto w000 = w[0] * w[1] * w[2];
	const auto w001 = w[0] * w[1] * (1.0 - w[2]);
	const auto w010 = w[0] * (1.0 - w[1]) * w[2];
	const auto w011 = w[0] * (1.0 - w[1]) * (1.0 - w[2]);
	const auto w100 = w[0] * w[1] * w[2];
	const auto w101 = (1.0 - w[0]) * w[1] * (1.0 - w[2]);
	const auto w110 = (1.0 - w[0]) * (1.0 - w[1]) * w[2];
	const auto w111 = (1.0 - w[0]) * (1.0 - w[1]) * (1.0 - w[2]);
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim] = 0.0;
		f[dim] += force[I[0]][I[1]][I[2]][dim] * w000;
		f[dim] += force[I[0]][I[1]][I[2] + 1][dim] * w001;
		f[dim] += force[I[0]][I[1] + 1][I[2]][dim] * w010;
		f[dim] += force[I[0]][I[1] + 1][I[2] + 1][dim] * w011;
		f[dim] += force[I[0] + 1][I[1] + 1][I[2]][dim] * w110;
		f[dim] += force[I[0] + 1][I[1] + 1][I[2] + 1][dim] * w111;
	}
	phi = 0.0;
	phi += potential[I[0]][I[1]][I[2]] * w000;
	phi += potential[I[0]][I[1]][I[2] + 1] * w001;
	phi += potential[I[0]][I[1] + 1][I[2]] * w010;
	phi += potential[I[0]][I[1] + 1][I[2] + 1] * w011;
	phi += potential[I[0] + 1][I[1] + 1][I[2]] * w110;
	phi += potential[I[0] + 1][I[1] + 1][I[2] + 1] * w111;
	const real r = abs(x);
	f = f - x / pow(r, 3);
	phi = phi - 1.0 / r;
}

static real EW(vect x) {
	vect h, n;

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
	return M_PI / 4.0 + sum1 + sum2 + 1 / abs(x);
}

