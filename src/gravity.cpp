#include <ntiger/gravity.hpp>
#include <ntiger/gravity_cuda.hpp>
#include <ntiger/options.hpp>

#include <cassert>

using vect_int =
general_vect<int, NDIM>;
real EW(general_vect<double, NDIM>);

static ewald_table_t potential;
static ewald_table_t force;

void init_ewald() {
	FILE *fp = fopen("ewald.dat", "rb");
	if (fp) {
		int cnt = 0;
		printf("Found ewald.dat\n");
		const int sz = (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1);
		cnt += fread(&potential, sizeof(real), sz, fp);
		cnt += fread(&force, sizeof(real), sz, fp);
		int expected = sz * 2;
		if (cnt != expected) {
			printf("ewald.dat is corrupt, read %i bytes, expected %i. Remove and re-run\n", cnt, expected);
			abort();
		}
		fclose(fp);
	} else {
		printf("ewald.dat not found\n");
		printf("Initializing Ewald (this may take some time)\n");

		const double dx0 = 0.5 / EWALD_NBIN;
		force[0][0][0] = 0.0;
		potential[0][0][0] = 2.8372975;
		real n = 0;
		for (int i = 0; i <= EWALD_NBIN; i++) {
			for (int j = 0; j <= i; j++) {
				printf("%% %.2f complete\r", 2.0 * n.get() / double(EWALD_NBIN + 2) / double(EWALD_NBIN + 1) * 100.0);
				n += 1.0;
				fflush (stdout);
				for (int k = 0; k <= j; k++) {
					general_vect<double, NDIM> x;
					x[0] = i * dx0;
					x[1] = j * dx0;
					x[2] = k * dx0;
					if (x.dot(x) == 0.0) {
						continue;
					}
					const double dx = 0.25 * dx0;
					const auto n = x / abs(x);
					const auto ym = x - n * dx * 0.5;
					const auto yp = x + n * dx * 0.5;
					const auto f = -(EW(yp) - EW(ym)) / dx;
					const auto p = EW(x);
					force[i][j][k] = f;
					force[i][k][j] = f;
					force[j][i][k] = f;
					force[j][k][i] = f;
					force[k][i][j] = f;
					force[k][j][i] = f;
					potential[i][j][k] = p;
					potential[i][k][j] = p;
					potential[j][i][k] = p;
					potential[j][k][i] = p;
					potential[k][i][j] = p;
					potential[k][j][i] = p;
				}
			}
		}
		printf("\nDone initializing Ewald\n");
		fp = fopen("ewald.dat", "wb");
		const int sz = (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1);
		fwrite(&potential, sizeof(real), sz, fp);
		fwrite(&force, sizeof(real), sz, fp);
		fclose(fp);

	}
	if (options::get().cuda) {
		printf("Loading Ewald table to CUDA\n");
		set_cuda_ewald_tables(force, potential);
	}

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

void ewald_force_and_pot(vect x, vect &f, real &phi, real h) {
	const auto x0 = x;
	vect sgn(1.0);
	for (int dim = 0; dim < NDIM; dim++) {
		if (x[dim] < 0.0) {
			x[dim] = -x[dim];
			sgn[dim] *= -1.0;
		}
		if (x[dim] > 0.5) {
			x[dim] = 1.0 - x[dim];
			sgn[dim] *= -1.0;
		}
	}
	const real r = abs(x);
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim] = 0.0;
	}
	phi = 0.0;
	// Skip ewald
	real fmag = 0.0;
	if (r > 1.0e-4) {
		general_vect<int, NDIM> I;
		general_vect<real, NDIM> w;
		constexpr real dx = 0.5 / EWALD_NBIN;
		for (int dim = 0; dim < NDIM; dim++) {
			I[dim] = std::min(int((x[dim] / dx).get()), EWALD_NBIN - 1);
			w[dim] = 1.0 - (x[dim] / dx - real(I[dim]));
		}
		const auto w000 = w[0] * w[1] * w[2];
		const auto w001 = w[0] * w[1] * (1.0 - w[2]);
		const auto w010 = w[0] * (1.0 - w[1]) * w[2];
		const auto w011 = w[0] * (1.0 - w[1]) * (1.0 - w[2]);
		const auto w100 = (1.0 - w[0]) * w[1] * w[1] * w[2];
		const auto w101 = (1.0 - w[0]) * w[1] * (1.0 - w[2]);
		const auto w110 = (1.0 - w[0]) * (1.0 - w[1]) * w[2];
		const auto w111 = (1.0 - w[0]) * (1.0 - w[1]) * (1.0 - w[2]);
		fmag += force[I[0]][I[1]][I[2]] * w000;
		fmag += force[I[0]][I[1]][I[2] + 1] * w001;
		fmag += force[I[0]][I[1] + 1][I[2]] * w010;
		fmag += force[I[0]][I[1] + 1][I[2] + 1] * w011;
		fmag += force[I[0] + 1][I[1]][I[2]] * w100;
		fmag += force[I[0] + 1][I[1]][I[2] + 1] * w101;
		fmag += force[I[0] + 1][I[1] + 1][I[2]] * w110;
		fmag += force[I[0] + 1][I[1] + 1][I[2] + 1] * w111;
		f = x * (fmag / r);
		phi += potential[I[0]][I[1]][I[2]] * w000;
		phi += potential[I[0]][I[1]][I[2] + 1] * w001;
		phi += potential[I[0]][I[1] + 1][I[2]] * w010;
		phi += potential[I[0]][I[1] + 1][I[2] + 1] * w011;
		phi += potential[I[0] + 1][I[1]][I[2]] * w100;
		phi += potential[I[0] + 1][I[1]][I[2] + 1] * w101;
		phi += potential[I[0] + 1][I[1] + 1][I[2]] * w110;
		phi += potential[I[0] + 1][I[1] + 1][I[2] + 1] * w111;
	} else {
		phi = 2.8372975;
	}
	const real r3 = r * r * r;
	if (r > h) {
		f = f - x / r3;
	} else {
		f = f - x / (h * h * h);
	}
	for (int dim = 0; dim < NDIM; dim++) {
		f[dim] *= sgn[dim];
	}
	if (r > h) {
		phi = phi - 1.0 / r;
	} else {
		phi = phi - (1.5 * h * h - 0.5 * r * r) / (h * h * h);
	}
}

real EW(general_vect<double, NDIM> x) {
	general_vect<double, NDIM> n, h;
	constexpr int nmax = 5;
	constexpr int hmax = 10;

	double sum1 = 0.0;
	for (int i = -nmax; i <= nmax; i++) {
		for (int j = -nmax; j <= nmax; j++) {
			for (int k = -nmax; k <= nmax; k++) {
				n[0] = i;
				n[1] = j;
				n[2] = k;
				const auto xmn = x - n;
				double absxmn = abs(x - n);
				if (absxmn < 3.6) {
					const double xmn2 = absxmn * absxmn;
					const double xmn3 = xmn2 * absxmn;
					sum1 += -(1.0 - erf(2.0 * absxmn)) / absxmn;
				}
			}
		}
	}
	double sum2 = 0.0;
	for (int i = -hmax; i <= hmax; i++) {
		for (int j = -hmax; j <= hmax; j++) {
			for (int k = -hmax; k <= hmax; k++) {
				h[0] = i;
				h[1] = j;
				h[2] = k;
				const double absh = abs(h);
				const double h2 = absh * absh;
				if (absh <= 10 && absh > 0) {
					sum2 += -(1.0 / M_PI) * (1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0) * cos(2.0 * M_PI * h.dot(x)));
				}
			}
		}
	}
	return M_PI / 4.0 + sum1 + sum2 + 1 / abs(x);
}

std::vector<gravity> gravity_near_cpu(const std::vector<vect> &x, const std::vector<vect> &y) {
	static const bool ewald = options::get().ewald;
	static const real h = options::get().kernel_size;
	static const real h2 = h * h;
	static const real hinv = 1.0 / h;
	static const real h3inv = hinv * hinv * hinv;
	static const real m = 1.0 / options::get().problem_size;
	const int cnt = x.size();
	std::vector<gravity> g(cnt);

	for (int i = 0; i < cnt; i++) {
		g[i].g = vect(0);
		g[i].phi = 0.0;
		for (int j = 0; j < y.size(); j++) {
			vect f;
			real phi;
			const auto dx = x[i] - y[j];
			if (ewald) {
				ewald_force_and_pot(dx, f, phi, h);
			} else {
				const auto r = abs(dx);
				if (r > h) {
					const auto rinv = 1.0 / r;
					const auto r3inv = rinv * rinv * rinv;
					f = -dx * r3inv;
					phi = -rinv;
				} else {
					f = -dx * h3inv;
					phi = -(1.5 * h2 - 0.5 * r * r) * h3inv;
				}
			}
			g[i].g = g[i].g + f;
			g[i].phi += phi;
		}
		g[i].g = g[i].g * m;
		g[i].phi *= m;
	}
	return g;
}

std::vector<gravity> gravity_far_cpu(const std::vector<vect> &x, const std::vector<source> &y) {
	static const bool ewald = options::get().ewald;
	static const real h = options::get().kernel_size;
	static const real h2 = h * h;
	static const real hinv = 1.0 / h;
	static const real h3inv = hinv * hinv * hinv;
	const int cnt = x.size();
	std::vector<gravity> g(cnt);

	for (int i = 0; i < cnt; i++) {
		g[i].g = vect(0);
		g[i].phi = 0.0;
		for (int j = 0; j < y.size(); j++) {
			vect f;
			real phi;
			const auto dx = x[i] - y[j].x;
			if (ewald) {
				ewald_force_and_pot(dx, f, phi, h);
			} else {
				const auto r = abs(dx);
				if (r > h) {
					const auto rinv = 1.0 / r;
					const auto r3inv = rinv * rinv * rinv;
					f = -dx * r3inv;
					phi = -rinv;
				} else {
					f = -dx * h3inv;
					phi = -(1.5 * h2 - 0.5 * r * r) * h3inv;
				}
			}
			g[i].g = g[i].g + f * y[j].m;
			g[i].phi += phi * y[j].m;
		}
	}
	return g;
}

std::vector<gravity> gravity_near(const std::vector<vect> &x, const std::vector<vect> &y) {
	const bool cuda = options::get().cuda;
	if (cuda && x.size()) {
		auto g = gravity_near_cuda(x, y);
		return g;
	} else {
		return gravity_near_cpu(x, y);
	}
}

std::vector<gravity> gravity_far(const std::vector<vect> &x, const std::vector<source> &y) {
//	const bool cuda = options::get().cuda;
	const bool cuda = false;
	if (cuda) {
		return gravity_far_cuda(x, y);
	} else {
		return gravity_far_cpu(x, y);
	}
}

