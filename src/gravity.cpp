#include <ntiger/gravity.hpp>
#include <ntiger/gravity_cuda.hpp>
#include <ntiger/options.hpp>

#include <cassert>

using vect_int =
general_vect<int, NDIM>;
real EW(vect);

constexpr int EWALD_NBIN = 64;
static std::array<std::array<std::array<real, EWALD_NBIN + 1>, EWALD_NBIN + 1>, EWALD_NBIN + 1> potential;
static std::array<std::array<std::array<vect, EWALD_NBIN + 1>, EWALD_NBIN + 1>, EWALD_NBIN + 1> force;

struct ewald {
	ewald() {
		FILE *fp = fopen("ewald.dat", "rb");
		if (fp) {
			int cnt = 0;
			printf("Found ewald.dat\n");
			const int sz = (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1);
			cnt += fread(&potential, sizeof(real), sz, fp);
			cnt += fread(&force, sizeof(real), sz * NDIM, fp);
			int expected = sz * (NDIM + 1);
			if (cnt != expected) {
				printf("ewald.dat is corrupt, read %i bytes, expected %i. Remove and re-run\n", cnt, expected);
				abort();
			}
			fclose(fp);
		} else {
			printf("ewald.dat not found\n");
			printf("Initializing Ewald (this may take some time)\n");

			const real dx0 = 0.5 / EWALD_NBIN;
			for (int dim = 0; dim < NDIM; dim++) {
				force[0][0][0][dim] = 0.0;
			}
			potential[0][0][0] = 2.8372975;
			real n = 0;
			for (int i = 0; i <= EWALD_NBIN; i++) {
				for (int j = 0; j <= i; j++) {
					printf("%% %.2f complete\r", 2.0 * n.get() / double(EWALD_NBIN + 2) / double(EWALD_NBIN + 1) * 100.0);
					n += 1.0;
					fflush (stdout);
					for (int k = 0; k <= j; k++) {
						vect x;
						x[0] = i * dx0;
						x[1] = j * dx0;
						x[2] = k * dx0;
						if (x.dot(x) == 0.0) {
							continue;
						}
						const real dx = 0.001 * dx0;
						for (int dim = 0; dim < NDIM; dim++) {
							vect ym = x;
							vect yp = x;
							yp[dim] += 0.5 * dx;
							ym[dim] -= 0.5 * dx;
							const auto f = -(EW(yp) - EW(ym)) / dx;
							force[i][j][k][dim] = f;
						}
						const auto f = force[i][j][k];
						force[i][k][j][0] = f[0];
						force[i][k][j][1] = f[2];
						force[i][k][j][2] = f[1];

						force[j][i][k][0] = f[1];
						force[j][i][k][1] = f[0];
						force[j][i][k][2] = f[2];

						force[j][k][i][0] = f[1];
						force[j][k][i][1] = f[2];
						force[j][k][i][2] = f[0];

						force[k][i][j][0] = f[2];
						force[k][i][j][1] = f[0];
						force[k][i][j][2] = f[1];

						force[k][j][i][0] = f[2];
						force[k][j][i][1] = f[1];
						force[k][j][i][2] = f[0];

						const auto p = EW(x);

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
		for (int dim = 0; dim < NDIM; dim++) {
			f[dim] += force[I[0]][I[1]][I[2]][dim] * w000;
			f[dim] += force[I[0]][I[1]][I[2] + 1][dim] * w001;
			f[dim] += force[I[0]][I[1] + 1][I[2]][dim] * w010;
			f[dim] += force[I[0]][I[1] + 1][I[2] + 1][dim] * w011;
			f[dim] += force[I[0] + 1][I[1]][I[2]][dim] * w100;
			f[dim] += force[I[0] + 1][I[1]][I[2] + 1][dim] * w101;
			f[dim] += force[I[0] + 1][I[1] + 1][I[2]][dim] * w110;
			f[dim] += force[I[0] + 1][I[1] + 1][I[2] + 1][dim] * w111;
		}
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

real EW(vect x) {
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
	if (cuda && x.size() >= 32) {
		return gravity_near_cuda(x, y);
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

