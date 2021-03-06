#include <ntiger/gravity.hpp>
#include <ntiger/gravity_cuda.hpp>
#include <ntiger/options.hpp>

#include <hpx/include/async.hpp>

#include <cassert>


void yield_to_hpx() {
	hpx::this_thread::yield();
}

using vect_int =
general_vect<int, NDIM>;
real EW(general_vect<double, NDIM>);

static ewald_table_t potential;
static std::array<ewald_table_t, NDIM> force;

void init_ewald() {
	FILE *fp = fopen("ewald.dat", "rb");
	if (fp) {
		int cnt = 0;
		printf("Found ewald.dat\n");
		const int sz = (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1);
		cnt += fread(&potential, sizeof(real), sz, fp);
		cnt += fread(&force, sizeof(real), NDIM * sz, fp);
		int expected = sz * (1 + NDIM);
		if (cnt != expected) {
			printf("ewald.dat is corrupt, read %i bytes, expected %i. Remove and re-run\n", cnt, expected);
			abort();
		}
		fclose(fp);
	} else {
		printf("ewald.dat not found\n");
		printf("Initializing Ewald (this may take some time)\n");

		const double dx0 = 0.5 / EWALD_NBIN;
		for (int dim = 0; dim < NDIM; dim++) {
			force[dim][0][0][0] = 0.0;
		}
		potential[0][0][0] = 2.8372975;
		real n = 0;
		for (int i = 0; i <= EWALD_NBIN; i++) {
			for (int j = 0; j <= EWALD_NBIN; j++) {
				printf("%% %.2f complete\r", n.get() / double(EWALD_NBIN + 1) / double(EWALD_NBIN + 1) * 100.0);
				n += 1.0;
				fflush (stdout);
				std::vector<hpx::future<void>> futs;
				for (int k = 0; k <= EWALD_NBIN; k++) {
					const auto func = [i, j, k, dx0]() {
						general_vect<double, NDIM> x;
						x[0] = i * dx0;
						x[1] = j * dx0;
						x[2] = k * dx0;
						if (x.dot(x) != 0.0) {
							const double dx = 0.01 * dx0;
							for (int dim = 0; dim < NDIM; dim++) {
								auto ym = x;
								auto yp = x;
								ym[dim] -= 0.5 * dx;
								yp[dim] += 0.5 * dx;
								const auto f = -(EW(yp) - EW(ym)) / dx;
								force[dim][i][j][k] = f;
							}
							const auto p = EW(x);
							potential[i][j][k] = p;
						}
					};
					futs.push_back(hpx::async(func));
				}
				hpx::wait_all(futs);
			}
		}
		printf("\nDone initializing Ewald\n");
		fp = fopen("ewald.dat", "wb");
		const int sz = (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1);
		fwrite(&potential, sizeof(real), sz, fp);
		fwrite(&force, sizeof(real), NDIM * sz, fp);
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

void ewald_force_and_pot(vect x, vect &f, real &phi) {
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
	if (r > 0.0) {
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
		const auto w100 = (1.0 - w[0]) * w[1] * w[2];
		const auto w101 = (1.0 - w[0]) * w[1] * (1.0 - w[2]);
		const auto w110 = (1.0 - w[0]) * (1.0 - w[1]) * w[2];
		const auto w111 = (1.0 - w[0]) * (1.0 - w[1]) * (1.0 - w[2]);
		for (int dim = 0; dim < NDIM; dim++) {
			f[dim] += force[dim][I[0]][I[1]][I[2]] * w000;
			f[dim] += force[dim][I[0]][I[1]][I[2] + 1] * w001;
			f[dim] += force[dim][I[0]][I[1] + 1][I[2]] * w010;
			f[dim] += force[dim][I[0]][I[1] + 1][I[2] + 1] * w011;
			f[dim] += force[dim][I[0] + 1][I[1]][I[2]] * w100;
			f[dim] += force[dim][I[0] + 1][I[1]][I[2] + 1] * w101;
			f[dim] += force[dim][I[0] + 1][I[1] + 1][I[2]] * w110;
			f[dim] += force[dim][I[0] + 1][I[1] + 1][I[2] + 1] * w111;
		}
		phi += potential[I[0]][I[1]][I[2]] * w000;
		phi += potential[I[0]][I[1]][I[2] + 1] * w001;
		phi += potential[I[0]][I[1] + 1][I[2]] * w010;
		phi += potential[I[0]][I[1] + 1][I[2] + 1] * w011;
		phi += potential[I[0] + 1][I[1]][I[2]] * w100;
		phi += potential[I[0] + 1][I[1]][I[2] + 1] * w101;
		phi += potential[I[0] + 1][I[1] + 1][I[2]] * w110;
		phi += potential[I[0] + 1][I[1] + 1][I[2] + 1] * w111;
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
				const auto xmn = x - n;                          // 3 OP
				double absxmn = abs(x - n);                      // 5 OP
				if (absxmn < 3.6) {
					const double xmn2 = absxmn * absxmn;         // 1 OP
					const double xmn3 = xmn2 * absxmn;           // 1 OP
					sum1 += -(1.0 - erf(2.0 * absxmn)) / absxmn; // 6 OP
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
				const double absh = abs(h);                     // 5 OP
				const double h2 = absh * absh;                  // 1 OP
				if (absh <= 10 && absh > 0) {
					sum2 += -(1.0 / M_PI) * (1.0 / h2 * exp(-M_PI * M_PI * h2 / 4.0) * cos(2.0 * M_PI * h.dot(x))); // 14 OP
				}
			}
		}
	}
	return M_PI / 4.0 + sum1 + sum2 + 1 / abs(x);
}

pinned_vector<gravity> direct_gravity_cpu(const pinned_vector<vect> &x, const pinned_vector<source> &y) {
	static const bool ewald = options::get().ewald;
	static const real h = options::get().kernel_size;
	static const real h2 = h * h;
	static const real hinv = 1.0 / h;
	static const real h3inv = hinv * hinv * hinv;
	const int cnt = x.size();
	pinned_vector<gravity> g(cnt);

	for (int i = 0; i < cnt; i++) {
		g[i].g = vect(0);
		g[i].phi = 0.0;
		for (int j = 0; j < y.size(); j++) {
			vect f;
			real phi;
			auto dx = x[i] - y[j].x;

			vect sgn(1.0);
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					if (dx[dim] < 0.0) {
						dx[dim] = -dx[dim];
						sgn[dim] *= -1.0;
					}
					if (dx[dim] > 0.5) {
						dx[dim] = 1.0 - dx[dim];
						sgn[dim] *= -1.0;
					}
				}
			}

			const auto r = abs(dx);
			if (r > h) {
				const auto rinv = 1.0 / r;
				const auto r3inv = rinv * rinv * rinv;
				f = -dx * r3inv;
				phi = -rinv;
			} else if (r > 0.0) {
				f = -dx * h3inv;
				phi = -(1.5 * h2 - 0.5 * r * r) * h3inv;
			} else {
				f = vect(0);
				phi = 0.0;
			}
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim] *= sgn[dim];
				}
			}
			g[i].g = g[i].g + f * y[j].m;
			g[i].phi += phi * y[j].m;
		}
	}
	return g;
}

pinned_vector<gravity> ewald_gravity_cpu(const pinned_vector<vect> &x, const pinned_vector<source> &y) {
	const int cnt = x.size();
	pinned_vector<gravity> g(cnt);

	for (int i = 0; i < cnt; i++) {
		g[i].g = vect(0);
		g[i].phi = 0.0;
		for (int j = 0; j < y.size(); j++) {
			vect f;
			real phi;
			const auto dx = x[i] - y[j].x;
			ewald_force_and_pot(dx, f, phi);
			g[i].g = g[i].g + f * y[j].m;
			g[i].phi += phi * y[j].m;
		}
	}
	return g;
}

pinned_vector<gravity> direct_gravity(const pinned_vector<vect> &x, const pinned_vector<source> &y) {
	const bool cuda = options::get().cuda;
//	const bool cuda = false;
	if (cuda) {
		return direct_gravity_cuda(x, y);
	} else {
		return direct_gravity_cpu(x, y);
	}
}

pinned_vector<gravity> ewald_gravity(const pinned_vector<vect> &x, const pinned_vector<source> &y) {
	const bool cuda = options::get().cuda;
//	const bool cuda = false;
	if (cuda) {
		return ewald_gravity_cuda(x, y);
	} else {
		return ewald_gravity_cpu(x, y);
	}
}

