#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/physcon.hpp>
#include <ntiger/rand.hpp>

std::vector<particle> plummer(int cnt) {
	std::vector<particle> p(cnt);
	real rmax = 5.0;
	for (int i = 0; i < cnt; i++) {
		vect x;
		real rho;
		real r;
		do {
			x = rand_unit_vect() * rmax * rand1();
			r = abs(x);
			rho = pow(1.0 + r * r, -5.0 / 2.0);
		} while (rand1() > rho);
		p[i].x = x;
	}
	return p;
}

std::vector<particle> toomre1(int cnt) {
	std::vector<particle> p(cnt);
	real rmax = 5.0;
	for (int i = 0; i < cnt; i++) {
		vect x;
		real rho;
		real r;
		do {
			x = rand_unit_vect() * rmax * rand1();
			r = sqrt(abs(x) * abs(x) - x[2] * x[2]);
			rho = pow(1.0 + r * r, -3.0 / 2.0);
			x[2] = rand_normal() * rand_sign() * 0.001;
		} while (rand1() > rho);
		p[i].x = x;
	}
	return p;
}

std::vector<particle> kepler(int cnt) {
	std::vector<particle> parts(cnt);
	const auto m = 1.0e-3 / cnt;
	parts[0].v = vect(0);
	parts[0].x = vect(0);
	for (int i = 1; i < cnt; i++) {
		bool done = false;
		do {
			const auto x = rand_unit_box();
			const auto y = rand_unit_box();
			const auto r = sqrt(x * x + y * y);
			const auto rsqrrinv = 1.0 / (r * sqrt(r));
			auto &p = parts[i];
			if (r < 0.5 && r > 0.1) {
				p.x[0] = x;
				p.x[1] = y;
				p.v[0] = -y * rsqrrinv;
				p.v[1] = x * rsqrrinv;
				if ( NDIM > 2) {
					p.x[2] = p.v[2] = 0.0;
				}
				done = true;
			}
		} while (!done);
	}
	return parts;
}

std::vector<particle> cosmos(int cnt) {
	std::vector<particle> p(cnt);
	for (int i = 0; i < cnt; i++) {
		vect x;
		for( int dim = 0; dim < NDIM; dim++) {
			x[dim] = rand_unit_box();
		}
		p[i].v = vect(0);
		p[i].x = x;
	}

//	int N = pow(cnt,1.0/3.0);
//	p.resize(N*N*N);
//	cnt = N*N*N;
//	int l = 0;
//	for( int i = 0; i < N; i++) {
//		for( int j = 0; j < N; j++) {
//			for( int k= 0; k < N; k++) {
//				vect x;
//				x[0] = (i + 0.5)/N - 0.5;
//				x[1] = (j + 0.5)/N - 0.5;
//				x[2] = (k + 0.5)/N - 0.5;
//				p[l].x = x;
//				p[l].v = vect(0);
//				p[l].m = 1.0 / cnt;
//				l++;
//			}
//		}
//	}
	p[0].x[0] = -0.25;
	p[0].x[1] = -0.25;
	p[0].x[2] = 0.0;
	p[1].x[0] = 0.25;
	p[1].x[1] = +0.25;
	p[1].x[2] = 0.0;
//	p[2].x[0] = +0.25;
//	p[2].x[1] = -0.25;
//	p[2].x[2] = 0.0;
//	p[3].x[0] = +0.25;
//	p[3].x[1] = +0.25;
//	p[3].x[2] = 0.0;
	return p;
}

std::vector<particle> get_initial_particles(const std::string &name, int cnt) {
	if (false) {
	} else if (name == "kepler") {
		return kepler(cnt);
	} else if (name == "cosmos") {
		return cosmos(cnt);
	} else if (name == "toomre") {
		return toomre1(cnt);
	} else if (name == "plummer") {
		return plummer(cnt);
	} else {
		printf("Error: Initialization function %s is not known\n", name.c_str());
		abort();
	}
}
