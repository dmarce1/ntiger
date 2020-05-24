#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/physcon.hpp>
#include <ntiger/polytrope.hpp>
#include <ntiger/rand.hpp>

std::vector<particle> kepler(int cnt) {
	std::vector<particle> parts(cnt);
	const auto m = 1.0e-3 / cnt;
	parts[0].m0 = 1.0;
	parts[0].vf = vect(0);
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
				p.m0 = 1.0e-20;
				p.x[0] = x;
				p.x[1] = y;
				p.vf[0] = -y * rsqrrinv;
				p.vf[1] = x * rsqrrinv;
				if ( NDIM > 2) {
					p.x[2] = p.vf[2] = 0.0;
				}
				done = true;
			}
		} while (!done);
	}
	return parts;
}

std::vector<particle> get_initial_particles(const std::string &name, int cnt) {
	if (false) {
#if(NDIM>1)
	} else if (name == "kepler") {
		return kepler(cnt);
#endif
	} else {
		printf("Error: Initialization function %s is not known\n", name.c_str());
		abort();
	}
}
