#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/physcon.hpp>
#include <ntiger/rand.hpp>


std::vector<particle> cosmos(int cnt) {
	std::vector<particle> parts;
	for (int i = 0; i < cnt; i++) {
		vect x;
		for( int dim = 0; dim < NDIM; dim++) {
			x[dim] = rand_unit_box();
		}
		particle p;
		p.v = vect(0);
		p.x = x;
		p.t = p.dt = 0.0;
		parts.push_back(p);
	}

	return parts;
}

std::vector<particle> get_initial_particles(const std::string &name, int cnt) {
	if (false) {
	} else if (name == "cosmos") {
		return cosmos(cnt);
	} else {
		printf("Error: Initialization function %s is not known\n", name.c_str());
		abort();
	}
}
