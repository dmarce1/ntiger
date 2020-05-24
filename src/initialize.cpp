#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/physcon.hpp>
#include <ntiger/polytrope.hpp>
#include <ntiger/rand.hpp>


void kepler(particle &p) {
#if(NDIM==1)
	printf("Cannot do Kepler problem in 1D\n");
	abort();
#else
	static const auto opts = options::get();
	static const auto eps = opts.kep_eps;
	const auto r = abs(p.x);
	const auto y = p.x[1];
	const auto x = p.x[0];
	const auto tmp = pow(eps * eps + r * r, -0.75);
	p.m0 = 1.0;
	p.vf[0] = -y * tmp;
	p.vf[1] = x * tmp;
	if( NDIM > 2 ) {
		p.vf[2] = 0.0;
	}
#endif
}


init_func_type get_initialization_function(const std::string &name) {
	init_func_type init;
	if (name == "kepler") {
		init = kepler;
	} else {
		printf("Error: Initialization function %s is not known\n", name.c_str());
		abort();
	}
	return [init](particle &p) {
		init(p);
		p.t = p.dt = fixed_real(0);
	};
}
