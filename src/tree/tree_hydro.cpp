#include <ntiger/math.hpp>
#include <ntiger/tree.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>



void tree::advance_time(fixed_real t) {
	static auto opts = options::get();
	const auto use_grav = opts.gravity || opts.problem == "kepler";
	if (leaf) {
		PROFILE();
		parts.resize(nparts0);
		for (int i = 0; i < nparts0; i++) {
			auto &p = parts[i];
			if (p.t + p.dt == t || opts.global_time) {
				p.t += p.dt;
			}
		}
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async<advance_time_action>(children[ci], t);
		}
		hpx::wait_all(futs);
	}
}
