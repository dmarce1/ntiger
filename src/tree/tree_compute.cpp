#include <ntiger/gravity.hpp>
#include <ntiger/math.hpp>
#include <ntiger/tree.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/rand.hpp>

#include <hpx/lcos/local/mutex.hpp>

void tree::apply_boost(vect x) {
	static const auto opts = options::get();
	if (leaf) {
		for (auto &pi : parts) {
			pi.v = pi.v + x;
		}
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < apply_boost_action > (children[ci].id, x);
		}
		hpx::wait_all(futs);
	}

}
void tree::compute_drift(fixed_real dt) {
	static const auto opts = options::get();
	if (leaf) {
		list<particle> parent_parts;
		{
			PROFILE();
			std::lock_guard < hpx::lcos::local::mutex > lock(*mtx);
			int sz = parts.size();

			bool found;
			for (auto i = parts.begin(); i != parts.end();) {
				auto &pi = *i;
				pi.x = pi.x + pi.v * double(dt);
//				printf( "%f %f %f\n", pi.x[0].get(), pi.x[1].get(), pi.x[2].get());
				if (opts.ewald) {
					pi.x = ewald_location(pi.x);
				}
				i++;
				if (!in_range(pi.x, box)) {
					parent_parts.push_front(pi);
					std::swap(parts.front(), pi);
					parts.pop_front();
				}
			}
		}
		if (parent != hpx::invalid_id) {
			if (parent_parts.size()) {
				find_home_action()(parent, std::move(parent_parts));
			}
		}
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < compute_drift_action > (children[ci].id, dt);
		}
		hpx::wait_all(futs);
	}

}

fixed_real tree::compute_timestep(fixed_real t) {
	const static auto opts = options::get();
	const auto h = opts.kernel_size;
	fixed_real tmin = fixed_real::max();
	if (leaf) {
		PROFILE();
		for (auto &pi : parts) {
			if (pi.t + pi.dt == t || opts.global_time) {
				pi.t += pi.dt;
			}
			if (pi.t == t || opts.global_time) {
				pi.dt = fixed_real::max();
				const auto a = abs(pi.g);
				if (a > 0.0) {
					const real this_dt = sqrt(h / a);
					if (this_dt < (double) fixed_real::max()) {
						pi.dt = double(min(pi.dt, fixed_real(this_dt.get())));
					}
				}
				pi.dt *= opts.cfl;
				pi.dt = pi.dt.nearest_log2();
				pi.dt = min(pi.dt, t.next_bin() - t);
				tmin = min(tmin, pi.dt);
			}
		}
	} else {
		std::array<hpx::future<fixed_real>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < compute_timestep_action > (children[ci].id, t);
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			tmin = min(tmin, futs[ci].get());
		}
	}
	return tmin;
}

void tree::virialize() {
	if (leaf) {
		for (auto &p : parts) {
			p.v = rand_unit_vect() * sqrt(-p.phi / 2.0) * rand_normal();
		}
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < virialize_action > (children[ci].id);
		}
		hpx::wait_all(futs);
	}
}

void tree::keplerize() {
	if (leaf) {
		for (auto &p : parts) {
			const auto r = abs(p.x);
			const auto g = abs(p.g);
			p.v[0] = sqrt(g * r) * (-p.x[1]) / r;
			p.v[1] = sqrt(g * r) * (+p.x[0]) / r;
			p.v[2] = 0.0;
		}
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < keplerize_action > (children[ci].id);
		}
		hpx::wait_all(futs);
	}
}

