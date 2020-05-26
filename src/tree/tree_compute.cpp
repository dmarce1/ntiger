#include <ntiger/ewald.hpp>
#include <ntiger/math.hpp>
#include <ntiger/tree.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/rand.hpp>

#include <hpx/synchronization/mutex.hpp>

void tree::compute_drift(fixed_real dt) {
	static const auto opts = options::get();
	if (leaf) {
		std::vector<particle> parent_parts;
		{
			PROFILE();
			int sz = parts.size();
			bool found;
			for (int i = 0; i < sz; i++) {
				auto &pi = parts[i];
				pi.x = pi.x + pi.v * double(dt);
				if (opts.ewald) {
					pi.x = ewald_location(pi.x);
				}
				if (!in_range(pi.x, box)) {
					parent_parts.push_back(pi);
					sz--;
					parts[i] = parts[sz];
					i--;
					parts.resize(sz);
				}
			}
		}
		if (parent != hpx::invalid_id) {
			if (parent_parts.size()) {
				find_home_action()(parent, std::move(parent_parts));
			}
		} else if (parent_parts.size()) {
			std::lock_guard < hpx::lcos::local::mutex > lock(lost_parts_mtx);
			lost_parts.insert(lost_parts.end(), parent_parts.begin(), parent_parts.end());
		}
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < compute_drift_action > (children[ci].id, dt);
		}
		hpx::wait_all(futs);
	}

}

real tree::compute_scale_factor() {
	if (lost_parts.size()) {
		real max_dim = 0.0;
		real current_dim = box.max[0];
		for (int i = 0; i < lost_parts.size(); i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				max_dim = std::max(max_dim, abs(lost_parts[i].x[dim]));
			}
		}
		return max_dim / current_dim * 1.1;
	} else {
		return 1.0;
	}
}

fixed_real tree::compute_timestep(fixed_real t) {
	const static auto opts = options::get();
	const auto h = opts.kernel_size;
	fixed_real tmin = fixed_real::max();
	if (leaf) {
		PROFILE();
		for (int i = 0; i < parts.size(); i++) {
			auto &pi = parts[i];
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

