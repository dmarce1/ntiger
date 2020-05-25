#include <ntiger/math.hpp>
#include <ntiger/tree.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>

#include <hpx/synchronization/mutex.hpp>

#if(NDIM == 1 )
constexpr real CV = 2.0;
constexpr int NNGB = 4;
#else
#if( NDIM == 2 )
constexpr real CV = M_PI;
constexpr int NNGB = 16;
#else
constexpr real CV = 4.0 * M_PI / 3.0;
constexpr int NNGB = 32;
#endif
#endif

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
			futs[ci] = hpx::async < advance_time_action > (children[ci].id, t);
		}
		hpx::wait_all(futs);
	}
}

void tree::compute_drift(fixed_real dt) {
	static const auto opts = options::get();
	if (leaf) {
		std::vector < std::vector < particle >> send_parts(siblings.size());
		std::vector<particle> parent_parts;
		{
			PROFILE();
			int sz = parts.size();
			bool found;
			for (int i = 0; i < sz; i++) {
				auto &pi = parts[i];
				pi.x = pi.x + pi.v * double(dt);
				if (!in_range(pi.x, box)) {
					found = false;
					for (int j = 0; j < siblings.size(); j++) {
						if (in_range(pi.x, siblings[j].box)) {
							send_parts[j].push_back(pi);
							found = true;
							break;
						}
					}
					if (!found) {
						parent_parts.push_back(pi);
					}
					sz--;
					parts[i] = parts[sz];
					i--;
					parts.resize(sz);
				}
			}
		}
		std::vector<hpx::future<void>> futs(siblings.size());
		for (int j = 0; j < siblings.size(); j++) {
			futs[j] = hpx::async < send_particles_action > (siblings[j].id, std::move(send_parts[j]));
		}
		if (parent != hpx::invalid_id) {
			if (parent_parts.size()) {
				find_home_action()(parent, std::move(parent_parts), false);
			}
		} else if (parent_parts.size()) {
			std::lock_guard < hpx::lcos::local::mutex > lock(lost_parts_mtx);
			lost_parts.insert(lost_parts.end(), parent_parts.begin(), parent_parts.end());
		}
		hpx::wait_all(futs);
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
		for (int i = 0; i < nparts0; i++) {
			auto &pi = parts[i];
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
		parts.resize(nparts0);
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

void tree::compute_interactions() {
	if (leaf) {
		static auto opts = options::get();
		nparts0 = parts.size();
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < compute_interactions_action > (children[ci].id);
		}
		hpx::wait_all(futs);
	}
}

