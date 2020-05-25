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
	fixed_real tmin = fixed_real::max();
	if (leaf) {
		PROFILE();
		for (int i = 0; i < nparts0; i++) {
			auto &pi = parts[i];
			if (pi.t == t || opts.global_time) {
				pi.dt = fixed_real::max();
				const auto a = abs(pi.g);
				if (a > 0.0) {
					const real this_dt = sqrt(pi.h / a);
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
		if (opts.kernel_size < 0.0) {
			const auto toler = NNGB * 10.0 * real::eps();
			if (nparts0) {
				std::vector<vect> pos;
				pos.reserve(2 * parts.size());
				const auto h0 = pow(range_volume(box) / (CV * parts.size()), 1.0 / NDIM);
				for (auto &pi : parts) {
					pos.push_back(pi.x);
					if (pi.h == -1) {
						pi.h = h0;
					}
				}
				const auto hmax = box.max[0] - box.min[0];
				for (int pass = 0; pass < 2; pass++) {
					{
						PROFILE();
						for (auto &pi : parts) {
							if (!(pass == 1 && in_range(range_around(pi.x, pi.h), box))) {
								bool done = false;
								auto &h = pi.h;
//						real max_dh = real::max();
								int iters = 0;
								real dh = pi.h / 2.0;
								do {
									real N = 0.0;
									real Np = 0.0;
									real dNdh;
									const auto eps = abs(dh);
									for (const auto &pj : pos) {
										if (pj != pi.x) {
											const auto r = abs(pj - pi.x);
											if (r < h + eps) {
												Np += CV * pow(h + eps, NDIM) * W(r, h + eps);
												if (r < h) {
													N += CV * pow(h, NDIM) * W(r, h);
												}
											}
										}
									}
									if (abs(NNGB - N) < toler) {
										done = true;
									} else {
										dNdh = (Np - N) / eps;
										if (dNdh == 0.0) {
											h *= 1.2;
										} else {
											dh = -(N - NNGB) / dNdh;
											dh = min(h * 0.5, max(-0.5 * h, dh));
											//		max_dh = min(0.999 * max_dh, abs(dh));
											//		dh = copysign(min(max_dh, abs(dh)), dh);
											h += dh;
										}
									}
									iters++;
									if (pass == 0) {
										if (iters > 100 || h > hmax) {
											break;
										}
									} else {
										if (iters >= 50) {
											printf("%e %e %e\n", h.get(), dh.get(), /*max_dh.get(), */N.get());
											if (iters == 100) {
												printf("Smoothing length failed to converge\n");
												abort();
											}
										}
									}
								} while (!done);
							}
						}
					}
					if (pass == 0) {
						range sbox = null_range();
						for (const auto &pi : parts) {
							for (int dim = 0; dim < NDIM; dim++) {
								sbox.min[dim] = min(sbox.min[dim], pi.x[dim] - pi.h);
								sbox.max[dim] = max(sbox.max[dim], pi.x[dim] + pi.h);
							}
						}
						std::vector < hpx::future<std::vector<vect>> > futs(siblings.size());
						for (int i = 0; i < siblings.size(); i++) {
							assert(siblings[i].id != hpx::invalid_id);
							futs[i] = hpx::async < get_particle_positions_action > (siblings[i].id, sbox);
						}
						for (int i = 0; i < siblings.size(); i++) {
							const auto tmp = futs[i].get();
							pos.insert(pos.end(), tmp.begin(), tmp.end());
						}
					}
				}
			}
		} else {
			for (auto &p : parts) {
				p.h = opts.kernel_size;
			}
		}
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < compute_interactions_action > (children[ci].id);
		}
		hpx::wait_all(futs);
	}
}

