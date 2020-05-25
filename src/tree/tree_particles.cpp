/*
 * tree_particles.cpp
 *
 *  Created on: Jan 26, 2020
 *      Author: dmarce1
 */

#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>



void tree::get_neighbor_particles(tree::bnd_ex_type type) {
	static auto opts = options::get();
	if (leaf) {
		parts.resize(nparts0);
		range sbox = null_range();
		for (const auto &pi : parts) {
			for (int dim = 0; dim < NDIM; dim++) {
				sbox.min[dim] = min(sbox.min[dim], pi.x[dim] - pi.h);
				sbox.max[dim] = max(sbox.max[dim], pi.x[dim] + pi.h);
			}
		}
		int k = nparts0;

		switch (type) {
		case NESTING: {
			std::vector<hpx::future<std::vector<nesting_particle>>> futs(siblings.size());
			for (int i = 0; i < siblings.size(); i++) {
				futs[i] = hpx::async<get_nesting_particles_action>(siblings[i].id, sbox, box);
			}
			for (int i = 0; i < siblings.size(); i++) {
				const auto these_parts = futs[i].get();
				std::lock_guard<hpx::lcos::local::mutex> lock(*mtx);
				parts.insert(parts.end(), these_parts.begin(), these_parts.end());
			}
			break;
		}
		case TIMESTEP: {
			std::vector<hpx::future<std::vector<timestep_particle>>> futs(siblings.size());
			for (int i = 0; i < siblings.size(); i++) {
				futs[i] = hpx::async<get_timestep_particles_action>(siblings[i].id, sbox, box);
			}
			for (int i = 0; i < siblings.size(); i++) {
				const auto these_parts = futs[i].get();
				std::lock_guard<hpx::lcos::local::mutex> lock(*mtx);
				parts.insert(parts.end(), these_parts.begin(), these_parts.end());
			}
			break;
		}
		case ALL: {
			std::vector<hpx::future<std::vector<particle>>> futs(siblings.size());
			for (int i = 0; i < siblings.size(); i++) {
				futs[i] = hpx::async<get_particles_action>(siblings[i].id, sbox, box);
			}
			for (int i = 0; i < siblings.size(); i++) {
				const auto these_parts = futs[i].get();
				std::lock_guard<hpx::lcos::local::mutex> lock(*mtx);
				parts.insert(parts.end(), these_parts.begin(), these_parts.end());
			}
			break;
		}
		default:
			assert(false);
		}
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async<get_neighbor_particles_action>(children[ci].id, type);
		}
		hpx::wait_all(futs);
	}
}


std::vector<particle> tree::get_particles(range big, range small) const {
	std::lock_guard<hpx::lcos::local::mutex> lock(*mtx);
	PROFILE();
	std::vector<particle> pj;
	for (int i = 0; i < nparts0; i++) {
		auto pi = parts[i];
		if (in_range(pi.x, big) || ranges_intersect(range_around(pi.x, pi.h), small)) {
			pj.push_back(pi);
		}
	}
	return pj;
}

std::vector<nesting_particle> tree::get_nesting_particles(range big, range small) const {
	std::lock_guard<hpx::lcos::local::mutex> lock(*mtx);
	PROFILE();
	std::vector<nesting_particle> pj;
	for (int i = 0; i < nparts0; i++) {
		auto pi = parts[i];
		if (in_range(pi.x, big) || ranges_intersect(range_around(pi.x, pi.h), small)) {
			pj.push_back(pi);
		}
	}
	return pj;
}


std::vector<timestep_particle> tree::get_timestep_particles(range big, range small) const {
	std::lock_guard<hpx::lcos::local::mutex> lock(*mtx);
	PROFILE();
	std::vector<timestep_particle> pj;
	for (int i = 0; i < nparts0; i++) {
		auto pi = parts[i];
		if (in_range(pi.x, big) || ranges_intersect(range_around(pi.x, pi.h), small)) {
			pj.push_back(pi);
		}
	}
	return pj;
}



std::vector<vect> tree::get_particle_positions(range search) const {
	PROFILE();
	const int sz = parts.size();
	std::vector<vect> pos;
	for (int i = 0; i < sz; i++) {
		const auto &pi = parts[i];
		if (in_range(pi.x, search)) {
			pos.push_back(parts[i].x);
		}
	}
	return pos;
}


void tree::send_particles(const std::vector<particle> &pj) {
	std::lock_guard<hpx::lcos::local::mutex> lock(*mtx);
	PROFILE();
	for (auto p : pj) {
		new_parts.push_back(std::move(p));
	}
}


