#pragma once

#include <ntiger/vect.hpp>
#include <ntiger/pinned_vector.hpp>
#include <vector>


constexpr int EWALD_NBIN = 64;
#define EWALD_R0 0.08

using ewald_table_t = std::array<std::array<std::array<real, EWALD_NBIN + 1>, EWALD_NBIN + 1>, EWALD_NBIN + 1>;

void ewald_force_and_pot(vect x, vect &f, real &phi);
real ewald_separation(vect x);
vect ewald_location(vect x);
real EW(vect);
void init_ewald();

struct gravity {
	real phi;
	vect g;
};

struct source {
	real m;
	vect x;

	template<class Arc>
	void serialize(Arc &&a, unsigned) {
		a & m;
		a & x;
	}
};

pinned_vector<gravity> direct_gravity(const pinned_vector<vect> &x, const pinned_vector<source> &y);
pinned_vector<gravity> ewald_gravity(const pinned_vector<vect> &x, const pinned_vector<source> &y);
