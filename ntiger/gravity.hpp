#pragma once

#include <ntiger/vect.hpp>
#include <vector>

constexpr int EWALD_NBIN = 64;

using ewald_table_t = std::array<std::array<std::array<real, EWALD_NBIN + 1>, EWALD_NBIN + 1>, EWALD_NBIN + 1>;


void ewald_force_and_pot(vect x, vect &f, real &phi, real);
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
};

std::vector<gravity> gravity_near(const std::vector<vect> &x, const std::vector<vect> &y);
std::vector<gravity> gravity_far(const std::vector<vect> &x, const std::vector<source> &y);
