/*
 * rand.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: dmarce1
 */

#include <cstdlib>

#include <ntiger/rand.hpp>

real rand_unit_box() {
	return (rand() + 0.5) / (RAND_MAX + 1.0) - 0.5;
}

real rand1() {
	return (rand() + 0.5) / (RAND_MAX + 1.0);
}

vect rand_unit_vect() {
	vect n;
	real sum = 0.0;
	for (int dim = 0; dim < NDIM; dim++) {
		n[dim] = rand_unit_box();
		sum += n[dim] * n[dim];
	}
	sum = 1.0 / sqrt(sum);
	for (int dim = 0; dim < NDIM; dim++) {
		n[dim] *= sum;
	}
	return n;
}

real rand_normal() {
	const auto u1 = rand1();
	const auto u2 = rand1();
	return sqrt(-log(u1)) * cos(2.0 * M_PI * u2);
}
