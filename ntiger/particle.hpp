/*
 * particle.hpp
 *
 *  Created on: Dec 5, 2019
 *      Author: dmarce1
 */

#ifndef SRC_PARTICLE_HPP_
#define SRC_PARTICLE_HPP_

#include <ntiger/fixed_real.hpp>
#include <ntiger/vect.hpp>
#include <vector>

struct particle {
	real m;
	real phi;
	vect x;
	vect v;
	vect g;
	fixed_real t;
	fixed_real dt;
	void write(FILE*) const;
	int read(FILE*);
	template<class Arc>
	void serialize(Arc &&a, unsigned) {
		a & t;
		a & dt;
		a & m;
		a & v;
		a & g;
		a & x;
		a & phi;
	}
	particle() {
		t = dt = 0.0;
	}
};



struct gravity_part {
	real m;
	vect x;
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & m;
		arc & x;
	}
};

#endif /* SRC_PARTICLE_HPP_ */
