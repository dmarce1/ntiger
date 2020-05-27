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
		a & v;
		a & g;
		a & x;
		a & phi;
	}
	particle() {
		t = dt = 0.0;
	}
};



#endif /* SRC_PARTICLE_HPP_ */
