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
	real phi;   // 4
	vect x;     // 12
	vect v;     // 12
	vect g;     // 12
	fixed_real t; // 8
	fixed_real dt; //8
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
	}
};

real sort_by_dimension( std::vector<particle>&, int dim);


#endif /* SRC_PARTICLE_HPP_ */
