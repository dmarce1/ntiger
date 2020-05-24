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

struct timestep_particle;
struct nesting_particle;

struct particle {
	real m0;
	vect x;
	vect vf;
	vect g;
	vect g0;
	real V;
	real h;
	real Nc;
	fixed_real t;
	fixed_real dt;
	fixed_real tmp;
	std::array<vect, NDIM> B;
	void write(FILE*) const;
	int read(FILE*);
	template<class Arc>
	void serialize(Arc &&a, unsigned) {
		a & t;
		a & dt;
		a & m0;
		a & g0;
		a & vf;
		a & g;
		a & x;
		a & V;
		a & h;
		a & B;
		a & Nc;
	}
	particle() {
		h = -1.0;
		t = dt = 0.0;
	}
	particle& operator=(const timestep_particle& p);
	particle(const timestep_particle &p);
	particle& operator=(const nesting_particle&);
	particle(const nesting_particle &p);
};

struct nesting_particle {
	fixed_real dt;
	vect x;
	real h;
	template<class Arc>
	void serialize(Arc &&a, unsigned) {
		a & dt;
		a & x;
		a & h;
	}
	nesting_particle() = default;
	nesting_particle& operator=(const particle &p);
	nesting_particle(const particle &p);
};

struct timestep_particle {
	vect vf;
	vect x;
	real h;
	template<class Arc>
	void serialize(Arc &&a, unsigned) {
		a & vf;
		a & h;
		a & x;
	}
	timestep_particle() = default;
	timestep_particle& operator=(const particle &p);
	timestep_particle(const particle &p);
};



struct gravity_part {
	real m;
	real h;
	vect x;
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & h;
		arc & m;
		arc & x;
	}
};

std::vector<particle> spherical_particle_set(int);
std::vector<particle> cartesian_particle_set(int);
std::vector<particle> random_particle_set(int);
std::vector<particle> disc_particle_set(int N);

#endif /* SRC_PARTICLE_HPP_ */
