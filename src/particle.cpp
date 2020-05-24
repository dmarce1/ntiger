/*
 * particle.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: dmarce1
 */
#include <ntiger/options.hpp>
#include <ntiger/particle.hpp>
#include <ntiger/rand.hpp>

std::vector<particle> disc_particle_set(int N) {
	std::vector<particle> rparts;
	const auto cparts = cartesian_particle_set(N);
	for (auto p : cparts) {
		const auto r = abs(p.x);
		if (r < 0.5 && r > 0.0) {
			rparts.push_back(p);
		}
	}
	return rparts;
}

std::vector<particle> cartesian_particle_set(int N) {
	std::vector<particle> parts;
	parts.reserve(std::pow(N, NDIM));
	particle part;
#if(NDIM==3)
	for (int l = 0; l < N; l++) {
		part.x[2] = (l + 0.5) / N - 0.5;
#endif
#if(NDIM>=2)
		for (int j = 0; j < N; j++) {
			part.x[1] = (real(j) + 0.5) / real(N) - real(0.5);
#endif
			for (int i = 0; i < N; i++) {
				part.x[0] = (i + 0.5) / N - 0.5;
				parts.push_back(part);
			}
#if(NDIM>=2)
		}
#endif
#if(NDIM==3)
	}
#endif
	return std::move(parts);
}

std::vector<particle> spherical_particle_set(int N) {
	std::vector<particle> parts;
	parts.reserve(std::pow(N, NDIM));
	particle part;
#if(NDIM==3)
	for (int l = 0; l < N; l++) {
		part.x[2] = (l + 0.5) / N - 0.5;
#endif
#if(NDIM>=2)
		for (int j = 0; j < N; j++) {
			part.x[1] = (real(j) + 0.5) / real(N) - real(0.5);
#endif
			for (int i = 0; i < N; i++) {
				part.x[0] = (i + 0.5) / N - 0.5;
				if( abs(part.x) <= 0.5 ) {
					parts.push_back(part);
				}
			}
#if(NDIM>=2)
		}
#endif
#if(NDIM==3)
	}
#endif
	return std::move(parts);
}

std::vector<particle> random_particle_set(int N) {
	std::vector<particle> parts(N);
	for (auto &part : parts) {
		for (int dim = 0; dim < NDIM; dim++) {
			part.x[dim] = rand_unit_box();
		}
	}
	return std::move(parts);
}

void particle::write(FILE *fp) const {
	real r;
	fwrite(&vf, sizeof(vect), 1, fp);
	fwrite(&x, sizeof(vect), 1, fp);
	fwrite(&g, sizeof(vect), 1, fp);
	fwrite(&g0, sizeof(vect), 1, fp);
	fwrite(&m0, sizeof(real), 1, fp);
	fwrite(&V, sizeof(real), 1, fp);
	fwrite(&h, sizeof(real), 1, fp);
	fwrite(&Nc, sizeof(real), 1, fp);
	fwrite(&t, sizeof(fixed_real), 1, fp);
	fwrite(&dt, sizeof(fixed_real), 1, fp);
	fwrite(&B, sizeof(vect), NDIM, fp);
}

int particle::read(FILE *fp) {
	int cnt = 0;
	cnt += fread(&vf, sizeof(vect), 1, fp);
	cnt += fread(&x, sizeof(vect), 1, fp);
	cnt += fread(&g, sizeof(vect), 1, fp);
	cnt += fread(&g0, sizeof(vect), 1, fp);
	cnt += fread(&m0, sizeof(real), 1, fp);
	cnt += fread(&V, sizeof(real), 1, fp);
	cnt += fread(&h, sizeof(real), 1, fp);
	cnt += fread(&Nc, sizeof(real), 1, fp);
	cnt += fread(&t, sizeof(fixed_real), 1, fp);
	cnt += fread(&dt, sizeof(fixed_real), 1, fp);
	cnt += fread(&B, sizeof(vect), NDIM, fp);
	return cnt;
}

particle& particle::operator=(const timestep_particle &p) {
	vf = p.vf;
	h = p.h;
	x = p.x;
	return *this;
}

nesting_particle& nesting_particle::operator=(const particle &p) {
	dt = p.dt;
	x = p.x;
	h = p.h;
	return *this;
}

nesting_particle::nesting_particle(const particle &p) {
	*this = p;
}

particle::particle(const timestep_particle &p) {
	*this = p;
}

particle& particle::operator=(const nesting_particle &p) {
	dt = p.dt;
	x = p.x;
	h = p.h;
	return *this;
}

particle::particle(const nesting_particle &p) {
	*this = p;
}

timestep_particle& timestep_particle::operator=(const particle &p) {
	vf = p.vf;
	h = p.h;
	x = p.x;
	return *this;
}

timestep_particle::timestep_particle(const particle &p) {
	*this = p;
}

