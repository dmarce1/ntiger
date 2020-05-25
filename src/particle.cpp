/*
 * particle.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: dmarce1
 */
#include <ntiger/options.hpp>
#include <ntiger/particle.hpp>
#include <ntiger/rand.hpp>


void particle::write(FILE *fp) const {
	real r;
	fwrite(&v, sizeof(vect), 1, fp);
	fwrite(&x, sizeof(vect), 1, fp);
	fwrite(&g, sizeof(vect), 1, fp);
	fwrite(&m, sizeof(real), 1, fp);
	fwrite(&phi, sizeof(real), 1, fp);
	fwrite(&t, sizeof(fixed_real), 1, fp);
	fwrite(&dt, sizeof(fixed_real), 1, fp);
}

int particle::read(FILE *fp) {
	int cnt = 0;
	cnt += fread(&v, sizeof(vect), 1, fp);
	cnt += fread(&x, sizeof(vect), 1, fp);
	cnt += fread(&g, sizeof(vect), 1, fp);
	cnt += fread(&m, sizeof(real), 1, fp);
	cnt += fread(&phi, sizeof(real), 1, fp);
	cnt += fread(&t, sizeof(fixed_real), 1, fp);
	cnt += fread(&dt, sizeof(fixed_real), 1, fp);
	return cnt;
}

