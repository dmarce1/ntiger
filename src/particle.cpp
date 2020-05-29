/*
 * particle.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: dmarce1
 */
#include <ntiger/options.hpp>
#include <ntiger/particle.hpp>
#include <ntiger/rand.hpp>

#include <algorithm>

void particle::write(FILE *fp) const {
	real r;
	fwrite(&x, sizeof(vect), 1, fp);
	fwrite(&g, sizeof(vect), 1, fp);
	fwrite(&v, sizeof(vect), 1, fp);
	fwrite(&phi, sizeof(real_type), 1, fp);
	fwrite(&t, sizeof(fixed_real), 1, fp);
	fwrite(&dt, sizeof(fixed_real), 1, fp);
}

int particle::read(FILE *fp) {
	int cnt = 0;
	cnt += fread(&x, sizeof(vect), 1, fp);
	cnt += fread(&g, sizeof(vect), 1, fp);
	cnt += fread(&v, sizeof(vect), 1, fp);
	cnt += fread(&phi, sizeof(real_type), 1, fp);
	cnt += fread(&t, sizeof(fixed_real), 1, fp);
	cnt += fread(&dt, sizeof(fixed_real), 1, fp);
	return cnt;
}

real sort_by_dimension(std::vector<particle> &p, int d) {
	if (p.size() == 1) {
		return p[0].x[d];
	} else if (p.size()) {
		const auto comp = [d](const particle &a, const particle &b) {
			return a.x[d] < b.x[d];
		};
		std::sort(p.begin(), p.end(), comp);
		const int i = p.size() / 2;
		return (p[i].x[d] + p[i - 1].x[d]) / 2.0;
	} else {
		return 0.0;
	}
}

