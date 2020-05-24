/*
 * polytrope.cpp
 *
 *  Created on: Dec 26, 2019
 *      Author: dmarce1
 */


#include <ntiger/physcon.hpp>


static inline real pow_3_2(real y) {
    return y * sqrt(y);
}

static inline real fy(real y, real z, real r) {
	return z;
}

static inline real fz(real y, real z, real r) {
	if (r != 0.0) {
		return -(pow_3_2(y) + 2.0 * z / r);
	}
	return -3.0;
}


real polytrope(real r0) {
	real dr = 0.1;
    real dy1, dz1, y, z, r, dy2, dz2, dy3, dz3, dy4, dz4, y0, z0;
	int done = 0;
	y = 1.0;
	z = 0.0;
	int N = static_cast<int>((r0 / dr + 0.5).get());
	if (N < 1) {
		N = 1;
	}
	r = 0.0;
	do {
		if (r + dr > r0) {
			dr = r0 - r;
			done = 1;
		}
		y0 = y;
		z0 = z;
		dy1 = fy(y, z, r) * dr;
		dz1 = fz(y, z, r) * dr;
		y += 0.5 * dy1;
		z += 0.5 * dz1;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
        real rdr2 = r + 0.5 * dr;
		dy2 = fy(y, z, rdr2) * dr;
		dz2 = fz(y, z, rdr2) * dr;
		y = y0 + 0.5 * dy2;
		z = z0 + 0.5 * dz2;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
		dy3 = fy(y, z, rdr2) * dr;
		dz3 = fz(y, z, rdr2) * dr;
		y = y0 + dy3;
		z = z0 + dz3;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
        real rdr = r + dr;
		dy4 = fy(y, z, rdr) * dr;
		dz4 = fz(y, z, rdr) * dr;
		y = y0 + (dy1 + dy4 + 2.0 * (dy3 + dy2)) / 6.0;
		z = z0 + (dz1 + dz4 + 2.0 * (dz3 + dz2)) / 6.0;
		if (y <= 0.0) {
			y = 0.0;
			break;
		}
		r += dr;
	} while (done == 0);
	if (y < 0.0) {
		return 0.0;
	}
	return y;
}
