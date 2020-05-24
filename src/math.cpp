/*
 * math.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: dmarce1
 */

#include <ntiger/math.hpp>
#include <fenv.h>
#include <limits>

real condition_number(const std::array<vect, NDIM> &A, std::array<vect, NDIM> &Ainv) {
	static constexpr auto NMAX = real::max();
	if (matrix_inverse(A, Ainv)) {
		real Asum = 0.0;
		real Ainvsum = 0.0;
		for (int n = 0; n < NDIM; n++) {
			Asum += A[n].dot(A[n]);
			Ainvsum += Ainv[n].dot(Ainv[n]);
		}
		return sqrt(Asum * Ainvsum) / NDIM;
	} else {
		return NMAX;
	}
}

bool matrix_inverse(const std::array<vect, NDIM> &A, std::array<vect, NDIM> &Ainv) {
#if(NDIM==1)
	if( A[0][0] == 0.0 ) {
		return false;
	}
	Ainv[0][0] = 1.0 / A[0][0];
#else
#if(NDIM==2)
	real det = A[0][0] * A[1][1] - A[1][0] * A[0][1];
	if (det == 0.0) {
		return false;
	}
	const real detinv = 1.0 / det;
	Ainv[0][0] = +A[1][1] * detinv;
	Ainv[0][1] = -A[0][1] * detinv;
	Ainv[1][0] = -A[1][0] * detinv;
	Ainv[1][1] = +A[0][0] * detinv;
#else
	real det = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
	if( det == 0.0 ) {
		return false;
	}
	det /**/-= A[0][1] * (A[1][0] * A[2][2] - A[2][0] * A[1][2]);
	det /**/+= A[0][2] * (A[1][0] * A[2][1] - A[2][0] * A[1][1]);
	const real detinv = 1.0 / det;
	Ainv[0][0] = (-A[1][2] * A[2][1] + A[1][1] * A[2][2]) * detinv;
	Ainv[0][1] = (+A[0][2] * A[2][1] - A[0][1] * A[2][2]) * detinv;
	Ainv[0][2] = (-A[0][2] * A[1][1] + A[0][1] * A[1][2]) * detinv;
	Ainv[1][0] = (+A[1][2] * A[2][0] - A[1][0] * A[2][2]) * detinv;
	Ainv[1][1] = (-A[0][2] * A[2][0] + A[0][0] * A[2][2]) * detinv;
	Ainv[1][2] = (+A[0][2] * A[1][0] - A[0][0] * A[1][2]) * detinv;
	Ainv[2][0] = (-A[1][1] * A[2][0] + A[1][0] * A[2][1]) * detinv;
	Ainv[2][1] = (+A[0][1] * A[2][0] - A[0][0] * A[2][1]) * detinv;
	Ainv[2][2] = (-A[0][1] * A[1][0] + A[0][0] * A[1][1]) * detinv;
#endif
#endif
	return true;
}

vect rotate_to(const vect &u, const vect &n) {
#if(NDIM==1)
	return u.dot(n);
#elif(NDIM==2)
	vect m;
	vect v;
	m[0] = n[1];
	m[1] = -n[0];
	v[0] = u.dot(n);
	v[1] = u.dot(m);
	return v;
#else
	vect m;
	vect l;
	m = cross(u,n);
	if( m.dot(m)==0.0) {
		if( u.dot(n) > 0.0 ) {
			return u;
		} else {
			return -u;
		}
	} else {
		vect v;
		m = m / abs(m);
		l = cross(m,n);
		v[0] = u.dot(n);
		v[1] = u.dot(m);
		v[2] = u.dot(l);
		return v;
	}
#endif

}

vect rotate_from(const vect &u, vect n) {
#if(NDIM==1)
	return u.dot(n);
#elif(NDIM==2)
	vect m;
	vect v;
	m[0] = n[1];
	m[1] = -n[0];
	std::swap(n[1], m[0]);
	v[0] = u.dot(n);
	v[1] = u.dot(m);
	return v;
#else
	vect m;
	vect l;
	m = cross(u,n);
	if( m.dot(m)==0.0) {
		if( u.dot(n) > 0.0 ) {
			return u;
		} else {
			return -u;
		}
	} else {
		vect v;
		m = m / abs(m);
		l = cross(m,n);
		std::swap(n[1], m[0]);
		std::swap(n[2], l[0]);
		std::swap(m[2], l[1]);
		v[0] = u.dot(n);
		v[1] = u.dot(m);
		v[2] = u.dot(l);
		return v;
	}
#endif

}

void enable_floating_point_exceptions() {
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);
//	feenableexcept(FE_UNDERFLOW);
}

