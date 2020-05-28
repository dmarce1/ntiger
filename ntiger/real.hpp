/*
 * real.hpp
 *
 *  Created on: Nov 29, 2019
 *      Author: dmarce1
 */

#ifndef REAL_HPP_
#define REAL_HPP_

#include <ntiger/cuda_export.hpp>

#include <limits>
#include <algorithm>
#include <cmath>

using real_type = float;

class real {
	real_type r;
public:
	CONSTEXPR CUDA_EXPORT real()
#ifndef __CUDA_ARCH__
	:
			r(std::numeric_limits < real_type > ::signaling_NaN())
#endif
	{
	}
	CONSTEXPR CUDA_EXPORT real(real_type a) :
			r(a) {
	}
	CUDA_EXPORT inline double get() const {
		return r;
	}
	CUDA_EXPORT inline real operator+(const real &b) const {
		return real(r + b.r);
	}
	CUDA_EXPORT inline real operator*(const real &b) const {
		return real(r * b.r);
	}
	CUDA_EXPORT inline real operator/(const real &b) const {
		return real(r / b.r);
	}
	CUDA_EXPORT inline real operator-(const real &b) const {
		return real(r - b.r);
	}
	CUDA_EXPORT inline real& operator+=(const real &b) {
		r += b.r;
		return *this;
	}
	CUDA_EXPORT inline real& operator-=(const real &b) {
		r -= b.r;
		return *this;
	}
	CUDA_EXPORT inline real& operator*=(const real &b) {
		r *= b.r;
		return *this;
	}
	CUDA_EXPORT inline real& operator/=(const real &b) {
		r /= b.r;
		return *this;
	}
	CUDA_EXPORT inline real operator-() const {
		return real(-r);
	}
	CUDA_EXPORT inline real operator+() const {
		return *this;
	}
	CUDA_EXPORT inline bool operator<(const real &a) const {
		return r < a.r;
	}
	CUDA_EXPORT inline bool operator>(const real &a) const {
		return r > a.r;
	}
	CUDA_EXPORT inline bool operator<=(const real &a) const {
		return r <= a.r;
	}
	CUDA_EXPORT inline bool operator>=(const real &a) const {
		return r >= a.r;
	}
	CUDA_EXPORT inline bool operator==(const real &a) const {
		return r == a.r;
	}
	CUDA_EXPORT inline bool operator!=(const real &a) const {
		return r != a.r;
	}
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & r;
	}
	static CONSTEXPR real max() {
		return std::numeric_limits < real_type > ::max();
	}
	static CONSTEXPR real min() {
		return std::numeric_limits < real_type > ::min();
	}
	static CONSTEXPR real eps() {
		return std::numeric_limits < real_type > ::epsilon();
	}
	friend CUDA_EXPORT real copysign(real a, real b);
	friend real max(real a, real b);
	friend real min(real a, real b);
	friend CUDA_EXPORT real sin(real a);
	friend CUDA_EXPORT real cos(real a);
	friend CUDA_EXPORT real abs(real a);
	friend CUDA_EXPORT real exp(real a);
	friend CUDA_EXPORT real log(real a);
	friend CUDA_EXPORT real sqrt(real a);
	friend CUDA_EXPORT real pow(real a, real b);
	friend CUDA_EXPORT real operator+(real_type a, real b);
	friend CUDA_EXPORT real operator-(real_type a, real b);
	friend CUDA_EXPORT real operator*(real_type a, real b);
	friend CUDA_EXPORT real operator/(real_type a, real b);
	friend CUDA_EXPORT real erf(real a);
};
//
CUDA_EXPORT inline real operator+(real_type a, real b) {
	return real(a + b.r);
}

CUDA_EXPORT inline real operator*(real_type a, real b) {
	return real(a * b.r);
}

CUDA_EXPORT inline real operator-(real_type a, real b) {
	return real(a - b.r);
}

CUDA_EXPORT inline real operator/(real_type a, real b) {
	return real(a / b.r);
}

CUDA_EXPORT CUDA_EXPORT inline real copysign(real a, real b) {
	return real(std::copysign(a.r, b.r));
}

CUDA_EXPORT inline real pow(real a, real b) {
	return real(std::pow(a.r, b.r));
}

CUDA_EXPORT inline real exp(real a) {
	return real(std::exp(a.r));
}

CUDA_EXPORT inline real log(real a) {
	return real(std::log(a.r));
}

CUDA_EXPORT inline real sqrt(real a) {
	return real(std::sqrt(a.r));
}

CUDA_EXPORT inline real erf(real a) {
	return real(std::erf(a.r));
}

CUDA_EXPORT inline real sin(real a) {
	return real(std::sin(a.r));
}

CUDA_EXPORT inline real cos(real a) {
	return real(std::cos(a.r));
}

CUDA_EXPORT inline real abs(real a) {
	return real(std::abs(a.r));
}

inline real max(real a, real b) {
	return real(std::max(a.r, b.r));
}

inline real min(real a, real b) {
	return real(std::min(a.r, b.r));
}

#endif /* REAL_HPP_ */
