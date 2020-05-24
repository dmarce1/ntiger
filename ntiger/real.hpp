/*
 * real.hpp
 *
 *  Created on: Nov 29, 2019
 *      Author: dmarce1
 */

#ifndef REAL_HPP_
#define REAL_HPP_

#include <limits>
#include <algorithm>
#include <cmath>

using real_type = double;

class real {
	real_type r;
public:
	constexpr real() :
			r(std::numeric_limits<real_type>::signaling_NaN()) {
	}
	constexpr real(real_type a) :
			r(a) {
	}
	double get() const {
		return r;
	}
	inline real operator+(const real &b) const {
		return real(r + b.r);
	}
	inline real operator*(const real &b) const {
		return real(r * b.r);
	}
	inline real operator/(const real &b) const {
		return real(r / b.r);
	}
	inline real operator-(const real &b) const {
		return real(r - b.r);
	}
	inline real& operator+=(const real &b) {
		r += b.r;
		return *this;
	}
	inline real& operator-=(const real &b) {
		r -= b.r;
		return *this;
	}
	inline real& operator*=(const real &b) {
		r *= b.r;
		return *this;
	}
	inline real& operator/=(const real &b) {
		r /= b.r;
		return *this;
	}
	inline real operator-() const {
		return real(-r);
	}
	inline real operator+() const {
		return *this;
	}
	inline bool operator<(const real &a) const {
		return r < a.r;
	}
	inline bool operator>(const real &a) const {
		return r > a.r;
	}
	inline bool operator<=(const real &a) const {
		return r <= a.r;
	}
	inline bool operator>=(const real &a) const {
		return r >= a.r;
	}
	inline bool operator==(const real &a) const {
		return r == a.r;
	}
	inline bool operator!=(const real &a) const {
		return r != a.r;
	}
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & r;
	}
	static constexpr real max() {
		return std::numeric_limits<real_type>::max();
	}
	static constexpr real min() {
		return std::numeric_limits<real_type>::min();
	}
	static constexpr real eps() {
		return std::numeric_limits<real_type>::epsilon();
	}
	friend real copysign(real a, real b);
	friend real max(real a, real b);
	friend real min(real a, real b);
	friend real sin(real a);
	friend real cos(real a);
	friend real abs(real a);
	friend real exp(real a);
	friend real log(real a);
	friend real sqrt(real a);
	friend real pow(real a, real b);
	friend real operator+(real_type a, real b);
	friend real operator-(real_type a, real b);
	friend real operator*(real_type a, real b);
	friend real operator/(real_type a, real b);
};

inline real operator+(real_type a, real b) {
	return real(a + b.r);
}

inline real operator*(real_type a, real b) {
	return real(a * b.r);
}

inline real operator-(real_type a, real b) {
	return real(a - b.r);
}

inline real operator/(real_type a, real b) {
	return real(a / b.r);
}

inline real copysign(real a, real b) {
	return real(std::copysign(a.r, b.r));
}

inline real pow(real a, real b) {
	return real(std::pow(a.r, b.r));
}

inline real exp(real a) {
	return real(std::exp(a.r));
}

inline real log(real a) {
	return real(std::log(a.r));
}

inline real sqrt(real a) {
	return real(std::sqrt(a.r));
}

inline real sin(real a) {
	return real(std::sin(a.r));
}

inline real cos(real a) {
	return real(std::cos(a.r));
}

inline real abs(real a) {
	return real(std::abs(a.r));
}

inline real max(real a, real b) {
	return real(std::max(a.r, b.r));
}

inline real min(real a, real b) {
	return real(std::min(a.r, b.r));
}

#endif /* REAL_HPP_ */
