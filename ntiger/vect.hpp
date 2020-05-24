/*
 * general_vect.hpp
 *
 *  Created on: Nov 30, 2019
 *      Author: dmarce1
 */

#ifndef VECT_HPP_
#define VECT_HPP_

#include "dim.hpp"
#include "real.hpp"

#include <array>
#include <atomic>
#include <cmath>

template<class T, int N>
class general_vect {
	std::array<T, N> v;
public:
	general_vect() = default;
	general_vect(std::array<real, N> a);
	general_vect(T a);
	T& operator[](int i);
	T operator[](int i) const;
	general_vect operator-() const;
	general_vect operator-(const general_vect &other) const;
	general_vect operator+(const general_vect &other) const;
	general_vect operator*(T r) const;
	general_vect operator/(T r) const;
	bool operator<(const general_vect &other) const;
	bool operator<=(const general_vect &other) const;
	bool operator>(const general_vect &other) const;
	bool operator>=(const general_vect &other) const;
	bool operator==(const general_vect &other) const;
	bool operator!=(const general_vect &other) const;
	T dot(const general_vect &other) const;
	template<class Arc>
	void serialize(Arc &&a, unsigned) {
		a & v;
	}

};

template<class T, int N>
bool inline general_vect<T, N>::operator<(const general_vect &other) const {
	for (int n = 0; n < N; n++) {
		if ((*this)[n] < other[n]) {
			return true;
		} else if ((*this)[n] > other[n]) {
			return false;
		}
	}
	return false;
}

template<class T, int N>
bool inline general_vect<T, N>::operator<=(const general_vect &other) const {
	return *this < other || *this == other;
}

template<class T, int N>
bool inline general_vect<T, N>::operator>(const general_vect &other) const {
	return !(*this <= other);
}

template<class T, int N>
bool inline general_vect<T, N>::operator>=(const general_vect &other) const {
	return !(*this < other);
}

template<class T, int N>
inline general_vect<T, N>::general_vect(std::array<real, N> a) :
		v(a) {
}

template<class T, int N>
inline general_vect<T, N>::general_vect(T a) {
	for (int i = 0; i < N; i++) {
		v[i] = a;
	}
}

template<class T, int N>
inline bool general_vect<T, N>::operator==(const general_vect<T, N> &other) const {
	for (int dim = 0; dim < NDIM; dim++) {
		if ((*this)[dim] != other[dim]) {
			return false;
		}
	}
	return true;
}

template<class T, int N>
inline bool general_vect<T, N>::operator!=(const general_vect<T, N> &other) const {
	return !((*this) == other);
}

template<class T, int N>
inline T& general_vect<T, N>::operator[](int i) {
	return v[i];
}

template<class T, int N>
inline T general_vect<T, N>::operator[](int i) const {
	return v[i];
}

template<class T, int N>
inline general_vect<T, N> general_vect<T, N>::operator-() const {
	general_vect<T, N> result;
	for (int dim = 0; dim < N; dim++) {
		result[dim] = -v[dim];
	}
	return result;
}

template<class T, int N>
inline general_vect<T, N> general_vect<T, N>::operator-(const general_vect<T, N> &other) const {
	general_vect<T, N> result;
	for (int dim = 0; dim < N; dim++) {
		result[dim] = v[dim] - other[dim];
	}
	return result;
}

template<class T, int N>
inline general_vect<T, N> general_vect<T, N>::operator+(const general_vect<T, N> &other) const {
	general_vect<T, N> result;
	for (int dim = 0; dim < N; dim++) {
		result[dim] = v[dim] + other[dim];
	}
	return result;
}

template<class T, int N>
inline general_vect<T, N> general_vect<T, N>::operator*(T r) const {
	general_vect<T, N> result;
	for (int dim = 0; dim < N; dim++) {
		result[dim] = v[dim] * r;
	}
	return result;
}

template<class T, int N>
inline general_vect<T, N> general_vect<T, N>::operator/(T r) const {
	general_vect<T, N> result;
	for (int dim = 0; dim < N; dim++) {
		result[dim] = v[dim] / r;
	}
	return result;
}

template<class T, int N>
inline T general_vect<T, N>::dot(const general_vect<T, N> &other) const {
	T result = 0.0;
	for (int dim = 0; dim < N; dim++) {
		result += v[dim] * other[dim];
	}
	return result;
}

template<class T, int N>
inline T abs(const general_vect<T, N> &v) {
	return sqrt(v.dot(v));
}

template<class T, int N>
inline general_vect<T, N> abs(const general_vect<T, N> &a, const general_vect<T, N> &b) {
	general_vect<T, N> c;
	for (int i = 0; i < N; i++) {
		c[i] = abs(a[i] - b[i]);
	}
	return c;
}

template<class T, int N>
inline general_vect<T, N> max(const general_vect<T, N> &a, const general_vect<T, N> &b) {
	general_vect<T, N> c;
	for (int i = 0; i < N; i++) {
		c[i] = max(a[i], b[i]);
	}
	return c;
}

template<class T, int N>
inline general_vect<T, N> min(const general_vect<T, N> &a, const general_vect<T, N> &b) {
	general_vect<T, N> c;
	for (int i = 0; i < N; i++) {
		c[i] = min(a[i], b[i]);
	}
	return c;
}

using vect = general_vect<real, NDIM>;
using atomic_vect = general_vect<std::atomic<real>, NDIM>;

#endif /* VECT_HPP_ */
