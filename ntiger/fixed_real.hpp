/*
 * fixed_real.hpp
 *
 *  Created on: Dec 28, 2019
 *      Author: dmarce1
 */

#ifndef OCTOPART_FIXED_REAL_HPP_
#define OCTOPART_FIXED_REAL_HPP_

#include <limits>
#include <hpx/config.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <ntiger/real.hpp>

class fixed_real {
	std::int64_t i;
	using int128_t = boost::multiprecision::int128_t;
	static constexpr auto one = real_type(std::int64_t(1) << std::int64_t(32));

public:

	inline fixed_real() = default;

	inline static fixed_real max() {
		fixed_real m;
		m.i = std::int64_t(1) << std::int64_t(62);
		return m;
	}
	inline static fixed_real min() {
		fixed_real m;
		m.i = 1;
		return m;
	}

	fixed_real next_bin() {
		if (i != 0) {
			std::int64_t m = i;
			std::int64_t l = 0;
			while ((m & 1) == 0) {
				//			printf( "!\n");
				m >>= std::int64_t(1);
				l++;
			}
			fixed_real n;
			n.i = i + (std::int64_t(1) << l);
			return n;
		} else {
			fixed_real n;
			n.i = std::int64_t(1) << 62;
			return n;
		}
	}

	std::int64_t get_int() const {
		return i;
	}

	fixed_real nearest_log2() const {
		if (i == std::int64_t(0)) {
			printf("Log of zero requested\n");
			abort();
		}
		fixed_real m;
		m.i = i;
		int lev = 0;
		while (m.i != 1) {
			m.i >>= std::int64_t(1);
			lev++;
			if (lev > 64) {
				printf("Nearest log 2 exceeds range\n");
				abort();
			}
		}
		m.i = std::int64_t(1) << lev;
		return m;
	}

	fixed_real half() const {
		fixed_real h;
		h.i = i / 2;
		return h;
	}

	fixed_real twice() const {
		fixed_real d;
		d.i = i * 2;
		return d;
	}

	void from_bin_and_level(int bin, int level) {
		i = bin * (1 << level);
	}

	void to_bin_and_level(int &bin, int &level) const {
		level = 0;
		bin = i;
		int m = i;
		while (m % 2 == 0 && level != 64) {
			m >>= 1;
			level++;
			bin /= 2;
		}
	}

	inline fixed_real(const fixed_real &other) {
		i = other.i;
	}

	inline fixed_real(fixed_real &&other) {
		i = other.i;
	}

	inline fixed_real& operator=(const fixed_real &other) {
		i = other.i;
		return *this;
	}

	inline fixed_real& operator=(fixed_real &&other) {
		i = other.i;
		return *this;
	}

	inline fixed_real& operator=(real_type other) {
		i = std::int64_t(other * real_type(one));
		return *this;
	}

	inline fixed_real(real_type other) {
		i = std::int64_t(other * real_type(one));
	}

	inline bool operator<(const fixed_real &other) const {
		return i < other.i;
	}

	inline bool operator>(const fixed_real &other) const {
		return i > other.i;
	}

	inline bool operator<=(const fixed_real &other) const {
		return i <= other.i;
	}

	inline bool operator>=(const fixed_real &other) const {
		return i >= other.i;
	}

	inline bool operator==(const fixed_real &other) const {
		return i == other.i;
	}

	inline bool operator!=(const fixed_real &other) const {
		return i != other.i;
	}

	inline fixed_real operator+(const fixed_real &other) const {
		fixed_real rc;
		rc.i = i + other.i;
		return rc;
	}

	inline fixed_real operator-(const fixed_real &other) const {
		fixed_real rc;
		rc.i = i - other.i;
		return rc;
	}

	inline fixed_real operator*(const fixed_real &other) const {
		fixed_real rc;
		rc.i = std::int64_t((int128_t(i) * int128_t(other.i)) >> 32);
		return rc;
	}

	inline fixed_real operator/(const fixed_real &other) const {
		fixed_real rc;
		rc.i = std::int64_t((int128_t(i) << 32) / int128_t(other.i));
		return rc;
	}

	inline fixed_real& operator+=(const fixed_real &other) {
		i += other.i;
		return *this;
	}

	inline fixed_real& operator-=(const fixed_real &other) {
		i -= other.i;
		return *this;
	}

	inline fixed_real& operator*=(const fixed_real &other) {
		i = std::int64_t(int128_t(i) * int128_t(other.i) >> 32);
		return *this;
	}

	inline fixed_real operator/=(const fixed_real &other) {
		i = std::int64_t((int128_t(i) << 32) / int128_t(other.i));
		return *this;
	}

	inline operator real_type() const {
		return real_type(i) / one;
	}

	inline operator int() const {
		return i / one;
	}

	template<class Arc>
	void serialize(Arc &&a, unsigned) {
		a & i;
	}

	friend fixed_real max(const fixed_real &a, const fixed_real &b);
	friend fixed_real min(const fixed_real &a, const fixed_real &b);

};

inline fixed_real max(const fixed_real &a, const fixed_real &b) {
	if (a.i > b.i) {
		return a;
	} else {
		return b;
	}
}

inline fixed_real min(const fixed_real &a, const fixed_real &b) {
	if (a.i < b.i) {
		return a;
	} else {
		return b;
	}
}

#endif /* OCTOPART_FIXED_REAL_HPP_ */
