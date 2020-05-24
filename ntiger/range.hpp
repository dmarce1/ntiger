/*
 * math.hpp
 *
 *  Created on: Nov 29, 2019
 *      Author: dmarce1
 */

#ifndef RANGE_HPP_
#define RANGE_HPP_

#include "vect.hpp"
#include <limits>

struct range {
	vect min;
	vect max;
	template<class Arc>
	void serialize(Arc& arc,unsigned) {
		arc & min;
		arc & max;
	}
};

range reflect_range(const range&, int dim, real axis);
vect range_center(const range &r);
range shift_range(const range& r, const vect&);
vect range_span(const range&);
bool in_range(const vect&, const range&);
bool in_range(const range&, const range&);
bool ranges_intersect(const range&, const range&);
range range_around(const vect&, real);
range expand_range(const range&, real);
real range_volume(const range&);
range null_range();
bool operator==(const range&, const range&);
bool operator!=(const range&, const range&);

#endif /* MATH_HPP_ */

