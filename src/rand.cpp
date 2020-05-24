/*
 * rand.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: dmarce1
 */


#include <cstdlib>

#include <ntiger/rand.hpp>

real rand_unit_box() {
	return (rand() + 0.5) / (RAND_MAX + 1.0) - 0.5;
}
