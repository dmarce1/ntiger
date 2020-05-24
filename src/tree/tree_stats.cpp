/*
 * tree_stats.cpp
 *
 *  Created on: Dec 8, 2019
 *      Author: dmarce1
 */

#include <cstdio>
#include <ntiger/tree_stats.hpp>

void tree_stats::print() const {
	printf("Tree Statistics\n");
	printf("N Particles   = %i\n", nparts);
	printf("N Nodes       = %i\n", nnodes);
	printf("N Leaves      = %i\n", nleaves);
	printf("Maximum Level = %i\n", max_level);
}
