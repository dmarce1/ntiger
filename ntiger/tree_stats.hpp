
#pragma once

#include <ntiger/vect.hpp>


struct tree_stats {
	int max_level;
	int nnodes;
	int nleaves;
	int nparts;
	real mass;
	real energy;
	vect momentum;
	void print() const;
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & max_level;
		arc & nnodes;
		arc & nleaves;
		arc & nparts;
		arc & mass;
		arc & energy;
		arc & momentum;
	}
};
