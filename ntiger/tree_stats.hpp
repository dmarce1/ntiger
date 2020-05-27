
#pragma once

#include <ntiger/vect.hpp>


struct tree_stats {
	int max_level;
	int nnodes;
	int nleaves;
	int nparts;
	real ek;
	real ep;
	real ev;;
	vect momentum;
	void print() const;
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & max_level;
		arc & nnodes;
		arc & nleaves;
		arc & nparts;
		arc & ek;
		arc & ep;
		arc & ev;
		arc & momentum;
	}
};
