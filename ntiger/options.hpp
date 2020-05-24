#pragma once
#include <ntiger/real.hpp>
#include <string>

class options {
public:
	std::string config_file;
	std::string checkpoint;
	bool fpe;
	bool global_time;
	bool gravity;
	int parts_per_node;
	int problem_size;
	double fgamma;
	double kep_eps;
	double theta;
	double tmax;
	double cfl;
	double output_freq;
	double grid_size;
	std::string problem;

	template<class Arc>
	void serialize(Arc &arc, unsigned) {
		arc & cfl;
		arc & kep_eps;
		arc & checkpoint;
		arc & config_file;
		arc & fpe;
		arc & global_time;
		arc & gravity;
		arc & parts_per_node;
		arc & problem_size;
		arc & fgamma;
		arc & theta;
		arc & tmax;
		arc & problem;
		arc & output_freq;
		arc & grid_size;
	}
	static options global;
	static options& get();
	static void set(options);
	bool process_options(int argc, char *argv[]);
};
