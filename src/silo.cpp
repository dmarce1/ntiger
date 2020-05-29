/*
 * silo.cpp
 *
 *  Created on: Dec 6, 2019
 *      Author: dmarce1
 */

#include <cstdio>
#include <silo.h>
#include <ntiger/options.hpp>
#include <ntiger/math.hpp>
#include <ntiger/particle.hpp>
#include <ntiger/math.hpp>

int main(int argc, char *argv[]) {
	const int silo_data_type = sizeof(real_type) == 8 ? DB_DOUBLE : DB_FLOAT;
	static auto opts = options::get();
	FILE *fp = fopen(argv[1], "rb");
	if (fp == NULL) {
		printf("File not found\n");
	} else {
		std::vector<particle> parts;
		particle p;
		fixed_real tm;
		std::string filename = argv[1];
		filename += ".silo";
		int cnt = 0;
		cnt += fread(&tm, sizeof(fixed_real), 1, fp);
		opts.set(opts);
		while (p.read(fp)) {
			parts.push_back(p);
		}
		fclose(fp);
		auto optlist = DBMakeOptlist(2);
		float ftime = (float) (double) tm;
		double dtime = tm;
		DBAddOption(optlist, DBOPT_TIME, &ftime);
		DBAddOption(optlist, DBOPT_DTIME, &dtime);
		DBfile *db = DBCreateReal(filename.c_str(), DB_CLOBBER, DB_LOCAL, "Meshless", DB_HDF5);
		const int nnodes = parts.size();
		real_type *coords[NDIM];
		for (int dim = 0; dim < NDIM; dim++) {
			coords[dim] = new real_type[nnodes];
			for (int i = 0; i < nnodes; i++) {
				coords[dim][i] = parts[i].x[dim].get();
			}
		}

		DBPutPointmesh(db, "points", NDIM, coords, nnodes, silo_data_type, optlist);

		std::vector<real_type> t;
		std::vector<real_type> dt;
		std::array<std::vector<real_type>, NDIM> vel;
		std::array<std::vector<real_type>, NDIM> g;
		t.reserve(nnodes);
		dt.reserve(nnodes);
		for (int dim = 0; dim < NDIM; dim++) {
			vel[dim].reserve(nnodes);
			g[dim].reserve(nnodes);
		}
		for (const auto &pi : parts) {
			t.push_back(double(pi.t));
			dt.push_back(double(pi.dt));
			std::array<vect, NDIM> E;
			for (int dim = 0; dim < NDIM; dim++) {
				vel[dim].push_back(pi.v[dim].get());
				g[dim].push_back(pi.g[dim].get());
			}
		}
		DBPutPointvar1(db, "t", "points", t.data(), nnodes, silo_data_type, optlist);
		DBPutPointvar1(db, "dt", "points", dt.data(), nnodes, silo_data_type, optlist);
		for (int dim = 0; dim < NDIM; dim++) {
			std::string nm = std::string() + "v_" + char('x' + char(dim));
			DBPutPointvar1(db, nm.c_str(), "points", vel[dim].data(), nnodes, silo_data_type, optlist);
		}
		for (int dim = 0; dim < NDIM; dim++) {
			std::string nm = std::string() + "g_" + char('x' + char(dim));
			DBPutPointvar1(db, nm.c_str(), "points", g[dim].data(), nnodes, silo_data_type, optlist);
		}
		for (int dim = 0; dim < NDIM; dim++) {
			delete[] coords[dim];
		}
		DBFreeOptlist(optlist);
		DBClose(db);

	}

}
