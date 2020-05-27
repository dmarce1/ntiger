#include <hpx/hpx_init.hpp>
#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>

hpx::id_type root;

void solve_gravity(fixed_real t, fixed_real dt, bool first_kick) {
	static const auto opts = options::get();
	if (opts.gravity && !first_kick) {
//		printf( "Multipoles\n" );
		tree::compute_mass_attributes_action()(root);
//		printf( "Interactions\n" );
		tree::compute_gravity_action()(root, std::vector < hpx::id_type > (1, root), std::vector<mass_attr>(), t, dt, false);
	}
//	printf( "Applying\n" );
	if (opts.problem == "kepler" || opts.problem == "rt" || opts.gravity) {
		tree::apply_gravity_action()(root, t, dt, first_kick);
	}
}

void drift(fixed_real t, fixed_real dt) {
	
	const auto opts = options::get();
//	printf( "Drift1\n" );
	tree::compute_drift_action()(root, dt);
//	printf( "Drift2\n" );
	tree::finish_drift_action()(root);
	if (opts.ewald && dt != fixed_real(0.0)) {
//		printf( "statistics\n" );
		const auto s = tree::tree_statistics_action()(root);
//		printf( "boost\n" );
		tree::apply_boost_action()(root, -s.momentum/s.mass);
	}
//	printf( "redistributed\n");
	tree::redistribute_workload_action()(root, 0, tree::compute_workload_action()(root));
//	printf( "Set selfand parent\n");
	tree::set_self_and_parent_action()(root, root, hpx::invalid_id);
}

void rescale() {
	const auto new_scale = tree::compute_scale_factor_action()(root);
	if (new_scale > 1.0) {
//		printf("Re-scaling by %13.6e\n", new_scale.get());
		tree::rescale_action()(root, new_scale, range());
		tree::compute_drift_action()(root, 0.0);
		tree::finish_drift_action()(root);
		tree::send_lost_parts_action()(root, std::vector<particle>());
		const auto s = tree::tree_statistics_action()(root);
		tree::redistribute_workload_action()(root, 0, tree::compute_workload_action()(root));
		tree::set_self_and_parent_action()(root, root, hpx::invalid_id);
	}

}

void init(fixed_real t, bool t0) {
	static const auto opts = options::get();
	tree::set_self_and_parent_action()(root, root, hpx::invalid_id);

}

void write_checkpoint(int i, fixed_real t) {
	tree::write_silo_action()(root, i + 1, t);
}

fixed_real timestep(fixed_real t) {
	static const auto opts = options::get();
	fixed_real dt = tree::compute_timestep_action()(root, t);
	return dt;
}

auto statistics() {
	return tree::tree_statistics_action()(root);
}

int hpx_main(int argc, char *argv[]) {
	fixed_real t = 0.0;
	options opts;
	opts.process_options(argc, argv);
	std::vector<particle> parts;
	bool t0;
	if (opts.checkpoint != "") {
		printf("Reading %s\n", opts.checkpoint.c_str());
		FILE *fp = fopen(opts.checkpoint.c_str(), "rb");
		if (fp == NULL) {
			printf("Could not find %s\n", opts.checkpoint.c_str());
			abort();
		} else {
			particle p;
			real dummy;
			int cnt = fread(&dummy, sizeof(real), 1, fp);
			cnt += fread(&t, sizeof(real), 1, fp);
			if (cnt == 0) {
				printf("Empty checkpoint\n");
				return hpx::finalize();
			}
			while (p.read(fp)) {
				parts.push_back(p);
			}
			fclose(fp);
		}
		t0 = false;
	} else {
		parts = get_initial_particles(opts.problem, opts.problem_size);
	}
	range box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.min[dim] = -opts.grid_size / 2.0;
		box.max[dim] = +opts.grid_size / 2.0;
	}
	root = hpx::new_ < tree > (hpx::find_here(), std::move(parts), box, null_range()).get();
	init(t, t0);
	printf("Initial load balance\n" );
	drift(t, 0.0);
	printf("Initial gravity solve\n" );
	solve_gravity(t, 0.0, false);
	if (opts.problem == "plummer") {
		tree::virialize_action()(root);
	} else if (opts.problem == "toomre") {
		tree::keplerize_action()(root);
	}
	printf( "Time-step\n");
	fixed_real dt = timestep(t);
	printf( "Start writing\n");
	write_checkpoint(0, t);
	printf( "Done writing\n");
	int oi = 0;
	int i = 0;
	fixed_real last_output = 0.0;
	while (t < fixed_real(opts.tmax)) {
		printf("statistics\n");
		auto s = statistics();

		printf("Step = %i t = %13.6e  dt = %13.6e Nparts = %i Nleaves = %i Max Level = %i Mass = %13.6e Momentum = ", i, double(t), double(dt), s.nparts, s.nleaves,
				s.max_level, s.mass.get());
		for (int dim = 0; dim < NDIM; dim++) {
			printf("%13.6e ", s.momentum[dim].get());
		}
		printf("ek = %13.6e ep = %13.6e ev = %13.6e verr = %13.6e etot = %13.6e\n", s.ek.get(), s.ep.get(), s.ev.get(), s.ev.get() / (std::abs(s.ep.get()) + 1.0e-100),
				s.ek.get() + s.ep.get());
		printf( "gravity\n" );
		solve_gravity(t, dt, true);
		printf( "drift\n" );
		drift(t, dt);
		printf( "gravity\n" );
		solve_gravity(t, dt, false);
		printf( "rescale\n" );
		rescale();
		t += dt;
		if (int((last_output / fixed_real(opts.output_freq))) != int(((t / fixed_real(opts.output_freq))))) {
			last_output = t;
			write_checkpoint(++oi, t);
			printf("output %i\n", oi);
		}
		printf( "timestep\n");
		dt = timestep(t);
		i++;
	}
	FILE *fp = fopen("profile.txt", "wt");
	profiler_output(fp);
	fclose(fp);
	return hpx::finalize();

}

int main(int argc, char *argv[]) {
	std::vector < std::string > cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}
