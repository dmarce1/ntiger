#include <hpx/hpx_init.hpp>
#include <ntiger/gravity.hpp>
#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>

void yield_to_hpx() {
	hpx::this_thread::yield();
}

hpx::id_type root;

fixed_real solve_gravity(fixed_real t, fixed_real dt) {
	static const auto opts = options::get();
	tree::compute_mass_attributes_action()(root);
	if (opts.ewald) {
		tree::set_ewald_sources(tree::gather_ewald_sources_action()(root));
	}
	return tree::compute_gravity_action()(root, std::vector < hpx::id_type > (1, root), pinned_vector<source>(), t, dt);
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
		tree::apply_boost_action()(root, -s.momentum);
	}
//	printf( "redistributed\n");
	tree::redistribute_workload_action()(root, 0, tree::compute_workload_action()(root));
//	printf( "Set selfand parent\n");
	tree::set_self_and_parent_action()(root, root, hpx::invalid_id);
}

void init(fixed_real t, bool t0) {
	static const auto opts = options::get();
	tree::set_self_and_parent_action()(root, root, hpx::invalid_id);

}

void write_checkpoint(int i, fixed_real t) {
	tree::write_silo_action()(root, i + 1, t);
}

auto statistics() {
	return tree::tree_statistics_action()(root);
}

int hpx_main(int argc, char *argv[]) {
	fixed_real t = 0.0;
	options opts;
	opts.process_options(argc, argv);
	init_ewald();
	range box;
	for (int dim = 0; dim < NDIM; dim++) {
		box.min[dim] = -opts.grid_size / 2.0;
		box.max[dim] = +opts.grid_size / 2.0;
	}
	root = hpx::new_ < tree > (hpx::find_here(), 1, std::vector<particle>(), box).get();
	init(t, 0.0);
	const auto localities = hpx::find_all_localities();
	for (int i = 0; i < 100; i++) {
//		printf("%i\n", i);
		auto parts = get_initial_particles(opts.problem, opts.problem_size / 100.0);
		tree::find_home_action()(root, std::move(parts));
		tree::finish_drift_action()(root);
		if (localities.size()) {
			tree::redistribute_workload_action()(root, 0, tree::compute_workload_action()(root));
		}
		tree::set_self_and_parent_action()(root, root, hpx::invalid_id);
	}

//	abort();
//	while (true) {
//	}
	printf("Initial load balance\n");
	drift(t, 0.0);
	printf("Initial gravity solve\n");
	fixed_real dt = solve_gravity(t, 0.0);
	if (opts.problem == "plummer") {
		tree::virialize_action()(root);
	} else if (opts.problem == "toomre") {
		tree::keplerize_action()(root);
	}

	printf("Start writing\n");
	write_checkpoint(0, t);
	printf("Done writing\n");
	int oi = 0;
	int i = 0;
	fixed_real last_output = 0.0;
	while (t < fixed_real(opts.tmax)) {
		//	printf("statistics\n");
		auto s = statistics();

		printf("Step = %i t = %13.6e  dt = %13.6e Nparts = %i Nleaves = %i Max Level = %i  Momentum = ", i, double(t), double(dt), s.nparts, s.nleaves,
				s.max_level);
		for (int dim = 0; dim < NDIM; dim++) {
			printf("%13.6e ", s.momentum[dim].get());
		}
		printf("ek = %13.6e ep = %13.6e ev = %13.6e verr = %13.6e etot = %13.6e\n", s.ek.get(), s.ep.get(), s.ev.get(),
				s.ev.get() / (std::abs(s.ep.get()) + 1.0e-100), s.ek.get() + s.ep.get());
//		printf("gravity\n");
//		solve_gravity(t, dt, true);
//		printf("drift\n");
		drift(t, dt);
//		printf("gravity\n");
		const auto new_dt = solve_gravity(t, dt);
//		printf("rescale\n");
		t += dt;
		if (int((last_output / fixed_real(opts.output_freq))) != int(((t / fixed_real(opts.output_freq))))) {
			last_output = t;
			write_checkpoint(++oi, t);
//			printf("output %i\n", oi);
		}
//		printf("timestep\n");
//		dt = timestep(t);
		dt = new_dt;
		i++;
	}
	printf("Exiting\n");
	FILE *fp = fopen("profile.txt", "wt");
	profiler_output(fp);
	fclose(fp);
	return hpx::finalize();

}

int main(int argc, char *argv[]) {
	std::vector < std::string > cfg = { "hpx.commandline.allow_unknown=1" };

	hpx::init(argc, argv, cfg);
}
