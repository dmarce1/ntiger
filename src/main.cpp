#include <hpx/hpx_init.hpp>
#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>

hpx::id_type root;

void solve_gravity(fixed_real t, fixed_real dt) {
	static const auto opts = options::get();
	if (opts.gravity) {
		tree::compute_mass_attributes_action()(root);
		tree::compute_gravity_action()(root, std::vector < hpx::id_type > (1, root), std::vector<mass_attr>(), t, dt, false);
	}
	if (opts.problem == "kepler" || opts.problem == "rt" || opts.gravity) {
		tree::apply_gravity_action()(root, t, dt);
	}
}

void drift(fixed_real t, fixed_real dt) {
	tree::compute_drift_action()(root, dt);
	tree::finish_drift_action()(root);
	const auto s = tree::tree_statistics_action()(root);
	tree::compute_workload_action()(root);
	tree::redistribute_workload_action()(root, 0, s.nparts);
	tree::set_self_and_parent_action()(root, root, hpx::invalid_id);
	tree::form_tree_action()(root, std::vector < hpx::id_type > (1, root), true);
	tree::compute_interactions_action()(root);
}

void rescale() {
	const auto new_scale = tree::compute_scale_factor_action()(root);
	if (new_scale > 1.0) {
		printf("Re-scaling by %e\n", new_scale.get());
		tree::rescale_action()(root, new_scale, range());
		tree::finish_drift_action()(root);
		tree::send_lost_parts_action()(root, std::vector<particle>());
		const auto s = tree::tree_statistics_action()(root);
		tree::compute_workload_action()(root);
		tree::redistribute_workload_action()(root, 0, s.nparts);
		tree::set_self_and_parent_action()(root, root, hpx::invalid_id);
		tree::form_tree_action()(root, std::vector < hpx::id_type > (1, root), true);
		tree::compute_interactions_action()(root);
	}

}

void init(fixed_real t, bool t0) {
	static const auto opts = options::get();
	tree::set_self_and_parent_action()(root, root, hpx::invalid_id);
	tree::form_tree_action()(root, std::vector < hpx::id_type > (1, root), true);
	tree::compute_interactions_action()(root);
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
	solve_gravity(t, 0.0);
//	tree::get_neighbor_particles_action()(root, tree::PRIMITIVE);
//	tree::set_drift_velocity_action()(root, t);
	fixed_real dt = timestep(t);
	write_checkpoint(0, t);
	int oi = 0;
	int i = 0;
	fixed_real last_output = 0.0;
	while (t < fixed_real(opts.tmax)) {
		auto s = statistics();

		printf("Step = %i t = %e  dt = %e Nparts = %i Nleaves = %i Max Level = %i Mass = %e Momentum = ", i, double(t), double(dt), s.nparts, s.nleaves,
				s.max_level, s.mass.get());
		for (int dim = 0; dim < NDIM; dim++) {
			printf("%e ", s.momentum[dim].get());
		}
		printf("ek = %e ep = %e ev = %e verr = %e etot = %e\n", s.ek.get(), s.ep.get(), s.ev.get(), s.ev.get() / s.ep.get(), s.ev.get() + s.ep.get());
		drift(t, dt / fixed_real(2));
		//	rescale();
		solve_gravity(t, dt);
		drift(t, dt / fixed_real(2));
		t += dt;
		tree::advance_time_action()(root, t);
		if (int((last_output / fixed_real(opts.output_freq))) != int(((t / fixed_real(opts.output_freq))))) {
			last_output = t;
			write_checkpoint(++oi, t);
			printf("output %i\n", oi);
		}
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
