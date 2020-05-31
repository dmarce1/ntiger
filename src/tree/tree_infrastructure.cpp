#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>

#include  <hpx/lcos/when_all.hpp>
#include <hpx/lcos/local/mutex.hpp>

#include <set>

HPX_REGISTER_COMPONENT(hpx::components::component<tree>, tree);

tree::tree() {
	mtx = std::make_shared<hpx::lcos::local::mutex>();
	dead = false;
	leaf = false;
}

tree::tree(const std::vector<particle> &_parts, const std::array<node_attr, NCHILD> &_children, const std::array<int, NCHILD> &_child_loads, const range &_box,
		bool _leaf) {
	mtx = std::make_shared<hpx::lcos::local::mutex>();
	parts = _parts;
	children = _children;
	child_loads = _child_loads;
	box = _box;
	leaf = _leaf;
	dead = false;
}

tree::tree(std::vector<particle> &&these_parts, const range &box_) :
		box(box_), dead(false) {
	const int sz = these_parts.size();
	static const auto opts = options::get();
	const auto npart_max = opts.parts_per_node;

	mtx = std::make_shared<hpx::lcos::local::mutex>();

	/* Create initial box if root */
	if (box == null_range()) {
		for (const auto &part : these_parts) {
			box.min = min(box.min, part.x);
			box.max = max(box.max, part.x);
		}
		for (int dim = 0; dim < NDIM; dim++) {
			const auto dx = 1.0e-10 * (box.max[dim] - box.min[dim]);
			box.min[dim] -= dx;
			box.max[dim] += dx;
		}
	}
	if (sz > npart_max) {
		parts = std::move(these_parts);
		create_children();
	} else {
		leaf = true;
		parts = std::move(these_parts);
	}
}

int tree::compute_workload() {
	int load;
	if (leaf) {
		std::fill(child_loads.begin(), child_loads.end(), 0);
		load = parts.size();
	} else {
		load = 0;
		std::array<hpx::future<int>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < compute_workload_action > (children[ci].id);
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			const auto this_load = futs[ci].get();
			child_loads[ci] = this_load;
			load += this_load;
		}
	}
	return load;
}

void tree::create_children() {
	real max_span = 0.0;
	range boxl = box;
	range boxr = box;
	const int szl = parts.size() / 2;
	const int szr = parts.size() - szl;
	std::vector<particle> pl(szl), pr(szr);
	int max_dim = 0;
	for (int dim = 0; dim < NDIM; dim++) {
		const auto s = box.max[dim] - box.min[dim];
		if (s > max_span) {
			max_span = s;
			max_dim = dim;
		}
	}
	const real mid = sort_by_dimension(parts, max_dim);
	boxl.max[max_dim] = boxr.min[max_dim] = mid;
	children[0].box = boxl;
	children[1].box = boxr;
	std::move(parts.begin(), parts.begin() + szl, pl.begin());
	std::move(parts.begin() + szl, parts.end(), pr.begin());
	auto fl = hpx::async([boxl, this](std::vector<particle> pl) {
		return hpx::new_ < tree > (hpx::find_here(), std::move(pl), boxl).get();
	}, std::move(pl));
	auto fr = hpx::async([boxr, this](std::vector<particle> pr) {
		return hpx::new_ < tree > (hpx::find_here(), std::move(pr), boxr).get();
	}, std::move(pr));
	parts.resize(0);
	leaf = false;
	children[0].id = fl.get();
	children[1].id = fr.get();
}

std::vector<particle> tree::destroy() {
	self = hpx::invalid_id;
	dead = true;
	return parts;
}

void tree::find_home(const std::vector<particle> &pis) {
	std::vector<particle> self_parts;
	std::vector<particle> parent_parts;

	{
		PROFILE();
		for (int i = 0; i < pis.size(); i++) {
			const auto &pi = pis[i];
			if (in_range(pi.x, box)) {
				self_parts.push_back(pi);
			} else {
				parent_parts.push_back(pi);
			}

		}
	}
	hpx::future<void> self_fut;
	if (self_parts.size()) {
		if (leaf) {
			self_fut = hpx::async < send_particles_action > (self, std::move(self_parts));
		} else {
			std::array<hpx::future<void>, NCHILD> cfuts;
			for (int ci = 0; ci < NCHILD; ci++) {
				std::vector<particle> cparts;
				for (int i = 0; i < self_parts.size(); i++) {
					if (in_range(self_parts[i].x, children[ci].box)) {
						cparts.push_back(self_parts[i]);
						self_parts[i] = self_parts[self_parts.size() - 1];
						self_parts.resize(self_parts.size() - 1);
						i--;
					}
				}
				cfuts[ci] = hpx::async < find_home_action > (children[ci].id, std::move(cparts));
			}
			self_fut = hpx::when_all(cfuts);
		}
	} else {
		self_fut = hpx::make_ready_future<void>();
	}
	if (parent != hpx::invalid_id) {
		if (parent_parts.size()) {
			find_home_action()(parent, std::move(parent_parts));
		}
	} else if (parent_parts.size()) {
		const auto &p = parent_parts[0];
		printf("Lost a particle %f %f %f |%f %f %f %f %f %f |\n", p.x[0].get(), p.x[1].get(), p.x[2].get(),
				box.min[0].get(), box.max[0].get(),box.min[1].get(), box.max[1].get(),box.min[2].get(), box.max[2].get());
	}
	self_fut.get();
}

tree_attr tree::finish_drift() {
	static const auto opts = options::get();
	const auto npart_max = opts.parts_per_node;
	if (leaf) {
		parts.insert(parts.end(), new_parts.begin(), new_parts.end());
		new_parts.clear();
		if (parts.size() > npart_max) {
			create_children();
		}
	} else {
		std::array<hpx::future<tree_attr>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < finish_drift_action > (children[ci].id);
		}
		int cparts = 0;
		bool all_leaves = true;
		for (auto &f : futs) {
			const auto tmp = f.get();
			if (tmp.leaf) {
				cparts += tmp.nparts;
			} else {
				all_leaves = false;
			}
		}
		if (cparts <= npart_max && all_leaves) {
			std::array < hpx::future<std::vector<particle>>, NCHILD > dfuts;
			for (int ci = 0; ci < NCHILD; ci++) {
				dfuts[ci] = hpx::async < destroy_action > (children[ci].id);
			}
			for (int ci = 0; ci < NCHILD; ci++) {
				const auto tmp = dfuts[ci].get();
				children[ci].id = hpx::invalid_id;
				parts.insert(parts.end(), tmp.begin(), tmp.end());
			}
			leaf = true;
		}
	}
	return get_attributes();
}

tree_attr tree::get_attributes() const {
	tree_attr attr;
	attr.dead = dead;
	attr.leaf = leaf;
	attr.box = box;
	attr.nparts = parts.size();
	return attr;
}

std::array<hpx::id_type, NCHILD> tree::get_children() const {
	std::array < hpx::id_type, NCHILD > cids;
	for (int ci = 0; ci < NCHILD; ci++) {
		cids[ci] = children[ci].id;
	}
	return cids;
}

hpx::id_type tree::get_parent() const {
	return parent;
}

hpx::id_type tree::migrate(const hpx::id_type &loc) {
	auto id_fut = hpx::new_ < tree > (loc, parts, children, child_loads, box, leaf);
	auto id = id_fut.get();
	dead = true;
	return id;
}

void tree::rescale(real factor, range mybox) {
	if (parent == hpx::invalid_id) {
		box = scale_range(box, factor);
	} else {
		box = mybox;
	}
	std::array<hpx::future<void>, NCHILD> futs;
	if (!leaf) {
		range child_box;
		for (int ci = 0; ci < NCHILD; ci++) {
			int m = ci;
			for (int dim = 0; dim < NDIM; dim++) {
				const auto &b = box.min[dim];
				const auto &e = box.max[dim];
				const auto mid = (e + b) * 0.5;
				if (m & 1) {
					child_box.min[dim] = mid;
					child_box.max[dim] = e;
				} else {
					child_box.min[dim] = b;
					child_box.max[dim] = mid;
				}
				m >>= 1;
			}
			children[ci].box = child_box;
			futs[ci] = hpx::async < rescale_action > (children[ci].id, factor, child_box);
		}

		hpx::wait_all(futs);
	}
}

void tree::redistribute_workload(int current, int total) {
	if (!leaf) {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			static const auto localities = hpx::find_all_localities();
			const int loc_id = current * localities.size() / (total + 1);
			assert(loc_id >= 0);
			assert(loc_id < localities.size());
			if (localities[loc_id] != hpx::find_here()) {
				children[ci].id = migrate_action()(children[ci].id, localities[loc_id]);
			}
			futs[ci] = hpx::async < redistribute_workload_action > (children[ci].id, current, total);
			current += child_loads[ci];
		}
		hpx::wait_all(futs);
	}
}

void tree::send_particles(const std::vector<particle> &pj) {
	std::lock_guard < hpx::lcos::local::mutex > lock(*mtx);
	PROFILE();
	for (auto p : pj) {
		new_parts.push_back(std::move(p));
	}
}

void tree::set_self_and_parent(const hpx::id_type s, const hpx::id_type p) {
	assert(!dead);
	self = std::move(s);
	parent = std::move(p);
	if (!leaf) {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			assert(children[ci].id != hpx::invalid_id);
			auto this_p = children[ci].id;
			futs[ci] = hpx::async < set_self_and_parent_action > (children[ci].id, std::move(this_p), self);
		}
		hpx::wait_all(futs);
	}
}

tree_stats tree::tree_statistics() const {
	tree_stats stats;
	stats.max_level = 0;
	stats.nnodes = 1;
	stats.ek = 0.0;
	stats.ep = 0.0;
	stats.momentum = vect(0);
	stats.ev = 0.0;
	const real m = 1.0 / options::get().problem_size;
	if (leaf) {
		stats.nparts = parts.size();
		stats.nleaves = 1;
		for (const auto &p : parts) {
			const real ek = 0.5 * p.v.dot(p.v) * m;
			const real ep = 0.5 * p.phi * m;
			const real ev = 2.0 * ek + ep;
			stats.ek += ek;
			stats.ep += ep;
			stats.ev += ev;
			stats.momentum = stats.momentum + p.v * m;
		}
	} else {
		stats.nparts = 0;
		stats.nleaves = 0;
		std::array<hpx::future<tree_stats>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < tree_statistics_action > (children[ci].id);
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			tree_stats cstat = futs[ci].get();
			stats.max_level = std::max(stats.max_level, cstat.max_level + 1);
			stats.nleaves += cstat.nleaves;
			stats.nnodes += cstat.nnodes;
			stats.nparts += cstat.nparts;
			stats.ev += cstat.ev;
			stats.ep += cstat.ep;
			stats.ek += cstat.ek;
			stats.momentum = stats.momentum + cstat.momentum;
		}
	}
	return stats;
}

void tree::write_checkpoint(const std::string &filename, fixed_real t) const {
	FILE *fp;
	static const auto opts = options::get();
	if (parent == hpx::invalid_id) {
		fp = fopen(filename.c_str(), "wb");
		fwrite(&t, sizeof(fixed_real), 1, fp);
		fclose(fp);
	}
	if (leaf) {
		fp = fopen(filename.c_str(), "ab");
		for (const auto &part : parts) {
			part.write(fp);
		}
		fclose(fp);
	} else {
		for (int ci = 0; ci < NCHILD; ci++) {
			write_checkpoint_action()(children[ci].id, filename, t);
		}
	}
}

void tree::write_silo(int num, fixed_real t) const {
	std::string base_name = "Y" + std::to_string(num);
	std::string command = "./check2silo " + base_name;
	write_checkpoint(base_name, t);
	if (system(command.c_str()) != 0) {
		printf("Unable to convert checkpoint to SILO\n");
	}
}

