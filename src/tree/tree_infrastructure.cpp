#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>
#include <set>

HPX_REGISTER_COMPONENT(hpx::components::component<tree>, tree);

tree::tree() {
	mtx = std::make_shared<hpx::lcos::local::mutex>();
	nparts0 = 0;
	dead = false;
	leaf = false;
}

tree::tree(const std::vector<particle> &_parts, const std::array<hpx::id_type, NCHILD> &_children, const std::array<int, NCHILD> &_child_loads,
		const range &_root_box, const range &_box, bool _leaf) {
	mtx = std::make_shared<hpx::lcos::local::mutex>();
	parts = _parts;
	children = _children;
	child_loads = _child_loads;
	root_box = _root_box;
	box = _box;
	leaf = _leaf;
	dead = false;
	nparts0 = parts.size();
}

tree::tree(std::vector<particle> &&these_parts, const range &box_, const range &root_box_) :
		box(box_), nparts0(0), dead(false), root_box(root_box_) {
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
	if (root_box == null_range()) {
		root_box = box;
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
			futs[ci] = hpx::async < compute_workload_action > (children[ci]);
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
	leaf = false;
	nparts0 = 0;
	std::array<hpx::future<hpx::id_type>, NCHILD> futs;
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
		int this_sz = parts.size();
		std::vector<particle> child_parts;
		for (int i = 0; i < this_sz; i++) {
			auto &part = parts[i];
			if (in_range(part.x, child_box)) {
				child_parts.push_back(std::move(part));
				this_sz--;
				parts[i] = parts[this_sz];
				i--;
			}
		}
		parts.resize(this_sz);
		futs[ci] = hpx::async([child_box, this](std::vector<particle> child_parts) {
			return hpx::new_ < tree > (hpx::find_here(), std::move(child_parts), child_box, root_box).get();
		}, std::move(child_parts));
	}
	for (int ci = 0; ci < NCHILD; ci++) {
		children[ci] = futs[ci].get();
	}
}

std::vector<particle> tree::destroy() {
	self = hpx::invalid_id;
	siblings.clear();
	dead = true;
	return parts;
}

tree_attr tree::finish_drift() {
	static const auto opts = options::get();
	const auto npart_max = opts.parts_per_node;
	if (leaf) {
		parts.insert(parts.end(), new_parts.begin(), new_parts.end());
		new_parts.clear();
		nparts0 = parts.size();
		if (nparts0 > npart_max) {
			create_children();
		}
	} else {
		std::array<hpx::future<tree_attr>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < finish_drift_action > (children[ci]);
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
				dfuts[ci] = hpx::async < destroy_action > (children[ci]);
			}
			for (int ci = 0; ci < NCHILD; ci++) {
				const auto tmp = dfuts[ci].get();
				children[ci] = hpx::invalid_id;
				parts.insert(parts.end(), tmp.begin(), tmp.end());
			}
			nparts0 = parts.size();
			leaf = true;
		}
	}
	return get_attributes();
}

void tree::form_tree(std::vector<hpx::id_type> nids, bool clear_sibs) {
	static auto opts = options::get();
	std::vector < hpx::future < tree_attr >> nfuts(nids.size());
	std::vector < hpx::future<std::array<hpx::id_type, NCHILD>> > cfuts;
	std::vector<tree_attr> attrs(nids.size());
	std::vector < hpx::id_type > next_nids;
	std::set < hpx::id_type > used;
	for (int i = 0; i < nids.size(); i++) {
		assert(nids[i] != hpx::invalid_id);
		nfuts[i] = hpx::async < get_attributes_action > (nids[i]);
	}
	if (clear_sibs) {
		siblings.clear();
	}
	for (int i = 0; i < nids.size(); i++) {
		attrs[i] = nfuts[i].get();
	}
	for (int i = 0; i < nids.size(); i++) {
		const auto srange = attrs[i].box;
		if (ranges_intersect(srange, box) && srange != box) {
			if (attrs[i].leaf) {
				siblings.push_back( { nids[i], attrs[i].box});
			} else {
				if (used.find(nids[i]) == used.end() && nids[i] != self) {
					cfuts.push_back(hpx::async < get_children_action > (nids[i]));
					used.insert(nids[i]);
				}
			}
		}
	}
	if (leaf) {
		for (auto &f : cfuts) {
			const auto list = f.get();
			next_nids.insert(next_nids.end(), list.begin(), list.end());
		}
		if (nids.size()) {
			form_tree(std::move(next_nids), false);
		}
	} else {
		used.clear();
		for (int i = 0; i < siblings.size(); i++) {
			if (used.find(siblings[i].id) == used.end()) {
				used.insert(siblings[i].id);
				next_nids.push_back(std::move(siblings[i].id));
			}
		}
		siblings.clear();
		next_nids.insert(next_nids.end(), children.begin(), children.end());
		for (auto &f : cfuts) {
			const auto list = f.get();
			next_nids.insert(next_nids.end(), list.begin(), list.end());
		}
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < form_tree_action > (children[ci], next_nids, true);
		}
		hpx::wait_all(futs);
	}
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
	return children;
}

hpx::id_type tree::get_parent() const {
	return parent;
}


hpx::id_type tree::migrate(const hpx::id_type &loc) {
	auto id_fut = hpx::new_ < tree > (loc, parts, children, child_loads, root_box, box, leaf);
	auto id = id_fut.get();
	dead = true;
	return id;
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
				children[ci] = migrate_action()(children[ci], localities[loc_id]);
			}
			futs[ci] = hpx::async < redistribute_workload_action > (children[ci], current, total);
			current += child_loads[ci];
		}
		hpx::wait_all(futs);
	}
}

void tree::set_self_and_parent(const hpx::id_type s, const hpx::id_type p) {
	assert(!dead);
	self = std::move(s);
	parent = std::move(p);
	if (!leaf) {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			assert(children[ci] != hpx::invalid_id);
			auto this_p = children[ci];
			futs[ci] = hpx::async < set_self_and_parent_action > (children[ci], std::move(this_p), self);
		}
		hpx::wait_all(futs);
	}
}

tree_stats tree::tree_statistics() const {
	tree_stats stats;
	stats.max_level = 0;
	stats.nnodes = 1;
	stats.mass = 0.0;
	stats.energy = 0.0;
	stats.momentum = vect(0);
	if (leaf) {
		stats.nparts = parts.size();
		stats.nleaves = 1;
		for (const auto &p : parts) {
			stats.mass += p.m0;
			stats.energy += 0.0;
			stats.momentum = stats.momentum + p.vf * p.m0;
		}
	} else {
		stats.nparts = 0;
		stats.nleaves = 0;
		std::array<hpx::future<tree_stats>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < tree_statistics_action > (children[ci]);
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			tree_stats cstat = futs[ci].get();
			stats.max_level = std::max(stats.max_level, cstat.max_level + 1);
			stats.nleaves += cstat.nleaves;
			stats.nnodes += cstat.nnodes;
			stats.nparts += cstat.nparts;
			stats.energy += cstat.energy;
			stats.mass += cstat.mass;
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
		fwrite(&opts.fgamma, sizeof(real), 1, fp);
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
			write_checkpoint_action()(children[ci], filename, t);
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

