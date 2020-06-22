#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>
#include <ntiger/checkitem.hpp>

#include  <hpx/lcos/when_all.hpp>
#include <hpx/lcos/local/mutex.hpp>

#include <set>
#include <ntiger/initialize.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>

#include  <hpx/lcos/when_all.hpp>
#include <hpx/lcos/local/mutex.hpp>

#include <set>

HPX_REGISTER_COMPONENT(hpx::components::component<tree>, tree);

pinned_vector<source> tree::ewald_sources;

hpx::lcos::local::spinlock tree::thread_mtx;
int tree::thread_cnt = 1;

HPX_PLAIN_ACTION(tree::set_ewald_sources, set_ewald_sources_action);

bool tree::inc_thread() {
	std::lock_guard < hpx::lcos::local::spinlock > lock(thread_mtx);
	if (thread_cnt < 48) {
		thread_cnt++;
		return true;
	} else {
		return false;
	}

}

void tree::dec_thread() {
	std::lock_guard < hpx::lcos::local::spinlock > lock(thread_mtx);
	thread_cnt--;
}

void tree::set_ewald_sources(std::vector<source> s) {
	ewald_sources.resize(s.size());
	for (int i = 0; i < s.size(); i++) {
		ewald_sources[i] = s[i];
	}
	if (hpx::get_locality_id() == 0) {
		const auto localities = hpx::find_all_localities();
		std::vector<hpx::future<void>> futs;
		for (int i = 1; i < localities.size(); i++) {
			futs.push_back(hpx::async < set_ewald_sources_action > (localities[i], s));
		}
		hpx::wait_all(futs);
	}
}

tree::tree(tree_id id_, const std::vector<particle> &_parts, const std::array<node_attr, NCHILD> &_children, const std::array<int, NCHILD> &_child_loads,
		const range &_box, bool _leaf) {
	id = id_;
	mtx = std::make_shared<hpx::lcos::local::mutex>();
	parts = _parts;
	children = _children;
	child_loads = _child_loads;
	box = _box;
	leaf = _leaf;
	dead = false;
}

tree::tree(tree_id id_, std::vector<particle> &&these_parts, const range &box_) :
		box(box_), dead(false) {
	const int sz = these_parts.size();
	static const auto opts = options::get();
	const auto npart_max = opts.parts_per_node;
	id = id_;
	mtx = std::make_shared<hpx::lcos::local::mutex>();

	/* Create initial box if root */
	if (box == null_range()) {
		for (int dim = 0; dim < NDIM; dim++) {
			box.min[dim] = -0.5;
			box.max[dim] = +0.5;
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

tree::~tree() {
}

std::pair<std::uint64_t,int> tree::get_local_pointer() {
	std::pair<std::uint64_t,int> rc;
	rc.first = reinterpret_cast<std::uint64_t>(this);
	rc.second = hpx::get_locality_id();
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
	std::vector<particle> pl, pr;
	int max_dim = 0;
	for (int dim = 0; dim < NDIM; dim++) {
		const auto s = box.max[dim] - box.min[dim];
		if (s > max_span) {
			max_span = s;
			max_dim = dim;
		}
	}
	const real mid = 0.5 * (box.min[max_dim] + box.max[max_dim]);
	boxl.max[max_dim] = boxr.min[max_dim] = mid;
	children[0].box = boxl;
	children[1].box = boxr;
	for (const auto &p : parts) {
		if (p.x[max_dim] < mid) {
			pl.push_back(p);
		} else {
			pr.push_back(p);
		}
	}
	parts = decltype(parts)();

	leaf = false;
//	printf( "%li %li\n", pr.size(), pl.size());
	auto fl = hpx::new_ < tree > (hpx::find_here(), tree_id_child_left(id), std::move(pl), boxl);
	auto fr = hpx::new_ < tree > (hpx::find_here(), tree_id_child_right(id), std::move(pr), boxr);
	children[0].id = fl.get();
	children[1].id = fr.get();
	children[0].check = checkitem(children[0].id);
	children[1].check = checkitem(children[1].id);
}

std::vector<particle> tree::destroy() {
	self = hpx::invalid_id;
	dead = true;
	return parts;
}

void tree::find_home(const std::vector<particle> &homeless) {
	std::vector<particle> self_parts;
	std::vector<particle> parent_parts;

	for (auto &pi : homeless) {
		if (in_range(pi.x, box)) {
			self_parts.push_back(pi);
		} else {
			parent_parts.push_back(pi);
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
				for (auto i = self_parts.begin(); i != self_parts.end();) {
					auto &p = *i;
					if (in_range(p.x, children[ci].box)) {
						cparts.push_back(p);
						std::swap(p, self_parts.back());
						self_parts.pop_back();
					} else {
						i++;
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
		const auto &p = parent_parts.front();
		printf("Lost a particle %f %f %f |%f %f %f %f %f %f |\n", p.x[0].get(), p.x[1].get(), p.x[2].get(), box.min[0].get(), box.max[0].get(),
				box.min[1].get(), box.max[1].get(), box.min[2].get(), box.max[2].get());
	}
	self_fut.get();
}

tree_attr tree::finish_drift() {
	static const auto opts = options::get();
	const auto npart_max = opts.parts_per_node;
	if (leaf) {
		if (parts.size() > npart_max || tree_id_level(id) < opts.min_level) {
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
		if (cparts <= npart_max && all_leaves && tree_id_level(id) >= opts.min_level) {
			std::array < hpx::future<std::vector<particle>>, NCHILD > dfuts;
			for (int ci = 0; ci < NCHILD; ci++) {
				dfuts[ci] = hpx::async < destroy_action > (children[ci].id);
			}
			for (int ci = 0; ci < NCHILD; ci++) {
				const auto tmp = dfuts[ci].get();
				children[ci].id = hpx::invalid_id;
				children[ci].check = checkitem();
				for (auto &p : tmp) {
					parts.push_back(p);
				}
			}
			leaf = true;
		}
	}
//	decltype(parts) tmp(parts.size());
//	tmp = parts;
//	parts = std::move(tmp);
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

std::array<checkitem, NCHILD> tree::open_check() const {
	std::array < checkitem, NCHILD > cids;
	for (int ci = 0; ci < NCHILD; ci++) {
		cids[ci] = children[ci].check;
	}
	return cids;
}

hpx::id_type tree::get_parent() const {
	return parent;
}

hpx::id_type tree::migrate(const hpx::id_type &loc) {
	auto id_fut = hpx::new_ < tree > (loc, id, parts, children, child_loads, box, leaf);
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
				children[ci].check = checkitem(children[ci].id);
			}
			futs[ci] = hpx::async < redistribute_workload_action > (children[ci].id, current, total);
			current += child_loads[ci];
		}
		hpx::wait_all(futs);
	}
}

void tree::send_particles(const std::vector<particle> &pj) {
	std::lock_guard < hpx::lcos::local::mutex > lock(*mtx);
	for (auto p : pj) {
		parts.push_back(std::move(p));
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
		for (auto p : parts) {
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
		for (auto part : parts) {
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

