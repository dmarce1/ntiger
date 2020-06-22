#include <ntiger/math.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>

//constexpr real G = 6.67259e-8;
constexpr real G = 1;

std::vector<source> tree::gather_ewald_sources() const {
	const auto minlevel = options::get().min_level;
	if (tree_id_level(id) == minlevel) {
		std::vector<source> s(1);
		s[0].m = mtot;
		s[0].x = Xcom;
		return s;
	} else {
		hpx::future < std::vector < source >> fl = hpx::async < gather_ewald_sources_action > (children[0].id);
		hpx::future < std::vector < source >> fr = hpx::async < gather_ewald_sources_action > (children[1].id);
		auto vl = fl.get();
		auto vr = fr.get();
		for (auto s : vr) {
			vl.push_back(s);
		}
		return vl;
	}
}

mass_attr tree::compute_mass_attributes() {
	const auto h = options::get().kernel_size;
	Xcom = vect(0);
	mtot = 0.0;
	rmaxs = 0.0;
	rmaxb = 0.0;
	const real m = 1.0 / options::get().problem_size;
	if (leaf) {
		rmaxb = 0.0;
		if (parts.size()) {
			for (const auto &p : parts) {
				Xcom = Xcom + p.x * m;
				mtot += m;
			}
			if (mtot != 0.0) {
				Xcom = Xcom / mtot;
			} else {
				Xcom = range_center(box);
			}
			for (const auto &p : parts) {
				rmaxb = max(rmaxb, h + abs(p.x - Xcom));
			}
		}
	} else {
		std::array<hpx::future<mass_attr>, NCHILD> futs;
		std::array<mass_attr, NCHILD> child_attr;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < compute_mass_attributes_action > (children[ci].id);
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			child_attr[ci] = futs[ci].get();
			const auto &c = child_attr[ci];
			Xcom = Xcom + c.com * c.mtot;
			mtot += c.mtot;
		}
		if (mtot > 0.0) {
			Xcom = Xcom / mtot;
			for (int ci = 0; ci < NCHILD; ci++) {
				const auto &c = child_attr[ci];
				rmaxb = max(rmaxb, c.rmaxb + abs(Xcom - c.com));
			}
		} else {
			Xcom = range_center(box);
			rmaxb = 0.0;
		}
	}
	vect Xv;
	for (int i = 0; i < NCHILD; i++) {
		int m = i;
		for (int dim = 0; dim < NDIM; dim++) {
			if (m & 1) {
				Xv[dim] = box.min[dim];
			} else {
				Xv[dim] = box.max[dim];
			}
			m >>= 1;
		}
		rmaxs = max(rmaxs, abs(Xv - Xcom));
	}
	mass_attr mass;
	mass.com = Xcom;
	mass.rmaxs = rmaxs;
	mass.rmaxb = rmaxb;
	mass.mtot = mtot;
	return mass;
}

pinned_vector<vect> tree::get_gravity_particles() const {
	PROFILE();
	pinned_vector<vect> gparts;
	for (auto &p : parts) {
		gparts.push_back(p.x);
	}
	return gparts;
}

monopole_attr tree::get_monopole_attributes() const {
	PROFILE();
	monopole_attr mono;
	mono.com = Xcom;           // 12
	mono.mtot = mtot;          //  4
	mono.radius = min(rmaxs, rmaxb);         // 4
	mono.leaf = leaf;
	return mono;
}

fixed_real tree::compute_gravity(std::vector<checkitem> checklist, pinned_vector<source> sources, fixed_real t, fixed_real dt) {
	const static auto opts = options::get();
	const auto theta = opts.theta;
	const auto h = options::get().kernel_size;
	const auto m = 1.0 / options::get().problem_size;
	fixed_real tmin = fixed_real::max();
	std::vector < monopole_attr > monos;
	std::vector < hpx::future<std::array<hpx::id_type, NCHILD>> > opened;
	for (const auto &n : checklist) {
		monos.push_back(n.get_monopole());
	}
	const auto rmaxA = min(rmaxb, rmaxs);
	const auto ZA = Xcom;


	if (leaf) {
		std::vector < hpx::future<pinned_vector<vect>> > part_futs;
		std::vector < hpx::id_type > direct;
		{
			PROFILE();
			opened.clear();
//		printf( "Getting masses\n");
			for (int i = 0; i < checklist.size(); i++) {
				const auto tmp = monos[i];
				const auto rmaxB = tmp.radius;
				const auto ZB = tmp.com;
				real sep;
				if (opts.ewald) {
					sep = ewald_separation(ZA - ZB);
				} else {
					sep = abs(ZA - ZB);
				}
				if (sep > (rmaxA + rmaxB) / theta) {
					source s;
					s.m = tmp.mtot;
					s.x = tmp.com;
					sources.push_back(s);
				} else if (tmp.leaf) {
					direct.push_back(checklist[i].get_gid());
				} else {
					opened.push_back(hpx::async < get_children_action > (checklist[i].get_gid()));
				}
			}
		}
		hpx::wait_all (opened);
		{
			PROFILE();
			checklist.clear();
			for (auto &f : opened) {
				const auto tmp = f.get();
				checklist.insert(checklist.end(), tmp.begin(), tmp.end());
			}
			part_futs.resize(direct.size());
			for (int i = 0; i < direct.size(); i++) {
				part_futs[i] = hpx::async < get_gravity_particles_action > (direct[i]);
			}
		}
		hpx::wait_all (part_futs);
		{
			for (auto &n : part_futs) {
				auto pj = n.get();
				for (const auto &x : pj) {
					source s;
					s.m = m;
					s.x = x;
					sources.push_back(s);
				}
			}
			if (checklist.size()) {
				compute_gravity_action()(self, checklist, std::move(sources), t, dt);
			} else {
				pinned_vector<vect> activeX;
				for (auto &p : parts) {
					if (p.t + p.dt == t + dt) {
						activeX.push_back(p.x);
					}
				}
				auto g = direct_gravity(activeX, sources);
				if (opts.ewald) {
					const auto ge = ewald_gravity(activeX, ewald_sources);
					for (int i = 0; i < g.size(); i++) {
						g[i].g = g[i].g + ge[i].g;
						g[i].phi = g[i].phi + ge[i].phi;
					}
				}
				int j = 0;
				for (auto &p : parts) {
					if (p.t + p.dt == t + dt) {
						p.g = g[j].g;
						p.phi = g[j].phi;
						j++;
						p.v = p.v + p.g * real_type(p.dt) * 0.5;
						p.t += p.dt;
						p.dt = fixed_real::max();
						const auto a = abs(p.g);
						if (a > 0.0) {
							const real this_dt = sqrt(h / a);
							if (this_dt < (double) fixed_real::max()) {
								p.dt = double(min(p.dt, fixed_real(this_dt.get())));
							}
						}
						p.dt *= opts.cfl;
						p.dt = p.dt.nearest_log2();
						p.dt = min(p.dt, (t + dt).next_bin() - (t + dt));
						tmin = min(tmin, p.dt);
						p.v = p.v + p.g * real_type(p.dt) * 0.5;
					}
				}
			}
		}
	} else {
		std::array<hpx::future<fixed_real>, NCHILD> cfuts;
		{
			PROFILE();
			std::vector < hpx::id_type > leaf_nids;
			for (int i = 0; i < checklist.size(); i++) {
				const auto tmp = monos[i];
				const auto rmaxB = tmp.radius;
				const auto ZB = tmp.com;
				real sep;
				if (opts.ewald) {
					sep = ewald_separation(ZA - ZB);
				} else {
					sep = abs(ZA - ZB);
				}
				//		printf( "%e\n", sep);
				if (sep > (rmaxA + rmaxB) / theta) {
					source s;
					s.m = tmp.mtot;
					s.x = tmp.com;
					sources.push_back(s);
				} else if (tmp.leaf) {
					leaf_nids.push_back(checklist[i].get_gid());
				} else {
					opened.push_back(hpx::async < get_children_action > (checklist[i].get_gid()));
				}
			}
			checklist.clear();
			checklist.insert(checklist.end(), leaf_nids.begin(), leaf_nids.end());
		}
		hpx::wait_all (opened);
		{
			PROFILE();
			for (auto &c : opened) {
				const auto tmp = c.get();
				checklist.insert(checklist.end(), tmp.begin(), tmp.end());
			}
		}
		auto tmps = sources;
		for (int ci = 0; ci < NCHILD; ci++) {
			if (inc_thread()) {
				cfuts[ci] = hpx::async([this, ci, t, dt](std::vector<checkitem> checklist, pinned_vector<source> sources) {
					auto tmp = compute_gravity_action()(children[ci].id, checklist, std::move(sources), t, dt);
					dec_thread();
					return tmp;
				},checklist, std::move(sources));
			} else {
				cfuts[ci] = hpx::async < compute_gravity_action > (children[ci].id, checklist, std::move(sources), t, dt);
			}
			sources = std::move(tmps);
		}
		hpx::wait_all(cfuts);
		{
			PROFILE();
			for (int ci = 0; ci < NCHILD; ci++) {
				tmin = min(tmin, cfuts[ci].get());
			}
		}
	}
	return tmin;
}
