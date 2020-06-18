#include <ntiger/gravity.hpp>
#include <ntiger/math.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>

//constexpr real G = 6.67259e-8;
constexpr real G = 1;

fixed_real tree::apply_gravity(fixed_real t, fixed_real dt, bool first_kick) {
	const static auto opts = options::get();
	const auto h = opts.kernel_size;
	fixed_real tmin = fixed_real::max();
	//PROFILE();
	const auto first_condition = [](fixed_real t0, fixed_real dt0, fixed_real t1, fixed_real dt1) {
		return t0 == t1;
	};
	const auto second_condition = [](fixed_real t0, fixed_real dt0, fixed_real t1, fixed_real dt1) {
		return t0 + dt0 == t1 + dt1;
	};
	const auto always = [](fixed_real t0, fixed_real dt0, fixed_real t1, fixed_real dt1) {
		return true;
	};
	std::function<bool(fixed_real, fixed_real, fixed_real, fixed_real)> cond;
	if (opts.global_time) {
		cond = always;
	} else if (first_kick) {
		cond = first_condition;
	} else {
		cond = second_condition;
	}
	if (leaf) {
		for (auto &p : parts) {
			if (cond(p.t, p.dt, t, dt)) {
				p.v = p.v + p.g * real_type(opts.global_time ? dt : p.dt) * 0.5;
				if (!first_kick) {
					p.t += opts.global_time ? dt : p.dt;
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
				}
			}
		}
	} else {
		std::array<hpx::future<fixed_real>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < apply_gravity_action > (children[ci].id, t, dt, first_kick);
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			tmin = min(tmin, futs[ci].get());
		}
	}
	return tmin;
}

mass_attr tree::compute_mass_attributes() {
	const auto h = options::get().kernel_size;
	auto &Xcom = mass.com;
	auto &mtot = mass.mtot;
	auto &rmaxs = mass.rmaxs;
	auto &rmaxb = mass.rmaxb;
	Xcom = vect(0);
	mtot = 0.0;
	rmaxs = 0.0;
	rmaxb = 0.0;
	mass.leaf = leaf;
	const real m = 1.0 / options::get().problem_size;
	if (leaf) {
		//PROFILE();
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
	//PROFILE();
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
	return mass;
}

std::vector<vect> tree::get_gravity_particles() const {
	//PROFILE();
	std::vector<vect> gparts;
	gparts.reserve(parts.size());
	for (auto &p : parts) {
		gparts.push_back(p.x);
	}
	return gparts;
}

mass_attr tree::get_mass_attributes() const {
	return mass;
}

void tree::compute_gravity(std::vector<hpx::id_type> nids, std::vector<mass_attr> masses, fixed_real t, fixed_real dt, bool self_call) {
	const static auto opts = options::get();
	const auto theta = opts.theta;
	const auto h = options::get().kernel_size;
	std::vector < hpx::future < mass_attr >> futs;
	std::vector < hpx::future<std::array<hpx::id_type, NCHILD>> > ncfuts;
	for (const auto &n : nids) {
		futs.push_back(hpx::async < get_mass_attributes_action > (n));
	}
	const auto rmaxA = min(mass.rmaxb, mass.rmaxs);
	const auto ZA = mass.com;
	const real m = 1.0 / options::get().problem_size;
	if (leaf) {
		std::vector < hpx::id_type > near;
		ncfuts.clear();
//		printf( "Getting masses\n");
		for (int i = 0; i < nids.size(); i++) {
			const auto tmp = futs[i].get();
			const auto rmaxB = min(tmp.rmaxb, tmp.rmaxs);
			const auto ZB = tmp.com;
			real sep;
			if (opts.ewald) {
				sep = ewald_separation(ZA - ZB);
			} else {
				sep = abs(ZA - ZB);
			}
			if (sep > (rmaxA + rmaxB) / theta) {
				masses.push_back(tmp);
			} else if (tmp.leaf) {
				near.push_back(nids[i]);
			} else {
				ncfuts.push_back(hpx::async < get_children_action > (nids[i]));
			}
		}
//		if( masses.size()) {
//			printf( "%i\n", masses.size());
//		}
		nids.clear();
		for (auto &f : ncfuts) {
			const auto tmp = f.get();
			nids.insert(nids.end(), tmp.begin(), tmp.end());
		}
		if (!self_call) {
			for (auto &pi : parts) {
				if (pi.t + pi.dt == t + dt || opts.global_time) {
					pi.g = vect(0);
					pi.phi = 0.0;
				}
			}
		}
		hpx::future<void> self_fut;
		if (nids.size()) {
//			printf( "Self call\n");
			self_fut = hpx::async < compute_gravity_action > (self, nids, std::vector<mass_attr>(), t, dt, true);
		} else {
			self_fut = hpx::make_ready_future<void>();
		}
		self_fut.get();
//		printf("Getting particles\n");
		std::vector < hpx::future<std::vector<vect>> > gfuts(near.size());
		for (int i = 0; i < near.size(); i++) {
			gfuts[i] = hpx::async < get_gravity_particles_action > (near[i]);
		}

		std::vector<vect> activeX;
		std::vector<source> sources;
		activeX.reserve(parts.size());
		for (auto &p : parts) {
			if (opts.global_time || (p.t + p.dt == t + dt)) {
				activeX.push_back(p.x);
			}
		}
		if (masses.size()) {
			sources.reserve(masses.size());
			for (int i = 0; i < masses.size(); i++) {
				source s;
				s.x = masses[i].com;
				s.m = masses[i].mtot;
				sources.push_back(s);
			}
			decltype(masses)().swap(masses);
//		if( masses.size() > 0 ) {
//			printf( "%i\n", (int) masses.size());
//		}
			const auto g = gravity_far(activeX, sources);
			std::lock_guard < hpx::lcos::local::mutex > lock(*mtx);
			int j = 0;
			for (auto &p : parts) {
				if (p.t + p.dt == t + dt) {
					p.g = p.g + g[j].g;
					p.phi += g[j].phi;
					j++;
				}
			}
			decltype(sources)().swap(sources);
		}
		hpx::wait_all (gfuts);
		std::vector<vect> plist, flist;
		for (auto &n : gfuts) {
			auto pj = n.get();
			plist.insert(plist.end(), pj.begin(), pj.end());
		}
		const auto R0 = max(mass.rmaxs, mass.rmaxb);
		int sz = plist.size();
		for (int i = 0; i < sz; i++) {
			if (abs(plist[i] - mass.com) + R0 < EWALD_R0) {
				flist.push_back(plist[i]);
				plist[i] = plist[sz - 1];
				sz--;
				i--;
				plist.resize(sz);
			}
		}
		if (flist.size()) {
			const auto g = gravity_near(activeX, std::move(flist), false);
			{
				std::lock_guard < hpx::lcos::local::mutex > lock(*mtx);
				int j = 0;
				for (auto &p : parts) {
					if (opts.global_time || (p.t + p.dt == t + dt)) {
						p.g = p.g + g[j].g;
						p.phi += g[j].phi;
						j++;
					}
				}
			}
		}
		if (plist.size()) {
			const auto g = gravity_near(activeX, std::move(plist), opts.ewald);
			{
				std::lock_guard < hpx::lcos::local::mutex > lock(*mtx);
				int j = 0;
				for (auto &p : parts) {
					if (opts.global_time || (p.t + p.dt == t + dt)) {
						p.g = p.g + g[j].g;
						p.phi += g[j].phi;
						j++;
					}
				}
			}
		}
	} else {
		std::array<hpx::future<void>, NCHILD> cfuts;
		std::vector < hpx::id_type > leaf_nids;
		for (int i = 0; i < nids.size(); i++) {
			const auto tmp = futs[i].get();
			const auto rmaxB = min(tmp.rmaxb, tmp.rmaxs);
			const auto ZB = tmp.com;
			real sep;
			if (opts.ewald) {
				sep = ewald_separation(ZA - ZB);
			} else {
				sep = abs(ZA - ZB);
			}
			//		printf( "%e\n", sep);
			if (sep > (rmaxA + rmaxB) / theta) {
				masses.push_back(tmp);
			} else if (tmp.leaf) {
				leaf_nids.push_back(nids[i]);
			} else {
				ncfuts.push_back(hpx::async < get_children_action > (nids[i]));
			}
		}
		nids.clear();
		nids.insert(nids.end(), leaf_nids.begin(), leaf_nids.end());
		for (auto &c : ncfuts) {
			const auto tmp = c.get();
			nids.insert(nids.end(), tmp.begin(), tmp.end());
		}
		for (int ci = 0; ci < NCHILD; ci++) {
			cfuts[ci] = hpx::async < compute_gravity_action > (children[ci].id, nids, masses, t, dt, false);
		}
		hpx::wait_all(cfuts);
	}
}
