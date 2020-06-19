#include <ntiger/math.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>

//constexpr real G = 6.67259e-8;
constexpr real G = 1;

mass_attr tree::compute_mass_attributes() {
	const auto h = options::get().kernel_size;
	Xcom = vect(0);
	mtot = 0.0;
	rmaxs = 0.0;
	rmaxb = 0.0;
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
	mass_attr mass;
	mass.com = Xcom;
	mass.rmaxs = rmaxs;
	mass.rmaxb = rmaxb;
	mass.mtot = mtot;
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

monopole_attr tree::get_monopole_attributes() const {
	monopole_attr mono;
	mono.com = Xcom;           // 12
	mono.mtot = mtot;          //  4
	mono.radius = min(rmaxs, rmaxb);         // 4
	mono.leaf = leaf;
	return mono;
}

fixed_real tree::compute_gravity(std::vector<hpx::id_type> nids, std::vector<source> masses, fixed_real t, fixed_real dt, bool self_call) {
	const static auto opts = options::get();
	const auto theta = opts.theta;
	const auto h = options::get().kernel_size;
	const auto m = 1.0 / options::get().problem_size;
	fixed_real tmin = fixed_real::max();
	std::vector < hpx::future < monopole_attr >> futs;
	std::vector < hpx::future<std::array<hpx::id_type, NCHILD>> > ncfuts;
	for (const auto &n : nids) {
		futs.push_back(hpx::async < get_monopole_attributes_action > (n));
	}
	const auto rmaxA = min(rmaxb, rmaxs);
	const auto ZA = Xcom;
	if (leaf) {
		std::vector < hpx::id_type > near;
		ncfuts.clear();
//		printf( "Getting masses\n");
		for (int i = 0; i < nids.size(); i++) {
			const auto tmp = futs[i].get();
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
				masses.push_back(s);
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
				if (pi.t + pi.dt == t + dt) {
					pi.g = vect(0);
					pi.phi = 0.0;
				}
			}
		}
		hpx::future<void> self_fut;
		if (nids.size()) {
			self_fut = hpx::async < compute_gravity_action > (self, nids, std::vector<source>(), t, dt, true);
		} else {
			self_fut = hpx::make_ready_future<void>();
		}
		self_fut.get();
		std::vector < hpx::future<std::vector<vect>> > gfuts(near.size());
		for (int i = 0; i < near.size(); i++) {
			gfuts[i] = hpx::async < get_gravity_particles_action > (near[i]);
		}
		hpx::wait_all (gfuts);
		for (auto &n : gfuts) {
			auto pj = n.get();
			for (const auto &x : pj) {
				source s;
				s.m = m;
				s.x = x;
				masses.push_back(s);
			}
		}
		std::vector<vect> activeX;
		activeX.reserve(parts.size());
		for (auto &p : parts) {
			if (p.t + p.dt == t + dt) {
				activeX.push_back(p.x);
			}
		}
		{
			const auto g = direct_gravity(activeX, masses);
			std::lock_guard < hpx::lcos::local::mutex > lock(*mtx);
			int j = 0;
			for (auto &p : parts) {
				if (p.t + p.dt == t + dt) {
					p.g = p.g + g[j].g;
					p.phi += g[j].phi;
					j++;
				}
			}
		}

		if (!self_call) {
			for (auto &p : parts) {
				if (p.t + p.dt == t + dt) {
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

	} else {
		std::vector < hpx::id_type > leaf_nids;
		for (int i = 0; i < nids.size(); i++) {
			const auto tmp = futs[i].get();
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
				masses.push_back(s);
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
		std::array<hpx::future<fixed_real>, NCHILD> cfuts;
		for (int ci = 0; ci < NCHILD; ci++) {
			cfuts[ci] = hpx::async < compute_gravity_action > (children[ci].id, nids, masses, t, dt, false);
		}
		hpx::wait_all(cfuts);
		for (int ci = 0; ci < NCHILD; ci++) {
			tmin = min(tmin, cfuts[ci].get());
		}
	}
	return tmin;
}
