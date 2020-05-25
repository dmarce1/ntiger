#include <ntiger/math.hpp>
#include <ntiger/options.hpp>
#include <ntiger/profiler.hpp>
#include <ntiger/tree.hpp>

//constexpr real G = 6.67259e-8;
constexpr real G = 1;

void tree::apply_gravity(fixed_real t, fixed_real dt) {
	//PROFILE();
	const static auto opts = options::get();
	if (leaf) {
		for (int i = 0; i < parts.size(); i++) {
			auto &p = parts[i];
			if (dt != fixed_real(0.0)) {
				if (p.t + p.dt == t + dt || opts.global_time) {
					const auto this_dt = opts.global_time ? dt : p.dt;
					p.v = p.v + p.g * double(this_dt);
				}
			}
		}
	} else {
		std::array<hpx::future<void>, NCHILD> futs;
		for (int ci = 0; ci < NCHILD; ci++) {
			futs[ci] = hpx::async < apply_gravity_action > (children[ci].id, t, dt);
		}
		hpx::wait_all(futs);
	}
}

mass_attr tree::compute_mass_attributes() {
	auto &Xcom = mass.com;
	auto &mtot = mass.mtot;
	auto &rmaxs = mass.rmaxs;
	auto &rmaxb = mass.rmaxb;
	Xcom = vect(0);
	mtot = 0.0;
	rmaxs = 0.0;
	rmaxb = 0.0;
	mass.leaf = leaf;
	if (leaf) {
		//PROFILE();
		rmaxb = 0.0;
		if (parts.size()) {
			for (const auto &p : parts) {
				Xcom = Xcom + p.x * p.m;
				mtot += p.m;
			}
			if (mtot != 0.0) {
				Xcom = Xcom / mtot;
			} else {
				Xcom = range_center(box);
			}
			for (const auto &p : parts) {
				if (p.m != 0.0) {
					rmaxb = max(rmaxb, p.h + abs(p.x - Xcom));
				}
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

std::vector<gravity_part> tree::get_gravity_particles() const {
	//PROFILE();
	std::vector<gravity_part> gparts(parts.size());
	for (int i = 0; i < parts.size(); i++) {
		gparts[i].m = parts[i].m;
		gparts[i].x = parts[i].x;
		gparts[i].h = parts[i].h;
	}
	return gparts;
}

mass_attr tree::get_mass_attributes() const {
	return mass;
}

void tree::compute_gravity(std::vector<hpx::id_type> nids, std::vector<mass_attr> masses, fixed_real t, fixed_real dt, bool self_call) {
	const static auto opts = options::get();
	const auto theta = opts.theta;
	assert(nparts0 == parts.size());
	std::vector < hpx::future < mass_attr >> futs;
	std::vector < hpx::future<std::array<hpx::id_type, NCHILD>> > ncfuts;
	for (const auto &n : nids) {
		futs.push_back(hpx::async < get_mass_attributes_action > (n));
	}
	const auto rmaxA = min(mass.rmaxb, mass.rmaxs);
	const auto ZA = mass.com;
	if (leaf) {
		std::vector < hpx::id_type > near;
		ncfuts.clear();
		for (int i = 0; i < nids.size(); i++) {
			const auto tmp = futs[i].get();
			const auto rmaxB = min(tmp.rmaxb, tmp.rmaxs);
			const auto ZB = tmp.com;
			if (abs(ZA - ZB) > (rmaxA + rmaxB) / theta) {
				masses.push_back(tmp);
			} else if (tmp.leaf) {
				near.push_back(nids[i]);
			} else {
				ncfuts.push_back(hpx::async < get_children_action > (nids[i]));
			}
		}
		nids.clear();
		for (auto &f : ncfuts) {
			const auto tmp = f.get();
			nids.insert(nids.end(), tmp.begin(), tmp.end());
		}
		if (!self_call) {
			for (auto &pi : parts) {
				if (pi.t + pi.dt == t + dt || opts.global_time) {
					pi.g = vect(0);
				}
			}
		}
		hpx::future<void> self_fut;
		if (nids.size()) {
			self_fut = hpx::async < compute_gravity_action > (self, nids, std::vector<mass_attr>(), t, dt, true);
		} else {
			self_fut = hpx::make_ready_future<void>();
		}
		self_fut.get();
		std::vector < hpx::future<std::vector<gravity_part>> > gfuts(near.size());
		for (int i = 0; i < near.size(); i++) {
			gfuts[i] = hpx::async < get_gravity_particles_action > (near[i]);
		}
		{
			PROFILE();
			for (int j = 0; j < masses.size(); j++) {
				for (int i = 0; i < parts.size(); i++) {
					auto &pi = parts[i];
					if (pi.t + pi.dt == t + dt || opts.global_time) {
						const auto r = pi.x - masses[j].com;
						const auto r3inv = pow(abs(r), -3);
						pi.g = pi.g - r * (G * masses[j].mtot * r3inv);
					}
				}
			}
		}
		static thread_local std::vector<gravity_part> pj;
		pj.clear();
		for (auto &n : gfuts) {
			const auto list = n.get();
			for (int i = 0; i < list.size(); i++) {
				pj.push_back(list[i]);
			}
		}
		{
			PROFILE();
			for (int i = 0; i < parts.size(); i++) {
				auto &pi = parts[i];
				if (pi.t + pi.dt == t + dt || opts.global_time) {
					for (int j = 0; j < pj.size(); j++) {
						const auto r = pi.x - pj[j].x;
						const auto r0 = abs(r);
						if (r0 > 0.0) {
							const auto h = std::max(pi.h,pj[j].h);
							const auto f = grav_force(abs(r), h);
							const auto c = f * G * pj[j].m;
							pi.g = pi.g + r * c / r0;
						}
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
			if (abs(ZA - ZB) > (rmaxA + rmaxB) / theta) {
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
