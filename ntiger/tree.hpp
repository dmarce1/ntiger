/*
 * tree.hpp
 *
 *  Created on: Dec 5, 2019
 *      Author: dmarce1
 */

#ifndef TREE_SERVER_CPP_
#define TREE_SERVER_CPP_

#include "particle.hpp"
#include "range.hpp"
#include "fixed_real.hpp"

#include <hpx/include/components.hpp>
#include <hpx/runtime/components/server/migrate_component.hpp>
#include <ntiger/tree_stats.hpp>

struct tree_attr {
	bool leaf;
	bool dead;
	int nparts;
	range box;
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & leaf;
		arc & dead;
		arc & nparts;
		arc & box;
	}
};

struct sibling_attr {
	hpx::id_type id;
	range box;
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & id;
		arc & box;
	}
};

struct mass_attr {
	vect com;
	real mtot;
	real rmaxs;
	real rmaxb;
	bool leaf;
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & com;
		arc & mtot;
		arc & rmaxs;
		arc & rmaxb;
		arc & leaf;
	}
};

class tree: public hpx::components::component_base<tree> {
	std::vector<particle> new_parts;
	std::vector<particle> parts;
	std::array<hpx::id_type, NCHILD> children;
	std::array<int, NCHILD> child_loads;
	std::vector<sibling_attr> siblings;
	hpx::id_type parent;
	hpx::id_type self;
	range root_box;
	range box;
	int nparts0;
	bool leaf;
	bool dead;
	mass_attr mass;
	std::shared_ptr<hpx::lcos::local::mutex> mtx;

public:

	enum bnd_ex_type { HYDRO, PRIMITIVE, TIMESTEP, NESTING, ALL};

	tree();
	tree( const std::vector<particle>& _parts, const std::array<hpx::id_type, NCHILD>& _children, const std::array<int, NCHILD>& _child_loads, const range& _root_box, const range& _box, bool _leaf );
	tree(std::vector<particle>&&, const range&, const range&);

	bool adjust_timesteps(fixed_real t, int);
	void apply_gravity(fixed_real, fixed_real);
	mass_attr compute_mass_attributes();
	void compute_drift(fixed_real);
	void compute_gravity(std::vector<hpx::id_type>, std::vector<mass_attr>, fixed_real, fixed_real);
	fixed_real compute_timestep(fixed_real);
	void compute_interactions(fixed_real, fixed_real);
	int compute_workload();
	void advance_time(fixed_real t);
	void create_children();
	std::vector<particle> destroy();
	void form_tree(std::vector<hpx::id_type>, bool = true);
	tree_attr finish_drift();
	tree_attr get_attributes() const;
	std::array<hpx::id_type, NCHILD> get_children() const;
	std::vector<gravity_part> get_gravity_particles() const;
	mass_attr get_mass_attributes() const;
	hpx::id_type get_parent() const;
	std::vector<vect> get_particle_positions(range) const;
	std::vector<particle> get_particles(range, range) const;
	void get_neighbor_particles(enum bnd_ex_type);
	std::vector<nesting_particle> get_nesting_particles(range, range) const;
	std::vector<timestep_particle> get_timestep_particles(range, range) const;
	void redistribute_workload(int, int);
	void send_particles(const std::vector<particle>&);
	void set_self_and_parent(const hpx::id_type, const hpx::id_type);
	tree_stats tree_statistics() const;
	void write_checkpoint(const std::string&, fixed_real) const;
	void write_silo(int, fixed_real) const;
	hpx::id_type migrate(const hpx::id_type&);

	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & parts;
		arc & children;
		arc & child_loads;
		arc & siblings;
		arc & parent;
		arc & self;
		arc & root_box;
		arc & box;
		arc & nparts0;
		arc & leaf;
		arc & dead;
		arc & mass;
	}

	HPX_DEFINE_COMPONENT_ACTION(tree,adjust_timesteps);
	HPX_DEFINE_COMPONENT_ACTION(tree,apply_gravity);
	HPX_DEFINE_COMPONENT_ACTION(tree,compute_mass_attributes);
	HPX_DEFINE_COMPONENT_ACTION(tree,compute_drift);
	HPX_DEFINE_COMPONENT_ACTION(tree,compute_timestep);
	HPX_DEFINE_COMPONENT_ACTION(tree,compute_interactions);
	HPX_DEFINE_COMPONENT_ACTION(tree,compute_workload);
	HPX_DEFINE_COMPONENT_ACTION(tree,advance_time);
	HPX_DEFINE_COMPONENT_ACTION(tree,destroy);
	HPX_DEFINE_COMPONENT_ACTION(tree,form_tree);
	HPX_DEFINE_COMPONENT_ACTION(tree,get_neighbor_particles);
	HPX_DEFINE_COMPONENT_ACTION(tree,finish_drift);
	HPX_DEFINE_COMPONENT_ACTION(tree,compute_gravity);
	HPX_DEFINE_COMPONENT_ACTION(tree,redistribute_workload);
	HPX_DEFINE_COMPONENT_ACTION(tree,tree_statistics);
	HPX_DEFINE_COMPONENT_ACTION(tree,write_checkpoint);
	HPX_DEFINE_COMPONENT_ACTION(tree,write_silo);
	HPX_DEFINE_COMPONENT_ACTION(tree,migrate);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_attributes);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_gravity_particles);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_mass_attributes);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_parent);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_particle_positions);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_particles);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_nesting_particles);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_timestep_particles);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_children);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,send_particles);
	HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,set_self_and_parent);

};

#endif /* TREE_SERVER_CPP_ */
