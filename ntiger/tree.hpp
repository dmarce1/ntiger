/*
 * tree.hpp
 *
 *  Created on: Dec 5, 2019
 *      Author: dmarce1
 */

#ifndef TREE_SERVER_CPP_
#define TREE_SERVER_CPP_

#include <ntiger/particle.hpp>
#include <ntiger/range.hpp>
#include <ntiger/fixed_real.hpp>

#include <hpx/include/components.hpp>
#include <hpx/runtime/components/server/migrate_component.hpp>

#include <ntiger/gravity.hpp>
#include <ntiger/tree_id.hpp>
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

struct node_attr {
	hpx::id_type id;
	range box;
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & id;
		arc & box;
	}
};

struct mass_attr {     // 28
	vect com;           // 12
	real mtot;          //  4
	real rmaxs;         // 4
	real rmaxb;         // 4
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & com;
		arc & mtot;
		arc & rmaxs;
		arc & rmaxb;
	}
};


struct monopole_attr {     // 28
	vect com;           // 12
	real mtot;          //  4
	real radius;         // 4
	bool leaf;
	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & com;
		arc & mtot;
		arc & radius;
		arc & leaf;
	}
};

class tree: public hpx::components::component_base<tree>  { // 196
	tree_id id;
	std::vector<particle> parts;								// 16
	std::array<node_attr, NCHILD> children;				// 64
	std::array<int, NCHILD> child_loads;				// 8
	hpx::id_type parent;                                // 8
	hpx::id_type self;                                  // 8
	range box;                                          // 24
	bool leaf;                                          // 4
	bool dead;                                          // 4
	vect Xcom;
	real mtot;
	real rmaxs;
	real rmaxb;
	std::shared_ptr<hpx::lcos::local::mutex> mtx;       // 32

	static pinned_vector<source> ewald_sources;
	static hpx::lcos::local::spinlock thread_mtx;
	static int thread_cnt;
	static bool inc_thread();
	static void dec_thread();
public:
	static void set_ewald_sources(std::vector<source>);

	tree();
	tree(tree_id id, const std::vector<particle> &_parts, const std::array<node_attr, NCHILD> &_children, const std::array<int, NCHILD> &_child_loads, const range &_box,
			bool _leaf);
	tree(tree_id id, std::vector<particle>&&, const range&);

	void apply_boost(vect);
	mass_attr compute_mass_attributes();
	void compute_drift(fixed_real);
	std::vector<source> gather_ewald_sources() const;
	fixed_real compute_gravity(std::vector<hpx::id_type>, pinned_vector<source>, fixed_real, fixed_real);
	void compute_interactions();
	int compute_workload();
	void create_children();
	std::vector<particle> destroy();
	void find_home(const std::vector<particle>&);
	tree_attr finish_drift();
	tree_attr get_attributes() const;
	std::array<hpx::id_type, NCHILD> get_children() const;
	pinned_vector<vect> get_gravity_particles() const;
	monopole_attr get_monopole_attributes() const;
	hpx::id_type get_parent() const;
	void rescale(real factor, range mybox);
	void redistribute_workload(int, int);
	void send_particles(const std::vector<particle>&);
	void set_self_and_parent(const hpx::id_type, const hpx::id_type);
	tree_stats tree_statistics() const;
	void keplerize();
	void virialize();
	void write_checkpoint(const std::string&, fixed_real) const;
	void write_silo(int, fixed_real) const;
	hpx::id_type migrate(const hpx::id_type&);

	template<class Arc>
	void serialize(Arc &&arc, unsigned) {
		arc & id;
		arc & parts;
		arc & children;
		arc & child_loads;
		arc & parent;
		arc & self;
		arc & box;
		arc & leaf;
		arc & dead;
		arc & Xcom;
		arc & mtot;
		arc & rmaxs;
		arc & rmaxb;
	}

	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,apply_boost);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,compute_mass_attributes);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,compute_drift);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,compute_interactions);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,compute_workload);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,destroy);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,find_home);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,finish_drift);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,compute_gravity);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,gather_ewald_sources);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,rescale);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,redistribute_workload);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,set_self_and_parent);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,tree_statistics);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,write_checkpoint);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,write_silo);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,migrate);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,keplerize);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,virialize);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_attributes);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_gravity_particles);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_monopole_attributes);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_parent);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,get_children);
	/***/HPX_DEFINE_COMPONENT_DIRECT_ACTION(tree,send_particles);

};

#endif /* TREE_SERVER_CPP_ */
