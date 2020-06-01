#pragma once

#include <vector>
#include <unordered_set>
#include <stack>

#include <hpx/lcos/local/mutex.hpp>

template<class T>
class arena {
	using mutex_type = hpx::lcos::local::mutex;
	mutex_type mtx;
	std::unordered_set<std::vector<T>*> used_vecs;
	std::stack<std::vector<T>*> free_vecs;
public:
	std::vector<T>& allocate() {
		std::lock_guard<mutex_type> lock(mtx);
		std::vector<T>* ptr;
		if( free_vecs.size() ) {
			ptr = free_vecs.top();
			free_vecs.pop();
		} else {
			ptr = new std::vector<T>;
		}
		used_vecs.insert(ptr);
		ptr->resize(0);
		return *ptr;
	}
	void deallocate(std::vector<T>& vec) {
		std::lock_guard<mutex_type> lock(mtx);
		auto* ptr = &vec;
		used_vecs.erase(ptr);
		free_vecs.push(ptr);

	}
	~arena() {
		for( auto* ptr : used_vecs ){
			delete ptr;
		}
		while( free_vecs.size()) {
			delete free_vecs.top();
			free_vecs.pop();
		}
	}

};
