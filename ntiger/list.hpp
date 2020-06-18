#pragma once

#include <forward_list>
#include <mutex>
#include <stack>

#include <hpx/lcos/local/spinlock.hpp>

#include <hpx/serialization/serialization_fwd.hpp>

template<class T>
class list_allocator {

	static std::stack<T*> free_list;
	static hpx::lcos::local::spinlock mutex;

public:
	using value_type = T;
	T* allocate(std::size_t) {
		std::lock_guard < hpx::lcos::local::spinlock > lock(mutex);
		if (free_list.size() == 0) {
			constexpr static std::size_t chunk_size = 1024 * ((1024 * 1024) / sizeof(T));
			printf("Allocating %li particles\n", chunk_size);
			T *base = (T*) malloc(sizeof(T) * chunk_size);
			for (int i = 0; i < chunk_size; i++) {
				free_list.push(base + i);
			}
		}
		T *ptr = free_list.top();
		free_list.pop();
		return ptr;
	}
	void deallocate(T *ptr, std::size_t) {
		std::lock_guard < hpx::lcos::local::spinlock > lock(mutex);
		free_list.push(ptr);
	}
	bool operator==(const list_allocator<T> &other) const {
		return true;
	}
	bool operator!=(const list_allocator<T> &other) const {
		return false;
	}

	list_allocator() {
	}

	list_allocator(const list_allocator<T> &other) {
	}

	template<class U>
	list_allocator(const list_allocator<U> &other) {
	}
};

template<class T>
class list {
	using list_type = std::forward_list<T, list_allocator<T>>;
	list_type list_;
	std::size_t size_;
public:
	list() {
		size_ = 0;
	}
	void push_front(const T &value) {
		size_++;
		list_.push_front(value);
	}
	T& front() {
		return list_.front();
	}
	void pop_front() {
		list_.pop_front();
		size_--;
	}
	std::size_t size() const {
		return size_;
	}
	typename list_type::iterator begin() {
		return list_.begin();
	}
	typename list_type::iterator end() {
		return list_.end();
	}
	typename list_type::const_iterator begin() const {
		return list_.cbegin();
	}
	typename list_type::const_iterator end() const {
		return list_.cend();
	}

	friend class hpx::serialization::access;

	template<class Archive>
	void save(Archive &ar, const unsigned int version) const {
		ar & size_;
		for (auto l : list_) {
			ar & l;
		}
	}

	template<class Archive>
	void load(Archive &ar, const unsigned int version) {
		ar & size_;
		for (int i = 0; i < size_; i++) {
			T tmp;
			ar & tmp;
			list_.push_front(tmp);
		}
	}

	HPX_SERIALIZATION_SPLIT_MEMBER()
};

template<class T>
std::stack<T*> list_allocator<T>::free_list;

template<class T>
hpx::lcos::local::spinlock list_allocator<T>::mutex;
