//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef PROFILER_HPP_
#define PROFILER_HPP_


#include <algorithm>
#include <array>

struct profiler_register {
	profiler_register(const char*, int);
};
void profiler_enter(const char* func, int line);
void profiler_exit();
void profiler_output(FILE* fp);

struct profiler_scope {
	profiler_scope(const char* function, int line) {
		profiler_enter(function, line);
	}
	~profiler_scope() {
		profiler_exit();
	}
};
#define PROFILE_OFF

#ifdef PROFILE_OFF
#define PROFILE()
#else
#define PROFILE() static profiler_register prof_reg(__FUNCTION__, __LINE__); \
	             profiler_scope __profile_object__(__FUNCTION__, __LINE__)
#endif


#endif /* PROFILER_HPP_ */
