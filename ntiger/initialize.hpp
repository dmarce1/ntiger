#pragma once

#include <functional>

#include <ntiger/particle.hpp>

using init_func_type = std::function<void(particle&)>;

init_func_type get_initialization_function(const std::string&);
