#pragma once

#include <ntiger/vect.hpp>



real ewald_potential(real x);
vect ewald_force(vect x);
real ewald_separation(vect x);
vect ewald_location(vect x);
