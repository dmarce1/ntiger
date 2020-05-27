#pragma once

#include <ntiger/vect.hpp>



void ewald_force_and_pot(vect x, vect& f, real& phi, real);
real ewald_separation(vect x);
vect ewald_location(vect x);
real EW(vect);
