/*
 * gravity_cuda.hpp
 *
 *  Created on: May 28, 2020
 *      Author: dmarce1
 */

#ifndef NTIGER_GRAVITY_CUDA_HPP_
#define NTIGER_GRAVITY_CUDA_HPP_

#include <ntiger/vect.hpp>

std::vector<gravity> gravity_cuda(const std::vector<vect> &x, const std::vector<source> &y);
void set_cuda_ewald_tables(const std::array<ewald_table_t,NDIM> &f, const ewald_table_t &phi);

#endif /* NTIGER_GRAVITY_CUDA_HPP_ */
