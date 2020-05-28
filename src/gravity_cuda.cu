#include <ntiger/gravity.hpp>
#include <ntiger/gravity_cuda.hpp>


__global__
void gravity_near_kernel(gravity *g, const vect *x, const vect *y, int size) {
	int i = threadIdx.x;

}

std::vector<gravity> gravity_near_cuda(const std::vector<vect> &x, const std::vector<vect>&) {
//	const auto size = g.size();
//	gravity *cg;
//	vect *cx;
//	vect *cy;
//	cudaMalloc((void**) &cg, sizeof(gravity) * size);
//	cudaMalloc((void**) &cx, sizeof(vect) * size);
//	cudaMalloc((void**) &cy, sizeof(vect) * size);
//	cudaMemcpy(cg,g.data(),sizeof(gravity)*size, cudaMemcpyHostToDevice);

}

std::vector<gravity> gravity_far_cuda(const std::vector<vect> &x, const std::vector<source> &y) {

}
