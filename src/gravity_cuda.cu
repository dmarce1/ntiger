#include <ntiger/gravity.hpp>
#include <ntiger/gravity_cuda.hpp>
#include <ntiger/options.hpp>

__global__
void gravity_near_kernel(gravity *g, const vect *x, const vect *y, int xsize, int ysize, real h, real m, bool ewald) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < xsize) {
		g[i].g = vect(0);
		g[i].phi = 0.0;
		const auto h2 = h * h;
		const auto h3inv = 1.0 / (h * h * h);
		for (int j = 0; j < ysize; j++) {
			vect f;
			real phi;
			const auto dx = x[i] - y[j];
//		if (ewald) {
			//			ewald_force_and_pot(dx, f, phi, h);
//		} else {
			const auto r = abs(dx);
			if (r > h) {
				const auto rinv = 1.0 / r;
				const auto r3inv = rinv * rinv * rinv;
				f = -dx * r3inv;
				phi = -rinv;
			} else {
				f = -dx * h3inv;
				phi = -(1.5 * h2 - 0.5 * r * r) * h3inv;
			}
//		}
			g[i].g = g[i].g + f;
			g[i].phi += phi;
		}
		g[i].g = g[i].g * m;
		g[i].phi *= m;
	}
}

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s\n", __LINE__, __FILE__)

std::vector<gravity> gravity_near_cuda(const std::vector<vect> &x, const std::vector<vect> &y) {
	std::vector<gravity> g(x.size());
	const auto threads_per_block = std::max(std::min(int(sqrt(x.size())), 512), 32);
	if (x.size() > 0) {
		static const bool ewald = options::get().ewald;
		static const real h = options::get().kernel_size;
		static const real m = 1.0 / options::get().problem_size;
		static thread_local gravity *cg = nullptr;
		static thread_local vect *cx = nullptr;
		static thread_local vect *cy = nullptr;
		static thread_local int xmax = 0;
		static thread_local int ymax = 0;
		if( x.size() > xmax ) {
			if( xmax > 0 ) {
				CUDA_CHECK(cudaFree(cg));
				CUDA_CHECK(cudaFree(cx));
			}
			CUDA_CHECK(cudaMalloc((void** ) &cg, sizeof(gravity) * x.size()));
			CUDA_CHECK(cudaMalloc((void** ) &cx, sizeof(vect) * x.size()));
			xmax = x.size();
		}
		if( y.size() > ymax) {
			if( ymax > 0 ) {
				CUDA_CHECK(cudaFree(cy));
			}
			CUDA_CHECK(cudaMalloc((void** ) &cy, sizeof(vect) * y.size()));
			ymax = y.size();
		}
		CUDA_CHECK(cudaMemcpy(cx, x.data(), x.size() * sizeof(vect), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(cy, y.data(), y.size() * sizeof(vect), cudaMemcpyHostToDevice));
		dim3 dimBlock(threads_per_block, 1);
		dim3 dimGrid((x.size() + threads_per_block - 1) / threads_per_block, 1);
gravity_near_kernel<<<dimGrid, dimBlock>>>(cg,cx,cy,x.size(),y.size(),h,m,ewald);
								CUDA_CHECK(cudaMemcpy(g.data(), cg, x.size() * sizeof(gravity), cudaMemcpyDeviceToHost));
	}
	return g;
}

std::vector<gravity> gravity_far_cuda(const std::vector<vect> &x, const std::vector<source> &y) {

}
