#include <ntiger/gravity.hpp>
#include <ntiger/gravity_cuda.hpp>
#include <ntiger/options.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))

constexpr int DZ = 1;
constexpr int DY = EWALD_NBIN + 1;
constexpr int DX = (EWALD_NBIN + 1) * (EWALD_NBIN + 1);

real_type *eforce;
real_type *epot;

texture<real_type, NDIM> ftex;
texture<real_type, NDIM> ptex;

void set_cuda_ewald_tables(const ewald_table_t &f, const ewald_table_t &phi) {
	CUDA_CHECK(cudaMalloc((void** ) &eforce, sizeof(real) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1)));
	CUDA_CHECK(cudaMalloc((void** ) &epot, sizeof(real) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1)));
	CUDA_CHECK(cudaMemcpy(eforce, f.data(), sizeof(ewald_table_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(epot, phi.data(), sizeof(ewald_table_t), cudaMemcpyHostToDevice));

	for (int i = 0; i < NDIM; i++) {
		ftex.addressMode[i] = cudaAddressModeClamp;
		ptex.addressMode[i] = cudaAddressModeClamp;

	}
	ftex.filterMode = cudaFilterModeLinear;
	ptex.filterMode = cudaFilterModeLinear;
	ftex.normalized = false;
	ptex.normalized = false;
	size_t offset = 0;
	constexpr int S3 = (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1);
	cudaBindTexture(&offset, ftex, eforce, sizeof(real_type) * S3);
	cudaBindTexture(&offset, ptex, epot, sizeof(real_type) * S3);
}

__global__
void gravity_near_kernel(gravity *g, const vect *x, const vect *y, int xsize, int ysize, real h, real m, bool ewald) {
	int base = blockIdx.x * blockDim.x;
	int i = threadIdx.x + base;
	gravity this_g;
	const real dxbin = 0.5 / EWALD_NBIN;                                   // 1 OP
	const real h2t15 = 1.5 * h * h;
	const real h3inv = 1.0 / (h * h * h);
	if (i < xsize) {
		this_g.g = vect(0);
		this_g.phi = 0.0;
		for (int j = 0; j < ysize; j++) {
			vect f;
			real phi;
			const auto dx = x[i] - y[j]; // 3 OP
			if (ewald) {
				auto x0 = dx;
				vect sgn(1.0);
				for (int dim = 0; dim < NDIM; dim++) {
					if (x0[dim] < 0.0) {
						x0[dim] = -x0[dim];                         // 3 * 1 OP
						sgn[dim] *= -1.0;                           // 3 * 1 OP
					}
					if (x0[dim] > 0.5) {
						x0[dim] = 1.0 - x0[dim];                   // 3 * 1 OP
						sgn[dim] *= -1.0;                          // 3 * 1 OP
					}
				}
				const real r = abs(x0);
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim] = 0.0;
				}
				phi = 0.0;
				// Skip ewald
				real fmag = 0.0;
				if (r > 1.0e-3) {
					general_vect<real_type, NDIM> I;
					for (int dim = 0; dim < NDIM; dim++) {
						I[dim] = (x0[dim] / dxbin).get() + real_type(0.5); 					// 3 * 1 OP
					}
					fmag = tex3D(ftex, I[0], I[1], I[2]);									// 33 op
					f = x0 * (fmag / r);													// 4 OP
					phi = tex3D(ptex, I[0], I[1], I[2]);                                    // 33 OP
				} else {
					phi = 2.8372975;
				}
				const real r3 = r * r * r;													// 2 OP
				if (r > 0.0) {
					if (r > h) {
						phi = phi - 1.0 / r;													// 2 OP
						f = f - x0 / r3;														// 6 OP
					} else {
						phi = phi - (h2t15 - 0.5 * r * r) * h3inv;
						f = f - x0 * h3inv;
					}
				}
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim] *= sgn[dim];														// 3 OP
				}
			} else {
				const auto r = abs(dx); // 5 OP
				if (r > 0.0) {
					if (r > h) {            // 1 OP
						const auto rinv = 1.0 / r;            //1 OP
						const auto r3inv = rinv * rinv * rinv;            //2 OP
						f = -dx * r3inv;            //6 OP
						phi = -rinv;                //1 OP
					} else {
						f = -dx * h3inv;
						phi = -(h2t15 - 0.5 * r * r) * h3inv;
					}
				}
			}
			this_g.g = this_g.g + f; // 3 OP
			this_g.phi += phi; // 1 OP
		}
		g[i].g = this_g.g * m; // 1 OP
		g[i].phi = this_g.phi * m; // 1 OP
	}
}

std::vector<gravity> gravity_near_cuda(const std::vector<vect> &x, const std::vector<vect> &y) {
	std::vector<gravity> g(x.size());
	const auto threads_per_block = 512;
	if (x.size() > 0) {
		static const bool ewald = options::get().ewald;
		static const real h = options::get().kernel_size;
		static const real m = 1.0 / options::get().problem_size;
		static thread_local gravity *cg = nullptr;
		static thread_local vect *cx = nullptr;
		static thread_local vect *cy = nullptr;
		static thread_local int xmax = 0;
		static thread_local int ymax = 0;
		if (x.size() > xmax) {
			if (xmax > 0) {
				CUDA_CHECK(cudaFree(cg));
				CUDA_CHECK(cudaFree(cx));
			}
			CUDA_CHECK(cudaMalloc((void** ) &cg, sizeof(gravity) * x.size()));
			CUDA_CHECK(cudaMalloc((void** ) &cx, sizeof(vect) * x.size()));
			xmax = x.size();
		}
		if (y.size() > ymax) {
			if (ymax > 0) {
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
