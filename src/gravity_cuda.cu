#include <ntiger/gravity.hpp>
#include <ntiger/gravity_cuda.hpp>
#include <ntiger/options.hpp>

real *eforce;
real *epot;

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))

constexpr int DZ = 1;
constexpr int DY = EWALD_NBIN + 1;
constexpr int DX = (EWALD_NBIN + 1) * (EWALD_NBIN + 1);

void set_cuda_ewald_tables(const ewald_table_t &f, const ewald_table_t &phi) {
	CUDA_CHECK(cudaMalloc((void** ) &eforce, sizeof(real) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1)));
	CUDA_CHECK(cudaMalloc((void** ) &epot, sizeof(real) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1) * (EWALD_NBIN + 1)));
	CUDA_CHECK(cudaMemcpy(eforce, f.data(), sizeof(ewald_table_t), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(epot, phi.data(), sizeof(ewald_table_t), cudaMemcpyHostToDevice));
}

__global__
void gravity_near_kernel(gravity *g, const vect *x, const vect *y, int xsize, int ysize, real h, real m, bool ewald, const real *ef, const real *ep) {
	int base = blockIdx.x * blockDim.x;
	int i = threadIdx.x + base;
	gravity this_g;
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
					const real dxbin = 0.5 / EWALD_NBIN;                                   // 1 OP
					general_vect<int, NDIM> I;
					general_vect<real, NDIM> w;
					for (int dim = 0; dim < NDIM; dim++) {

						I[dim] = max(min(int((x0[dim] / dxbin).get()), EWALD_NBIN - 1), 0); // 3 * 1 OP
						w[dim] = 1.0 - (x0[dim] / dxbin - real(I[dim]));                    // 3 * 3 OP
					}
					const auto w000 = w[0] * w[1] * w[2];                                   // 2 OP
					const auto w001 = w[0] * w[1] * (1.0 - w[2]);                           // 3 OP
					const auto w010 = w[0] * (1.0 - w[1]) * w[2];                           // 3 OP
					const auto w011 = w[0] * (1.0 - w[1]) * (1.0 - w[2]);                   // 4 OP
					const auto w100 = (1.0 - w[0]) * w[1] * w[1] * w[2];                    // 4 OP
					const auto w101 = (1.0 - w[0]) * w[1] * (1.0 - w[2]);                   // 4 OP
					const auto w110 = (1.0 - w[0]) * (1.0 - w[1]) * w[2];                   // 4 OP
					const auto w111 = (1.0 - w[0]) * (1.0 - w[1]) * (1.0 - w[2]);           // 5 OP
					const auto index = I[0] * DX + I[1] * DY + I[2] * DZ;
					fmag = ef[index] * w000;                                                // 1 OP
					fmag += ef[index + DZ] * w001;											// 2 OP
					fmag += ef[index + DY] * w010;											// 2 OP
					fmag += ef[index + DY + DZ] * w011;										// 2 OP
					fmag += ef[index + DX] * w100;											// 2 OP
					fmag += ef[index + DX + DZ] * w101;										// 2 OP
					fmag += ef[index + DX + DY] * w110;										// 2 OP
					fmag += ef[index + DX + DY + DZ] * w111;								// 2 OP
					f = x0 * (fmag / r);													// 4 OP
					phi = ep[index] * w000;                                                 // 1 OP
					phi += ep[index + DZ] * w001;											// 2 OP
					phi += ep[index + DY] * w010;											// 2 OP
					phi += ep[index + DY + DX] * w011;										// 2 OP
					phi += ep[index + DZ] * w100;											// 2 OP
					phi += ep[index + DX + DZ] * w101;										// 2 OP
					phi += ep[index + DX + DY] * w110;										// 2 OP
					phi += ep[index + DX + DY + DZ] * w111;									// 2 OP
				} else {
					phi = 2.8372975;
				}
				const real r3 = r * r * r;													// 2 OP
				if (r > 0.0) {
					if (r > h) {
						phi = phi - 1.0 / r;													// 2 OP
						f = f - x0 / r3;														// 6 OP
					} else {
						const auto h2 = h * h;
						const auto h3inv = 1.0 / (h * h * h);
						phi = phi - (1.5 * h * h - 0.5 * r * r) / (h * h * h);
						f = f - x0 / (h * h * h);
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
						const auto h2 = h * h;
						const auto h3inv = 1.0 / (h * h * h);
						f = -dx * h3inv;
						phi = -(1.5 * h2 - 0.5 * r * r) * h3inv;
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
	const auto threads_per_block = 256;
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
gravity_near_kernel<<<dimGrid, dimBlock>>>(cg,cx,cy,x.size(),y.size(),h,m,ewald, eforce, epot);
										CUDA_CHECK(cudaMemcpy(g.data(), cg, x.size() * sizeof(gravity), cudaMemcpyDeviceToHost));
	}
	return g;
}

std::vector<gravity> gravity_far_cuda(const std::vector<vect> &x, const std::vector<source> &y) {

}
