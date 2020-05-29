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
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	const real dxbin = 0.5 / EWALD_NBIN;
	if (i < xsize) {
		g[i].g = vect(0);
		g[i].phi = 0.0;
		const auto h2 = h * h;
		const auto h3inv = 1.0 / (h * h * h);
		for (int j = 0; j < ysize; j++) {
			vect f;
			real phi;
			const auto dx = x[i] - y[j];
			if (ewald) {
				const auto x0 = x[i];
				vect sgn(1.0);
				for (int dim = 0; dim < NDIM; dim++) {
					if (x0[dim] < 0.0) {
						x0[dim] = -x0[dim];
						sgn[dim] *= -1.0;
					}
					if (x0[dim] > 0.5) {
						x0[dim] = 1.0 - x0[dim];
						sgn[dim] *= -1.0;
					}
				}
				const real r = abs(x0);
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim] = 0.0;
				}
				phi = 0.0;
				// Skip ewald
				real fmag = 0.0;
				if (r > 1.0e-4) {
					general_vect<int, NDIM> I;
					general_vect<real, NDIM> w;
					for (int dim = 0; dim < NDIM; dim++) {
						I[dim] = min(int((x0[dim] / dxbin).get()), EWALD_NBIN - 1);
						w[dim] = 1.0 - (x0[dim] / dxbin - real(I[dim]));
					}
					const auto w000 = w[0] * w[1] * w[2];
					const auto w001 = w[0] * w[1] * (1.0 - w[2]);
					const auto w010 = w[0] * (1.0 - w[1]) * w[2];
					const auto w011 = w[0] * (1.0 - w[1]) * (1.0 - w[2]);
					const auto w100 = (1.0 - w[0]) * w[1] * w[1] * w[2];
					const auto w101 = (1.0 - w[0]) * w[1] * (1.0 - w[2]);
					const auto w110 = (1.0 - w[0]) * (1.0 - w[1]) * w[2];
					const auto w111 = (1.0 - w[0]) * (1.0 - w[1]) * (1.0 - w[2]);
					const auto index = I[0] * DX + I[1] * DY + I[2] * DZ;
					fmag += ef[index] * w000;
					fmag += ef[index + DX] * w001;
					fmag += ef[index + DY] * w010;
					fmag += ef[index + DY + DX] * w011;
					fmag += ef[index + DZ] * w100;
					fmag += ef[index + DZ + DX] * w101;
					fmag += ef[index + DZ + DY] * w110;
					fmag += ef[index + DZ + DY + DX] * w111;
					f = x0 * (fmag / r);
					phi += ep[index] * w000;
					phi += ep[index + DX] * w001;
					phi += ep[index + DY] * w010;
					phi += ep[index + DY + DX] * w011;
					phi += ep[index + DZ] * w100;
					phi += ep[index + DZ + DX] * w101;
					phi += ep[index + DZ + DY] * w110;
					phi += ep[index + DZ + DY + DX] * w111;
				} else {
					phi = 2.8372975;
				}
				const real r3 = r * r * r;
				if (r > h) {
					f = f - x0 / r3;
				} else {
					f = f - x0 / (h * h * h);
				}
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim] *= sgn[dim];
				}
				if (r > h) {
					phi = phi - 1.0 / r;
				} else {
					phi = phi - (1.5 * h * h - 0.5 * r * r) / (h * h * h);
				}
			} else {
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
			}
			g[i].g = g[i].g + f;
			g[i].phi += phi;
		}
		g[i].g = g[i].g * m;
		g[i].phi *= m;
	}
}

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
