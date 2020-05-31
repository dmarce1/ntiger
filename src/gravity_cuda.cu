#include <ntiger/gravity.hpp>
#include <ntiger/gravity_cuda.hpp>
#include <ntiger/options.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))

#define P 256

cudaArray *eforce = 0;
cudaArray *epot = 0;

texture<real_type, cudaTextureType3D> ftex;
texture<real_type, cudaTextureType3D> ptex;

void set_cuda_ewald_tables(const ewald_table_t &f, const ewald_table_t &phi) {
	cudaExtent volume = make_cudaExtent(EWALD_NBIN + 1, EWALD_NBIN + 1, EWALD_NBIN + 1);
	cudaChannelFormatDesc fchan = cudaCreateChannelDesc<real_type>();
	cudaChannelFormatDesc pchan = cudaCreateChannelDesc<real_type>();
	CUDA_CHECK(cudaMalloc3DArray(&eforce, &fchan, volume));
	CUDA_CHECK(cudaMalloc3DArray(&epot, &pchan, volume));
	cudaMemcpy3DParms fcopy = { 0 };
	fcopy.srcPtr = make_cudaPitchedPtr((void*) f.data(), volume.width * sizeof(real_type), volume.width, volume.height);
	fcopy.dstArray = eforce;
	fcopy.extent = volume;
	fcopy.kind = cudaMemcpyHostToDevice;
	CUDA_CHECK(cudaMemcpy3D(&fcopy));
	cudaMemcpy3DParms pcopy = { 0 };
	pcopy.srcPtr = make_cudaPitchedPtr((void*) phi.data(), volume.width * sizeof(real_type), volume.width, volume.height);
	pcopy.dstArray = epot;
	pcopy.extent = volume;
	pcopy.kind = cudaMemcpyHostToDevice;
	CUDA_CHECK(cudaMemcpy3D(&pcopy));
	for (int i = 0; i < NDIM; i++) {
		ftex.addressMode[i] = cudaAddressModeClamp;
		ptex.addressMode[i] = cudaAddressModeClamp;

	}
	ftex.filterMode = cudaFilterModePoint;
	ptex.filterMode = cudaFilterModePoint;
	ftex.normalized = false;
	ptex.normalized = false;
	CUDA_CHECK(cudaBindTextureToArray(ftex, eforce, fchan));
	CUDA_CHECK(cudaBindTextureToArray(ptex, epot, pchan));
}

__global__
void gravity_far_kernel_newton(gravity *__restrict__ g, const vect *__restrict__ x, const source *__restrict__ y, int xsize, int ysize) {
	extern __shared__ int ptr[];
	source *ys = (source*) ptr;
	int base = blockIdx.x * blockDim.x;
	int i0 = threadIdx.x + base;
	int i = min(i0, xsize - 1);
	gravity this_g;
	const real dxbin = 0.5 / EWALD_NBIN;                                   // 1 OP
	this_g.g = vect(0);
	this_g.phi = 0.0;
	for (int tile = 0; tile < (ysize + P - 1) / P; tile++) {
		const int j0 = tile * P;
		const int jmax = min((tile + 1) * P, ysize) - j0;
		ys[threadIdx.x] = y[threadIdx.x + j0];
		__syncthreads();
#pragma loop unroll 128
		for (int j = 0; j < jmax; j++) {
			vect f;
			real phi;
			const auto dx = x[i] - ys[j].x; // 3 OP
			const auto m = ys[j].m;
			const auto r2 = dx.dot(dx); // 6 OP
			const auto rinv = rsqrt(r2 + 1.e-20);            //1 OP
			const auto mr3inv = m * rinv * rinv * rinv;            //3 OP
			this_g.g -= dx * mr3inv;            //6 OP
			this_g.phi -= rinv * m;                //2 OP
		}
		__syncthreads();
	}
	if (i == i0) {
		g[i].g = this_g.g; // 1 OP
		g[i].phi = this_g.phi; // 1 OP
	}
}

__global__
void gravity_far_kernel_ewald(gravity *__restrict__ g, const vect *x, const source *y, int xsize, int ysize) {
	extern __shared__ int ptr[];
	source *ys = (source*) ptr;
	int base = blockIdx.x * blockDim.x;
	int i0 = threadIdx.x + base;
	int i = min(i0, xsize - 1);
	gravity this_g;
	const real dxbininv = (EWALD_NBIN << 1);                                   // 1 OP
	this_g.g = vect(0);
	this_g.phi = 0.0;
	for (int tile = 0; tile < (ysize + P - 1) / P; tile++) {
		const int j0 = tile * P;
		const int jmax = min((tile + 1) * P, ysize) - j0;
		ys[threadIdx.x] = y[min(threadIdx.x + j0, ysize - 1)];
		__syncthreads();
//#pragma loop unroll 128
		for (int j = 0; j < jmax; j++) {
			vect f;
			real phi;
			const auto dx = x[i] - ys[j].x; // 3 OP
			const auto m = ys[j].m;
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
			const real r2 = x0.dot(x0);
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim] = 0.0;
			}
			phi = 0.0;
			// Skip ewald
			real fmag = 0.0;
			const auto rinv = rsqrt(r2 + 1.0e-20);            //1 OP
			general_vect<float, NDIM> I;
			general_vect<float, NDIM> Ip;
			general_vect<real, NDIM> wm;
			general_vect<real, NDIM> w;
			for (int dim = 0; dim < NDIM; dim++) {
				I[dim] = fminf(int((x0[dim] * dxbininv).get()), EWALD_NBIN - 1); // 3 * 2 OP
				Ip[dim] = I[dim] + 1.0;                                        // 3 * 1 OP
				wm[dim] = (x0[dim] * dxbininv - I[dim]);                       // 3 * 2 OP
				w[dim] = 1.0 - wm[dim];                                        // 3 * 1 OP
			}
			const auto w00 = w[0] * w[1]; // 1 OP
			const auto w01 = w[0] * w[1]; // 1 OP
			const auto w10 = w[0] * wm[1]; // 1 OP
			const auto w11 = w[0] * wm[1]; // 1 OP
			const auto w000 = w00 * w[2]; // 1 OP
			const auto w001 = w00 * wm[2]; // 1 OP
			const auto w010 = w01 * w[2]; // 1 OP
			const auto w011 = w01 * wm[2]; // 1 OP
			const auto w100 = w10 * w[2]; // 1 OP
			const auto w101 = w10 * wm[2]; // 1 OP
			const auto w110 = w11 * w[2]; // 1 OP
			const auto w111 = w11 * wm[2]; // 1 OP
			fmag += tex3D(ftex, I[0], I[1], I[2]) * w000; // 2 OP
			fmag += tex3D(ftex, I[0], I[1], Ip[2]) * w001; // 2 OP
			fmag += tex3D(ftex, I[0], Ip[1], I[2]) * w010; // 2 OP
			fmag += tex3D(ftex, I[0], Ip[1], Ip[2]) * w011; // 2 OP
			fmag += tex3D(ftex, Ip[0], I[1], I[2]) * w100; // 2 OP
			fmag += tex3D(ftex, Ip[0], I[1], Ip[2]) * w101; // 2 OP
			fmag += tex3D(ftex, Ip[0], Ip[1], I[2]) * w110; // 2 OP
			fmag += tex3D(ftex, Ip[0], Ip[1], Ip[2]) * w111; // 2 OP
			f = x0 * (fmag * rinv); // 6 OP
			phi += tex3D(ptex, I[0], I[1], I[2]) * w000; // 2 OP
			phi += tex3D(ptex, I[0], I[1], Ip[2]) * w001; // 2 OP
			phi += tex3D(ptex, I[0], Ip[1], I[2]) * w010; // 2 OP
			phi += tex3D(ptex, I[0], Ip[1], Ip[2]) * w011; // 2 OP
			phi += tex3D(ptex, Ip[0], I[1], I[2]) * w100; // 2 OP
			phi += tex3D(ptex, Ip[0], I[1], Ip[2]) * w101; // 2 OP
			phi += tex3D(ptex, Ip[0], Ip[1], I[2]) * w110; // 2 OP
			phi += tex3D(ptex, Ip[0], Ip[1], Ip[2]) * w111; // 2 OP
			const auto r3inv = rinv * rinv * rinv;            //2 OP
			phi = phi - rinv;													// 2 OP
			f = f - x0 * r3inv;														// 6 OP
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim] *= sgn[dim];														// 3 OP
			}
			this_g.g += f * m;                                         // 3 OP
			this_g.phi += phi * m;                                      // 1 OP
		}
		__syncthreads();
	}
	if (i == i0) {
		g[i].g = this_g.g; // 1 OP
		g[i].phi = this_g.phi; // 1 OP
	}
}


__global__
void gravity_near_kernel_newton(gravity *__restrict__ g, const vect *__restrict__ x, const vect *__restrict__ y, int xsize, int ysize, real h, real m) {
	extern __shared__ int ptr[];
	vect *ys = (vect*) ptr;
	int base = blockIdx.x * blockDim.x;
	int i0 = threadIdx.x + base;
	int i = min(i0, xsize - 1);
	gravity this_g;
	const real dxbin = 0.5 / EWALD_NBIN;                                   // 1 OP
	const real h2 = h * h;
	const real h2t15 = 1.5 * h * h;
	const real h3inv = 1.0 / (h * h * h);
	this_g.g = vect(0);
	this_g.phi = 0.0;
	for (int tile = 0; tile < (ysize + P - 1) / P; tile++) {
		const int j0 = tile * P;
		const int jmax = min((tile + 1) * P, ysize) - j0;
		ys[threadIdx.x] = y[threadIdx.x + j0];
		__syncthreads();
#pragma loop unroll 128
		for (int j = 0; j < jmax; j++) {
			vect f;
			real phi;
			const auto dx = x[i] - ys[j]; // 3 OP
			const auto r2 = dx.dot(dx); // 6 OP
			const auto rinv = rsqrt(r2 + 1.e-20);            //1 OP
			if (r2 > h2) {
				const auto r3inv = rinv * rinv * rinv;            //2 OP
				this_g.g -= dx * r3inv;            //6 OP
				this_g.phi -= rinv;                //1 OP
			} else {
				this_g.g -= dx * h3inv;
				this_g.phi -= (h2t15 - 0.5 * r2) * h3inv;
			}
		}
		__syncthreads();
	}
	if (i == i0) {
		g[i].g = this_g.g * m; // 1 OP
		g[i].phi = this_g.phi * m; // 1 OP
	}
}

__global__
void gravity_near_kernel_ewald(gravity *__restrict__ g, const vect *x, const vect *y, int xsize, int ysize, real h, real m) {
	extern __shared__ int ptr[];
	vect *ys = (vect*) ptr;
	int base = blockIdx.x * blockDim.x;
	int i0 = threadIdx.x + base;
	int i = min(i0, xsize - 1);
	gravity this_g;
	const real dxbininv = (EWALD_NBIN << 1);                                   // 1 OP
	const real h2 = h * h;
	const real h2t15 = 1.5 * h * h;
	const real h3inv = 1.0 / (h * h * h);
	this_g.g = vect(0);
	this_g.phi = 0.0;
	for (int tile = 0; tile < (ysize + P - 1) / P; tile++) {
		const int j0 = tile * P;
		const int jmax = min((tile + 1) * P, ysize) - j0;
		ys[threadIdx.x] = y[min(threadIdx.x + j0, ysize - 1)];
		__syncthreads();
//#pragma loop unroll 128
		for (int j = 0; j < jmax; j++) {
			vect f;
			real phi;
			const auto dx = x[i] - ys[j]; // 3 OP
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
			const real r2 = x0.dot(x0);
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim] = 0.0;
			}
			phi = 0.0;
			// Skip ewald
			real fmag = 0.0;
			const auto rinv = rsqrt(r2 + 1.0e-20);            //1 OP
			general_vect<float, NDIM> I;
			general_vect<float, NDIM> Ip;
			general_vect<real, NDIM> wm;
			general_vect<real, NDIM> w;
			for (int dim = 0; dim < NDIM; dim++) {
				I[dim] = fminf(int((x0[dim] * dxbininv).get()), EWALD_NBIN - 1); // 3 * 2 OP
				Ip[dim] = I[dim] + 1.0;                                        // 3 * 1 OP
				wm[dim] = (x0[dim] * dxbininv - I[dim]);                       // 3 * 2 OP
				w[dim] = 1.0 - wm[dim];                                        // 3 * 1 OP
			}
			const auto w00 = w[0] * w[1]; // 1 OP
			const auto w01 = w[0] * w[1]; // 1 OP
			const auto w10 = w[0] * wm[1]; // 1 OP
			const auto w11 = w[0] * wm[1]; // 1 OP
			const auto w000 = w00 * w[2]; // 1 OP
			const auto w001 = w00 * wm[2]; // 1 OP
			const auto w010 = w01 * w[2]; // 1 OP
			const auto w011 = w01 * wm[2]; // 1 OP
			const auto w100 = w10 * w[2]; // 1 OP
			const auto w101 = w10 * wm[2]; // 1 OP
			const auto w110 = w11 * w[2]; // 1 OP
			const auto w111 = w11 * wm[2]; // 1 OP
			fmag += tex3D(ftex, I[0], I[1], I[2]) * w000; // 2 OP
			fmag += tex3D(ftex, I[0], I[1], Ip[2]) * w001; // 2 OP
			fmag += tex3D(ftex, I[0], Ip[1], I[2]) * w010; // 2 OP
			fmag += tex3D(ftex, I[0], Ip[1], Ip[2]) * w011; // 2 OP
			fmag += tex3D(ftex, Ip[0], I[1], I[2]) * w100; // 2 OP
			fmag += tex3D(ftex, Ip[0], I[1], Ip[2]) * w101; // 2 OP
			fmag += tex3D(ftex, Ip[0], Ip[1], I[2]) * w110; // 2 OP
			fmag += tex3D(ftex, Ip[0], Ip[1], Ip[2]) * w111; // 2 OP
			f = x0 * (fmag * rinv); // 6 OP
			phi += tex3D(ptex, I[0], I[1], I[2]) * w000; // 2 OP
			phi += tex3D(ptex, I[0], I[1], Ip[2]) * w001; // 2 OP
			phi += tex3D(ptex, I[0], Ip[1], I[2]) * w010; // 2 OP
			phi += tex3D(ptex, I[0], Ip[1], Ip[2]) * w011; // 2 OP
			phi += tex3D(ptex, Ip[0], I[1], I[2]) * w100; // 2 OP
			phi += tex3D(ptex, Ip[0], I[1], Ip[2]) * w101; // 2 OP
			phi += tex3D(ptex, Ip[0], Ip[1], I[2]) * w110; // 2 OP
			phi += tex3D(ptex, Ip[0], Ip[1], Ip[2]) * w111; // 2 OP
			const auto r3inv = rinv * rinv * rinv;            //2 OP
			if (r2 > h2) {
				phi = phi - rinv;													// 2 OP
				f = f - x0 * r3inv;														// 6 OP
			} else {
				phi = phi - (h2t15 - 0.5 * r2) * h3inv;							//4 OP
				f = f - x0 * h3inv;                                              // 6 OP
			}
			for (int dim = 0; dim < NDIM; dim++) {
				f[dim] *= sgn[dim];														// 3 OP
			}
			this_g.g += f;                                         // 3 OP
			this_g.phi += phi;                                      // 1 OP
		}
		__syncthreads();
	}
	if (i == i0) {
		g[i].g = this_g.g * m; // 1 OP
		g[i].phi = this_g.phi * m; // 1 OP
	}
}

std::vector<gravity> gravity_near_cuda(const std::vector<vect> &x, const std::vector<vect> &y, bool ewald) {
	std::vector<gravity> g(x.size());
	bool time = true;
	double start, stop;
	if (x.size() > 0 && y.size() > 0) {
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
		dim3 dimBlock(P, 1);
		dim3 dimGrid((x.size() + P - 1) / P, 1);
		if (time) {
			start = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
		}
			if (ewald) {
		gravity_near_kernel_ewald<<<dimGrid, dimBlock,P*sizeof(vect)>>>(cg,cx,cy,x.size(),y.size(),h,m);
	} else {
	gravity_near_kernel_newton<<<dimGrid, dimBlock,P*sizeof(vect)>>>(cg,cx,cy,x.size(),y.size(),h,m);
}

if (time) {
	cudaDeviceSynchronize();
	static double last_display = 0.0;
	static double t = 0.0;
	static double flops = 0.0;
	stop = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
	t += stop - start;
	flops += x.size() * y.size() * (ewald ? 101.0 : 20.0);
	if (t > last_display + 1.0) {
		printf("%e TFLOPS\n", flops / 1024.0 / 1024.0 / 1024.0 / t / 1024.0);
		last_display = t;
	}

}

CUDA_CHECK(cudaMemcpy(g.data(), cg, x.size() * sizeof(gravity), cudaMemcpyDeviceToHost));
}
return g;
}

std::vector<gravity> gravity_far_cuda(const std::vector<vect> &x, const std::vector<source> &y) {
	std::vector<gravity> g(x.size());
	double start, stop;
	if (x.size() > 0 && y.size() > 0) {
		bool ewald = options::get().ewald;
		static thread_local gravity *cg = nullptr;
		static thread_local vect *cx = nullptr;
		static thread_local source *cy = nullptr;
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
			CUDA_CHECK(cudaMalloc((void** ) &cy, sizeof(source) * y.size()));
			ymax = y.size();
		}
		CUDA_CHECK(cudaMemcpy(cx, x.data(), x.size() * sizeof(vect), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(cy, y.data(), y.size() * sizeof(source), cudaMemcpyHostToDevice));
		dim3 dimBlock(P, 1);
		dim3 dimGrid((x.size() + P - 1) / P, 1);
		if (true) {
			start = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
		}
		if (ewald) {
		gravity_far_kernel_ewald<<<dimGrid, dimBlock,P*sizeof(source)>>>(cg,cx,cy,x.size(),y.size());
	} else {
		gravity_far_kernel_newton<<<dimGrid, dimBlock,P*sizeof(source)>>>(cg,cx,cy,x.size(),y.size());
	}

		if (true) {
			static double last_display = 0.0;
			static double t = 0.0;
			static double flops = 0.0;
		cudaDeviceSynchronize();
			stop = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
			t += stop - start;
			flops += x.size() * y.size() * (ewald ? 101.0 : 20.0);
			if (t > last_display + 1.0) {
				printf("%e TFLOPS\n", flops / 1024.0 / 1024.0 / 1024.0 / t / 1024.0);
				last_display = t;
			}

		}

CUDA_CHECK(cudaMemcpy(g.data(), cg, x.size() * sizeof(gravity), cudaMemcpyDeviceToHost));
}
return g;
}
