#include <ntiger/gravity.hpp>
#include <ntiger/gravity_cuda.hpp>
#include <ntiger/options.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))

#define P 256

cudaArray *eforce1 = 0;
cudaArray *eforce2 = 0;
cudaArray *eforce3 = 0;
cudaArray *epot = 0;

texture<real_type, cudaTextureType3D> ftex1;
texture<real_type, cudaTextureType3D> ftex2;
texture<real_type, cudaTextureType3D> ftex3;
texture<real_type, cudaTextureType3D> ptex;

void set_cuda_ewald_tables(const std::array<ewald_table_t, NDIM> &f, const ewald_table_t &phi) {
	cudaExtent volume = make_cudaExtent(EWALD_NBIN + 1, EWALD_NBIN + 1, EWALD_NBIN + 1);
	cudaChannelFormatDesc fchan1 = cudaCreateChannelDesc<real_type>();
	cudaChannelFormatDesc fchan2 = cudaCreateChannelDesc<real_type>();
	cudaChannelFormatDesc fchan3 = cudaCreateChannelDesc<real_type>();
	cudaChannelFormatDesc pchan = cudaCreateChannelDesc<real_type>();
	CUDA_CHECK(cudaMalloc3DArray(&eforce1, &fchan1, volume));
	CUDA_CHECK(cudaMalloc3DArray(&eforce2, &fchan2, volume));
	CUDA_CHECK(cudaMalloc3DArray(&eforce3, &fchan3, volume));
	CUDA_CHECK(cudaMalloc3DArray(&epot, &pchan, volume));

	cudaMemcpy3DParms fcopy1 = { 0 };
	fcopy1.srcPtr = make_cudaPitchedPtr((void*) f[0].data(), volume.width * sizeof(real_type), volume.width, volume.height);
	fcopy1.dstArray = eforce1;
	fcopy1.extent = volume;
	fcopy1.kind = cudaMemcpyHostToDevice;
	CUDA_CHECK(cudaMemcpy3D(&fcopy1));

	cudaMemcpy3DParms fcopy2 = { 0 };
	fcopy2.srcPtr = make_cudaPitchedPtr((void*) f[1].data(), volume.width * sizeof(real_type), volume.width, volume.height);
	fcopy2.dstArray = eforce2;
	fcopy2.extent = volume;
	fcopy2.kind = cudaMemcpyHostToDevice;
	CUDA_CHECK(cudaMemcpy3D(&fcopy2));

	cudaMemcpy3DParms fcopy3 = { 0 };
	fcopy3.srcPtr = make_cudaPitchedPtr((void*) f[2].data(), volume.width * sizeof(real_type), volume.width, volume.height);
	fcopy3.dstArray = eforce3;
	fcopy3.extent = volume;
	fcopy3.kind = cudaMemcpyHostToDevice;
	CUDA_CHECK(cudaMemcpy3D(&fcopy3));

	cudaMemcpy3DParms pcopy = { 0 };
	pcopy.srcPtr = make_cudaPitchedPtr((void*) phi.data(), volume.width * sizeof(real_type), volume.width, volume.height);
	pcopy.dstArray = epot;
	pcopy.extent = volume;
	pcopy.kind = cudaMemcpyHostToDevice;
	CUDA_CHECK(cudaMemcpy3D(&pcopy));

	for (int i = 0; i < NDIM; i++) {
		ftex1.addressMode[i] = cudaAddressModeClamp;
		ftex2.addressMode[i] = cudaAddressModeClamp;
		ftex3.addressMode[i] = cudaAddressModeClamp;
		ptex.addressMode[i] = cudaAddressModeClamp;

	}
	ftex1.filterMode = cudaFilterModeLinear;
	ftex2.filterMode = cudaFilterModeLinear;
	ftex3.filterMode = cudaFilterModeLinear;
	ptex.filterMode = cudaFilterModeLinear;
	ftex1.normalized = false;
	ftex2.normalized = false;
	ftex3.normalized = false;
	ptex.normalized = false;
	CUDA_CHECK(cudaBindTextureToArray(ftex1, eforce1, fchan1));
	CUDA_CHECK(cudaBindTextureToArray(ftex2, eforce2, fchan2));
	CUDA_CHECK(cudaBindTextureToArray(ftex3, eforce3, fchan3));
	CUDA_CHECK(cudaBindTextureToArray(ptex, epot, pchan));
}

__global__


      __global__
void direct_gravity_kernel(gravity *__restrict__ g, const vect *x, const source *y, int xsize, int ysize, real h, bool ewald) {
	__shared__ gravity
	this_g[P];
	const real h2 = h * h;
	const real h2t15 = 1.5 * h * h;
	const real h3inv = 1.0 / (h * h * h);
	const int i = blockIdx.x;
	const int l = threadIdx.x;
	this_g[l].g = vect(0);
	this_g[l].phi = 0.0;
	for (int j = l; j < ysize; j += P) {
		vect f;
		real phi;
		const auto dx = x[i] - y[j].x; // 3 OP
		const auto m = y[j].m;
		auto x0 = dx;
		vect sgn(1.0);
		if (ewald) {
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
		}
		const real r2 = x0.dot(x0);
		for (int dim = 0; dim < NDIM; dim++) {
			f[dim] = 0.0;
		}
		phi = 0.0;
		if (r2 > 0.0) {
			const auto rinv = rsqrt(r2);            //1 OP
			const auto r3inv = rinv * rinv * rinv;            //2 OP
			if (r2 > h2) {
				phi = phi - rinv;													// 2 OP
				f = f - x0 * r3inv;														// 6 OP
			} else {
				phi = phi - (h2t15 - 0.5 * r2) * h3inv;							//4 OP
				f = f - x0 * h3inv;                                              // 6 OP
			}
			if (ewald) {
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim] *= sgn[dim];														// 3 OP
				}
			}
		}
		this_g[l].g += f * m;                                         // 3 OP
		this_g[l].phi += phi * m;                                      // 1 OP
	}
	__syncthreads();
	for (int N = P / 2; N > 0; N >>= 1) {
		if (l < N) {
			this_g[l].g += this_g[l + N].g;
			this_g[l].phi += this_g[l + N].phi;
		}
		__syncthreads();
	}
	g[i].g = this_g[0].g; // 1 OP
	g[i].phi = this_g[0].phi; // 1 OP
}

__global__
void ewald_gravity_kernel(gravity *__restrict__ g, const vect *x, const source *y, int xsize, int ysize, real h) {
	extern __shared__ int ptr[];
	source *ys = (source*) ptr;
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
			if (r2 > 0.0) {
				const auto rinv = rsqrt(r2);            //1 OP
				general_vect<float, NDIM> I;
				for (int dim = 0; dim < NDIM; dim++) {
					I[dim] = (x0[dim] * dxbininv).get() + 0.5; // 3 * 2 OP
				}
				f[0] += tex3D(ftex1, I[0], I[1], I[2]); // 2 OP
				f[1] += tex3D(ftex2, I[0], I[1], I[2]); // 2 OP
				f[2] += tex3D(ftex3, I[0], I[1], I[2]); // 2 OP
				phi += tex3D(ptex, I[0], I[1], I[2]); // 2 OP
				for (int dim = 0; dim < NDIM; dim++) {
					f[dim] *= sgn[dim];														// 3 OP
				}
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

std::vector<gravity> direct_gravity_cuda(const std::vector<vect> &x, const std::vector<source> &y) {
//	printf( "<-\n" );
	std::vector<gravity> g(x.size());
	double start, stop;
	if (x.size() > 0 && y.size() > 0) {
		bool ewald = options::get().ewald;
		real h = options::get().kernel_size;
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
		if (true) {
			start = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
		}

		direct_gravity_kernel<<<x.size(),P>>>(cg,cx,cy,x.size(),y.size(), h, ewald);

		if (true) {
			static double last_display = 0.0;
			static double t = 0.0;
			static double flops = 0.0;
			cudaDeviceSynchronize();
			stop = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
			t += stop - start;
			flops += x.size() * y.size() * 35;
			if (t > 0.0) {
			//if (t > last_display + 1.0) {
				printf("DIRECT %e TFLOPS\n", flops / 1024.0 / 1024.0 / 1024.0 / t / 1024.0);
				last_display = t;
			}

		}

		CUDA_CHECK(cudaMemcpy(g.data(), cg, x.size() * sizeof(gravity), cudaMemcpyDeviceToHost));
	}
//	printf( ">-\n" );
	return g;
}

std::vector<gravity> ewald_gravity_cuda(const std::vector<vect> &x, const std::vector<source> &y) {
//	printf( "<-\n" );
	std::vector<gravity> g(x.size());
	double start, stop;
	if (x.size() > 0 && y.size() > 0) {
		bool ewald = options::get().ewald;
		real h = options::get().kernel_size;
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

		ewald_gravity_kernel<<<dimGrid, dimBlock,P*sizeof(source)>>>(cg,cx,cy,x.size(),y.size(), h);

		if (true) {
			static double last_display = 0.0;
			static double t = 0.0;
			static double flops = 0.0;
			cudaDeviceSynchronize();
			stop = std::chrono::duration_cast < std::chrono::milliseconds > (std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
			t += stop - start;
			flops += x.size() * y.size() * 100.0;
			if (t > last_display + 1.0) {
				printf("%e TFLOPS\n", flops / 1024.0 / 1024.0 / 1024.0 / t / 1024.0);
				last_display = t;
			}

		}

		CUDA_CHECK(cudaMemcpy(g.data(), cg, x.size() * sizeof(gravity), cudaMemcpyDeviceToHost));
	}
//	printf( ">-\n" );
	return g;
}
