#include <time.h>
#include <stdio.h>

#define N    (48*1024)
#define M    (48*1024)
#define P    256
#define SIZE 4

float rand_unit_box() {
	return (rand() + 0.5) / (RAND_MAX + 1.0) - 0.5;
}

template<class T>
struct SharedMemory {
	__device__        inline operator T *() {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}

	__device__        inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}
};

__global__
void compute(float *__restrict__ G, const float *__restrict__ Y, const float *__restrict__ X) {
//	const int i = threadIdx.x + blockIdx.x * P;
	auto ys = SharedMemory<float>();
//
//	for (int tile = 0; tile < M / P; tile++) {
//		const int j0 = tile * P * SIZE;
//		int base = threadIdx.x * SIZE;
//		ys[base] = Y[j0 + base];
//		ys[base + 1] = Y[j0 + base + 1];
//		ys[base + 2] = Y[j0 + base + 2];
//		__syncthreads();
//#pragma unroll 128
//		for (int j = 0; j < P; j++) {
//			const int i0 = i * SIZE;
//			const int j0 = j * SIZE;
//			const auto dx = X[i0] - ys[j0];
//			const auto dy = X[i0 + 1] - ys[j0 + 1];
//			const auto dz = X[i0 + 2] - ys[j0 + 2];
//			const auto tmp = rsqrt(dx * dx + dy * dy + dz * dz);
//			const auto r3inv = tmp * tmp * tmp;
//			G[i0] += dx * r3inv ;
//			G[i0 + 1] += dy * r3inv;
//			G[i0 + 2] += dz * r3inv;
//		}
//		__syncthreads();
//	}
//
	const int i = threadIdx.x + blockIdx.x * P;
	const auto G0 = (SIZE+1) * i;
	float g1 = 0.0;
	float g2 = 0.0;
	float g3 = 0.0;
	float g4 = 0.0;
	for (int tile = 0; tile < M / P; tile++) {
		const int j0 = tile * P * SIZE;
		int base = threadIdx.x * SIZE;
		ys[base] = Y[j0 + base];
		ys[base + 1] = Y[j0 + base + 1];
		ys[base + 2] = Y[j0 + base + 2];
		__syncthreads();
		//if (i < N) {
		const auto X0 = SIZE * i;
		const auto x1 = X[X0];
		const auto x2 = X[X0 + 1];
		const auto x3 = X[X0 + 2];
		for (int j = 0; j < P; j++) {
			const auto Y0 = SIZE * j;
			const auto dx1 = x1 - ys[Y0]; // 1 OP
			const auto dx2 = x2 - ys[Y0 + 1]; // 1 OP
			const auto dx3 = x3 - ys[Y0 + 2]; // 1 OP
			const auto r2 = dx1 * dx1 + dx2 * dx2 + dx3 * dx3; // 5 OP
			const auto rinv = rsqrt(r2); // 1 OP
			const auto nrinv3 = -rinv * rinv * rinv; // 3 OP
			g1 = g1 + dx1 * nrinv3; // 2 OP
			g2 = g2 + dx2 * nrinv3; // 2 OP
			g3 = g3 + dx3 * nrinv3; // 2 OP
			g4 = g4 - rinv; // 1 OP
		}
		__syncthreads();
//	}
	}
	G[G0] = g1;
	G[G0 + 1] = g2;
	G[G0 + 2] = g3;
	G[G0 + 3] = g4;
}

int main() {

	float *hostX;
	float *hostY;
	float *hostG;
	float *deviceX;
	float *deviceY;
	float *deviceG;

	cudaSetDeviceFlags (cudaDeviceMapHost);
	cudaHostAlloc((void**) &hostG, (SIZE+1) * N * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable);
	cudaHostAlloc((void**) &hostX, (SIZE) * N * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable);
	cudaHostAlloc((void**) &hostY, (SIZE) * M * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable);
	cudaHostGetDevicePointer((void**) &deviceG, hostG, 0);
	cudaHostGetDevicePointer((void**) &deviceX, hostX, 0);
	cudaHostGetDevicePointer((void**) &deviceY, hostY, 0);

	for (int i = 0; i < N; i++) {
		for (int d = 0; d < SIZE; d++) {
			hostX[SIZE * i] = rand_unit_box();
			hostX[SIZE * i + 1] = rand_unit_box();
			hostX[SIZE * i + 2] = rand_unit_box();
		}
	}
	for (int i = 0; i < M; i++) {
		for (int d = 0; d < SIZE; d++) {
			hostY[SIZE * i] = rand_unit_box();
			hostY[SIZE * i + 1] = rand_unit_box();
			hostY[SIZE * i + 2] = rand_unit_box();
		}
	}

	auto start = time(NULL);
	for (int i = 0; i < 1000; i++) {

		compute<<<N/P,P,P * SIZE*sizeof(float)>>>(deviceG,deviceX,deviceY);
		cudaDeviceSynchronize();
		auto end = time(NULL);
		double ops = (i + 1) * (double) N * (double) M * 20.0 / (1024.0 * 1024.0 * 1024.0 * 1024.0);
		double t = (double) (end - start);
		double flops = ops / t;
		printf("%i %e TFLOP in %e seconds for %e TFLOPS\n", i, ops, t, flops);
	}

}
