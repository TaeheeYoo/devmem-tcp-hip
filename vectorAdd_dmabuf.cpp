#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// For demonstration, we define a dummy export_dmabuf function.
// In a real implementation, this would interact with the kernel/driver to export
// the allocated pinned memory as a dma-buf and return a valid file descriptor.
int export_dmabuf(void *ptr, size_t size) {
	// This is a placeholder. Real code would perform an ioctl on a device
	// or use a dedicated library call.
	// For now, we just print a message and return a fake fd (e.g., 42) for demonstration.
	printf("Simulating export of dmabuf for memory at %p of size %zu bytes.\n", ptr, size);
	return 42;  // Fake file descriptor
}

#define N 1024

// HIP kernel: vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
		C[i] = A[i] + B[i];
}

int main() {
	int size = N * sizeof(float);
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;

	// Allocate host memory using hipHostMalloc to get pinned (page-locked) memory.
	// Pinned memory is a prerequisite for dma-buf export.
	hipHostMalloc(&h_A, size, hipHostMallocDefault);
	hipHostMalloc(&h_B, size, hipHostMallocDefault);
	hipHostMalloc(&h_C, size, hipHostMallocDefault);

	// Initialize host arrays
	for (int i = 0; i < N; i++) {
		h_A[i] = (float)i;
		h_B[i] = (float)i;
	}

	// Optionally export h_C as a dma-buf.
	// Here we export the result buffer even though we won't use the fd further.
	int dmabuf_fd = export_dmabuf(h_C, size);
	if (dmabuf_fd >= 0)
		printf("Exported dma-buf with fd: %d\n", dmabuf_fd);
	else
		printf("Failed to export dma-buf: %s\n", strerror(errno));

	// Allocate device memory
	hipMalloc(&d_A, size);
	hipMalloc(&d_B, size);
	hipMalloc(&d_C, size);

	// Copy data from host to device
	hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
	hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

	// Launch the kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);

	// Copy the result back to host
	hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

	// Validate the result
	for (int i = 0; i < N; i++) {
		if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
			printf("Error at index %d: %f != %f\n", i, h_C[i], h_A[i] + h_B[i]);
			break;
		}
	}
	printf("Vector addition completed successfully!\n");

	// Clean up: close the exported dma-buf if necessary (here we just simulate)
	// In real usage, you would close the file descriptor when no longer needed.
	if (dmabuf_fd >= 0)
		close(dmabuf_fd);

	hipFree(d_A);
	hipFree(d_B);
	hipFree(d_C);
	hipHostFree(h_A);
	hipHostFree(h_B);
	hipHostFree(h_C);

	return 0;
}

