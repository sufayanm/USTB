#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <complex.h>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "..\beamform_iq_kernel.cuh"
#include "..\beamform_rf_kernel.cuh"

// Error check function
#define gpuErrchk(arg) { gpuAssert((arg), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		printf("CUDA error: %s in file %s line %d\n", cudaGetErrorString(code), file, line);
	}
}

float* randf(float min, float max, size_t N);
float complex* randc(float min, float max, size_t N);
float* linspace(float start, float end, size_t N);

int main()
{
	// Sizes
	const size_t N_frames = 1000;
	const size_t N_waves = 5;
	const size_t N_times = 1500;
	const size_t N_channels = 128;

	// variables
	const float Fs = 4.0e6;
	const float t0 = 0.0;

	const size_t N_pixels = 256 * 256;
	const bool IQ = true;

	// Transfer delay and apodization matrices to GPU
	// Retrieve pointer to host arrays
	float* host_ch_data = randf(-1000.0, 1000.0, N_times * N_channels * N_waves * N_frames);

	float* host_bf_data;

	float* host_tx_delay = linspace((float) 1.0e-6, (float) 5.0e-6, N_waves * N_pixels);
	float* host_tx_apod = mxGetSingles(M_APO_TX);
	float* host_rx_delay = mxGetSingles(M_DELAY_RX);
	float* host_rx_apod = mxGetSingles(M_APO_RX);

	// Allocate device memory
	float* device_tx_delay;
	float* device_tx_apod;
	float* device_rx_delay;
	float* device_rx_apod;

	gpuErrchk(cudaMalloc((void**)&device_tx_delay, N_pixels * N_waves * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&device_tx_apod, N_pixels * N_waves * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&device_rx_delay, N_pixels * N_channels * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&device_rx_apod, N_pixels * N_channels * sizeof(float)));

	// Transfer data
	gpuErrchk(cudaMemcpy(device_tx_delay, host_tx_delay, N_pixels * N_waves * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_tx_apod, host_tx_apod, N_pixels * N_waves * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_rx_delay, host_rx_delay, N_pixels * N_channels * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(device_rx_apod, host_rx_apod, N_pixels * N_channels * sizeof(float), cudaMemcpyHostToDevice));

	// Allocate device memory for beamformed data
	cuFloatComplex* device_bf_data[2];

	for (size_t n = 0; n < 2; n++)
	{
		gpuErrchk(cudaMalloc((void**)&device_bf_data[n], N_pixels * sizeof(cuFloatComplex)));
	}

	// Allocate an array of 1D Layered cudaArray and a cudaTextureObjects
	// Need 2 elements in the array to allow for asynchronous operations
	cudaArray** device_ch_data = (cudaArray**)malloc(2 * sizeof(cudaArray*)); // Array of pointers to cudaArrays
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat); // channel descriptor for a cuFloatComplex type.
	cudaTextureObject_t* tex = (cudaTextureObject_t*)malloc(2 * sizeof(cudaTextureObject_t));

	for (size_t n = 0; n < 2; n++)
	{
		gpuErrchk(cudaMalloc3DArray(&device_ch_data[n], &channelDesc, make_cudaExtent(N_times, 0, N_channels * N_waves), cudaArrayLayered)); // Allocate 1D Layered texture

		// Input data properties
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = device_ch_data[n];

		// Texture properties
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.filterMode = cudaFilterModeLinear; // linear interpolation between texels
		texDesc.normalizedCoords = 0; // coordinates are not normalized [0, 1, ..., N_times-1] and [0, 1, ..., N_channels-1]
		texDesc.addressMode[0] = cudaAddressModeBorder; // out of bound coordinates are 0
		texDesc.readMode = cudaReadModeElementType;

		// Texture Object
		gpuErrchk(cudaCreateTextureObject(&tex[n], &resDesc, &texDesc, NULL));
	}

	// Define block_size and N_blocks
	dim3 block_size = dim3(256);
	dim3 N_blocks = dim3((N_pixels + block_size.x - 1) / block_size.x);

	// Setupt cudaStream for asynchronous operations
	cudaStream_t* frame_stream = (cudaStream_t*)malloc(2 * sizeof(cudaStream_t));
	for (size_t n = 0; n < 2; n++)
	{
		gpuErrchk(cudaStreamCreate(&frame_stream[n]));
	}

	// cudaMemcpy3D properties
	cudaMemcpy3DParms memcpyParams;
	memset(&memcpyParams, 0, sizeof(cudaMemcpy3DParms));
	memcpyParams.extent = make_cudaExtent(N_times, 1, N_channels * N_waves); // cudaExtent object for a 1D layered texture;
	memcpyParams.kind = cudaMemcpyHostToDevice;


	// Beamforming loop
	for (size_t n_frame = 0; n_frame < N_frames; n_frame++)
	{
		// Use module 2 operator to select which cudaStream to send the data to

		// Copy channel data into dedicated texture memory
		memcpyParams.dstArray = device_ch_data[n_frame % 2];
		memcpyParams.srcPtr = make_cudaPitchedPtr(&host_ch_data[n_frame * N_waves * N_channels * N_times], N_times * sizeof(cuFloatComplex), N_times, 1);
		gpuErrchk(cudaMemcpy3DAsync(&memcpyParams, frame_stream[n_frame % 2]));

		// Set device beamformed data to 0
		gpuErrchk(cudaMemsetAsync(device_bf_data[n_frame % 2], 0, N_pixels * sizeof(cuFloatComplex), frame_stream[n_frame % 2]));

		// Call beamforming kernel
		if (!IQ)
		{
			beamform << < N_blocks, block_size, 0, frame_stream[n_frame % 2] >> > (N_pixels, N_channels, N_waves, Fs, device_bf_data[n_frame % 2], tex[n_frame % 2], device_tx_delay,
				device_rx_delay, device_tx_apod, device_rx_apod, t0, t0 * Fs);
			gpuErrchk(cudaPeekAtLastError());
		}
		else
		{
			beamform_iq << < N_blocks, block_size, 0, frame_stream[n_frame % 2] >> > (N_pixels, N_channels, N_waves, Fs, device_bf_data[n_frame % 2], tex[n_frame % 2], device_tx_delay,
				device_rx_delay, device_tx_apod, device_rx_apod, t0, wd, t0 * Fs);
			gpuErrchk(cudaPeekAtLastError());
		}

		// Transfer beamformed data back to host
		gpuErrchk(cudaMemcpyAsync(&host_bf_data[n_frame * N_pixels], device_bf_data[n_frame % 2], N_pixels * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost, frame_stream[n_frame % 2]));
	} // end of frame loop

	for (size_t n = 0; n < 2; n++)
	{
		// Destroy cudaStreams
		gpuErrchk(cudaStreamDestroy(frame_stream[n]));

		// Free Texture memory
		gpuErrchk(cudaFreeArray(device_ch_data[n]));
		gpuErrchk(cudaDestroyTextureObject(tex[n]));

		// Free beamformed data memory
		gpuErrchk(cudaFree(device_bf_data[n]));
	}

	gpuErrchk(cudaFree(device_tx_apod));
	gpuErrchk(cudaFree(device_tx_delay));
	gpuErrchk(cudaFree(device_rx_apod));
	gpuErrchk(cudaFree(device_rx_delay));

	// Unpin host memory
	gpuErrchk(cudaHostUnregister(host_ch_data));
	gpuErrchk(cudaHostUnregister(host_bf_data));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	gpuErrchk(cudaDeviceReset());

	return 0;
}

float* randf(float min, float max, size_t N)
{
	float* x = (float*)malloc(N * sizeof(float));

	float range = max - min;

	for (size_t i = 0; i < N; i++)
	{
		x[i] = (range * (float)rand() / (float)RAND_MAX) + min;
	}

	return x;
}

float* linspace(float fStart, float fEnd, size_t N)
{
	float* x = (float*)malloc(N * sizeof(float));

	float step = (fEnd - fStart) / (float)(N - 1);

	for (size_t i = 0; i < N; i++)
	{
		x[i] = fStart + ((float)i * step);
	}

	return x;
}

float complex* randc(float min, float max, size_t N)
{
	float complex* x = (float complex*) malloc(N * sizeof(float complex));

	float range = max - min;

	for (size_t i = 0; i < N; i++)
	{
		x[i] = (range * (float)rand() / (float)RAND_MAX) + min;
	}

	return x;
}

