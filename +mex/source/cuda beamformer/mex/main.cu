/*================================================
 *
 * mex.cu - CUDA general beamformer for USTB
 *
 *
 * MEX file
 *
 *================================================*/
#include <mex.h>
#include <matrix.h>

#include <math.h>
#include <stdbool.h>
#include <string.h>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "..\beamform_iq_kernel.cuh"
#include "..\beamform_rf_kernel.cuh"

 /* ERROR CHECK FUNCTIONS */
#define gpuErrchk(arg) { gpuAssert((arg), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:GPU", "CUDA error: %s in file %s line %d\n", cudaGetErrorString(code), file, line);
	}
}

#define mexErrchk(arg)  {mexAssert((arg), __FILE__, __LINE__); }
inline void mexAssert(int err, const char* file, int line)
{
	if (err)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:GPU", "MEX error in file %s line %d\n", file, line);
	}
}

// Constants
#define M_EPS 1E-6
#define M_PI acosf(-1.0)
#define M_BLOCK_SIZE 256

// Compulsory inputs
#define	M_P         prhs[0]	// channel_data [time, channel, wave, frame]
#define	M_FS		prhs[1] // sampling frequency (Hz)
#define M_T0		prhs[2]	// initial time (s)

#define	M_APO_TX	prhs[3]	// transmit apodization [pixel, wave]
#define	M_APO_RX	prhs[4]	// receive apodization [pixel, channel]

#define	M_DELAY_TX  prhs[5]	// transmit delay [pixel, wave]
#define	M_DELAY_RX  prhs[6]	// receive delay [pixel, channel]

#define	M_FD		prhs[7] // modulation frequency (Hz)
#define	M_SUM		prhs[8] // sum mode 0 -> NONE, 1->RX, 2->TX, 3->BOTH

// Optional input
#define	M_VERBOSE	prhs[9] // verbose flag [Optional]

// Output
#define	M_D			plhs[0] // delayed data [pixel, channel, wave, frame]

// Enumeration variable
enum SUM_DIMENSION { NONE = 0, RX, TX, BOTH };

#define VERSION "1.0.0"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{

	// CHECK NUMBER OF ARGUMENTS
	if (nrhs < 9)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:nrhs", "Too few input arguments");
	}
	if (nrhs > 10)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:nrhs", "Too many input arguments");
	}
	if (nlhs > 1)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:nlhs", "Too many output arguments");
	}

	// VERBOSE FLAG
	bool verbose;
	if (nrhs == 10)
	{
		verbose = *mxGetLogicals(M_VERBOSE);
	}
	else
	{
		verbose = false;
	}

	// SUM (NONE/RX/TX/BOTH)
	SUM_DIMENSION SUM = (SUM_DIMENSION)*mxGetInt32s(M_SUM);

	if (SUM != 3)
	{
		mexErrMsgTxt("In this implementation only uff.dimension.both is supported");
	}

	// CHANNEL DATA
	// check dimensions
	size_t n_dim = (size_t)mxGetNumberOfDimensions(M_P);
	if (n_dim < 2 || n_dim > 4)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Unknown channel data format. Expected from 2 to 4 dimensions: [time, channel, wave, frame]");
	}
	// check size
	size_t* p_dim = (size_t*)mxGetDimensions(M_P);

	size_t N_times = p_dim[0];			// number of time samples
	size_t N_channels = p_dim[1];		// number of channels
	size_t N_waves;						// number of waves
	if (n_dim > 2)
	{
		N_waves = p_dim[2];
	}
	else
	{
		N_waves = 1;
	}
	size_t N_frames;					// number of frames
	if (n_dim > 3)
	{
		N_frames = p_dim[3];
	}
	else
	{
		N_frames = 1;
	}

	// check single precision
	if (mxIsDouble(M_P))
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Float", "The channel data must be single precision.");
	}

	// complex or real
	bool complex_data = mxIsComplex(M_P);

	// TX DELAY
	// check dimensions
	n_dim = (size_t)mxGetNumberOfDimensions(M_DELAY_TX);
	if (n_dim > 2)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Unknown transmit delay format. Expected 2 dimensions: [pixel, wave]");
	}
	// check size
	p_dim = (size_t*)mxGetDimensions(M_DELAY_TX);

	size_t N_pixels = p_dim[0];		// number of pixels

	size_t N_waves_check;			// number of waves
	if (n_dim == 2)
	{
		N_waves_check = p_dim[1];
	}
	else
	{
		N_waves_check = 1;
	}
	if (N_waves != N_waves_check)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Number of waves in channel data & transmit delays do not match.");
	}
	// check single precision
	if (mxIsDouble(M_DELAY_TX))
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Float", "The transmit delay must be single precision.");
	}

	// DELAY RX
	// check dimensions
	n_dim = (size_t)mxGetNumberOfDimensions(M_DELAY_RX);

	if (n_dim > 2)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Unknown receive delay format. Expected 2 dimensions: [pixel, channel]");
	}
	// check size
	p_dim = (size_t*)mxGetDimensions(M_DELAY_RX);

	size_t N_pixels_check = p_dim[0];                  // number of pixels
	if (N_pixels != N_pixels_check)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Number of pixels in transmit and receive delays do not match.");
	}
	size_t N_channels_check;						// number of channels
	if (n_dim == 2)
	{
		N_channels_check = p_dim[1];
	}
	else
	{
		N_channels_check = 1;
	}
	if (N_channels != N_channels_check)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Number of channels in channel data & receive delays do not match.");
	}
	// check single precision
	if (mxIsDouble(M_DELAY_RX))
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Float", "The receive delay must be single precision.");
	}

	// APO TX
	n_dim = (size_t)mxGetNumberOfDimensions(M_APO_TX);
	if (n_dim > 2)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Unknown transmit apodization format. Expected 2 dimensions: [pixel, wave]");
	}
	// check size
	p_dim = (size_t*)mxGetDimensions(M_APO_TX);

	N_pixels_check = p_dim[0];   // number of pixels

	if (n_dim == 2)
	{
		N_waves_check = p_dim[1];
	}
	else
	{
		N_waves_check = 1;
	}
	if (N_waves != N_waves_check)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Number of waves in channel data & transmit delays do not match.");
	}
	// check single precision
	if (mxIsDouble(M_APO_TX))
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Float", "The transmit apodization must be single precision.");
	}

	// APO RX
	// check dimensions
	n_dim = mxGetNumberOfDimensions(M_APO_RX);
	if (n_dim > 2)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Unknown receive apodization format. Expected 2 dimensions: [pixel, channel]");
	}
	// check size
	p_dim = (size_t*)mxGetDimensions(M_APO_RX);

	N_pixels_check = p_dim[0];       // number of pixels
	if (N_pixels != N_pixels_check)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Number of pixels in transmit and receive apodization do not match.");
	}

	if (n_dim == 2)
	{
		N_channels_check = p_dim[1];
	}
	else
	{
		N_channels_check = 1;
	}
	if (N_channels != N_channels_check)
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Dimensions", "Number of channels in channel data & receive delays do not match.");
	}

	// check single precision
	if (mxIsDouble(M_APO_RX))
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Float", "The receive delay must be single precision.");
	}

	// SAMPLING FREQUENCY
	// check dimensions
	if (!mxIsScalar(M_FS))
	{
		mexErrMsgTxt("The sampling frequency should be an escalar");
	}
	// check single precision
	if (mxIsDouble(M_FS))
	{
		mexErrMsgTxt("The sampling frequency should be single precision");
	}
	// read data
	float Fs = *mxGetSingles(M_FS);	// Sampling frequency


	// INITIAL TIME
	// check dimensions
	if (!mxIsScalar(M_T0))
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Float", "The the initial time must be a scalar.");
	}
	// check single single precision
	if (mxIsDouble(M_T0))
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Float", "The initial time must be single precision.");
	}
	// read data
	float t0 = *mxGetSingles(M_T0);


	// MODULATION FREQUENCY
	// check dimension
	if (!mxIsScalar(M_FD))
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Scalar", "The modulation frequency must be a scalar.");
	}
	// check single precision
	if (mxIsDouble(M_FD))
	{
		mexErrMsgIdAndTxt("Toolbox:SRP_SRC:Float", "The modulation frequency must be single precision.");
	}
	// read data
	float Fd = *mxGetSingles(M_FD);

	float wd;
	bool IQ;
	if (fabsf(Fd) > M_EPS)
	{
		wd = 2 * M_PI * Fd;
		IQ = true;
	}
	else
	{
		wd = 0.0;
		IQ = false;
	}

	// OUTPUT MATRIX
	size_t out_size[4];
	out_size[0] = N_pixels;  // pixels
	out_size[1] = 1;  // channels
	out_size[2] = 1;  // waves
	out_size[3] = N_frames;  // frames

	// POINTER TO BEAMFORMED DATA
	M_D = mxCreateNumericArray(4, (const size_t*)&out_size, mxSINGLE_CLASS, mxCOMPLEX);
	mxComplexSingle* host_bf_data = mxGetComplexSingles(M_D);
	gpuErrchk(cudaHostRegister(host_bf_data, out_size[0] * out_size[1] * out_size[2] * out_size[3] * sizeof(mxComplexSingle), cudaHostRegisterDefault)); // Pin paged memory for asynchronous transfers

	// POINTER TO CHANNEL_DATA
	mxComplexSingle* host_ch_data = mxGetComplexSingles(M_P);
	gpuErrchk(cudaHostRegister(host_ch_data, N_times * N_channels * N_waves * N_frames * sizeof(mxComplexSingle), cudaHostRegisterDefault)); // Pin paged memory for asynchronous transfers

	// VERBOSE LOG
	if (verbose)
	{
		mexPrintf("---------------------------------------------------------------\n");
		mexPrintf(" USTB CUDA General beamformer\n");
		mexPrintf("---------------------------------------------------------------\n");
		mexPrintf(" Single precision\n");
		mexPrintf(" Vers:  %s\n", VERSION);
		mexPrintf(" Auth:  Alfonso Rodriguez-Molares <alfonso.r.molares@ntnu.no>\n");
		mexPrintf(" Auth:  Stefano Fiorentini		  <stefano.fiorentini@ntnu.no>\n");
		mexPrintf(" Date:  2022/24/10\n");
		mexPrintf("---------------------------------------------------------------\n");

		if (complex_data)
		{
			mexPrintf("Data Type                       Complex\n");
		}
		else
		{
			mexPrintf("Data Type                       Real\n");
		}
		switch (SUM)
		{
		case NONE:
			mexPrintf("Sum                             None\n");
			break;
		case RX:
			mexPrintf("Sum                             Channels\n");
			break;
		case TX:
			mexPrintf("Sum                             Waves\n");
			break;
		case BOTH:
			mexPrintf("Sum                             Both channels and waves\n");
			break;
		}
		mexPrintf("Time Samples                    %i\n", N_times);
		mexPrintf("Channels                        %i\n", N_channels);
		mexPrintf("Waves                           %i\n", N_waves);
		mexPrintf("Frames                          %i\n", N_frames);
		mexPrintf("Pixels						   %i\n", N_pixels);
		mexPrintf("Sampling frequency			   %0.2f MHz\n", Fs / 1e6);
		mexPrintf("Initial time					   %0.2f us\n", t0 * 1e6);
		mexPrintf("Modulation frequency            %0.2f MHz\n", Fd / 1e6);
		if (IQ)
		{
			mexPrintf("IQ data                         true\n");
		}
		else
		{
			mexPrintf("IQ data                         false\n");
		}
		mexPrintf("---------------------------------------------------------------\n");
		mexPrintf("Output data size					   %d x %d x %d x %d\n", out_size[0], out_size[1], out_size[2], out_size[3]);
		mexPrintf("---------------------------------------------------------------\n");
	}

	// Transfer delay and apodization matrices to GPU
	// Retrieve pointer to host arrays
	float* host_tx_delay = mxGetSingles(M_DELAY_TX);
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
	dim3 block_size = dim3(M_BLOCK_SIZE);
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
}