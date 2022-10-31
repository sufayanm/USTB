#include <cuda_runtime.h>
#include <cuComplex.h>

// IQ Beamforming kernel
__global__ void beamform_iq(const size_t N_pixels, const size_t N_channels, const size_t N_waves, const float Fs, cuFloatComplex* bf_data, const cudaTextureObject_t tex,
	const float* tx_delay, const float* rx_delay, const float* tx_apod, const float* rx_apod, const float t0, const float wd, const float i0)
{
	size_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x; // pixel idx
	size_t pixel_stride = blockDim.x * gridDim.x;

	cuFloatComplex buffer = { 0.0f, 0.0f };

	for (size_t i = pixel_idx; i < N_pixels; i += pixel_stride)
	{
		for (size_t j = 0; j < N_waves; j++)
		{
			float tDelay = tx_delay[i + j * N_pixels];
			float tApod = tx_apod[i + j * N_pixels];

			for (size_t g = 0; g < N_channels; g++)
			{
				float delay = tDelay + rx_delay[i + g * N_pixels];
				float apod = tApod * rx_apod[i + g * N_pixels];

				float nelay = delay * Fs - i0;

				cuFloatComplex phase;

				__sincosf(wd * delay, &phase.y, &phase.x);

				// For maximum bandwidth usage adiacent threads must fetch adiacent memory locations in texture --> inputSamplingRate ~= outputSamplingRate
				cuFloatComplex pre_bf_data = tex1DLayered<cuFloatComplex>(tex, nelay, g + j * N_channels);

				buffer.x += (pre_bf_data.x * phase.x - pre_bf_data.y * phase.y) * apod;
				buffer.y += (pre_bf_data.x * phase.y + pre_bf_data.y * phase.x) * apod;
			}
		}

		bf_data[i] = buffer;
	}
}