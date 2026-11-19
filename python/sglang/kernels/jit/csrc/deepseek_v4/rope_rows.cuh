#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

// In-place RoPE on the last 64 dims of every head of q, one CTA per row: lane l owns rope pair l,
// warp w owns heads w, w + 8, ... (deepseek_rope_kernel in rope.cuh runs one warp per (row, head)).
constexpr int64_t kRopeRowsDim = 64;
constexpr uint32_t kRopeRowsWarps = 8;
constexpr uint32_t kRopeRowsBlockSize = kRopeRowsWarps * device::kWarpThreads;
// Heads a lane loads before its first store; 8 x 8 warps covers 64 heads in one pass.
constexpr uint32_t kRopeRowsHeadsInFlight = 8;

struct RopeRowsParams {
  void* __restrict__ q;
  const float* __restrict__ freqs_cis;
  const void* __restrict__ positions;
  int64_t q_stride_batch;  // bytes
  int64_t q_stride_head;   // bytes
  uint32_t num_heads;
};

template <bool kUsePDL, bool kInverse, typename IndexType>
__global__
__launch_bounds__(kRopeRowsBlockSize) void deepseek_rope_rows_kernel(const __grid_constant__ RopeRowsParams param) {
  using namespace device;
  using DType2 = packed_t<bf16_t>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto batch_id = blockIdx.x;

  const auto position = static_cast<const IndexType*>(param.positions)[batch_id];
  const auto freq_ptr = reinterpret_cast<const fp32x2_t*>(param.freqs_cis + position * kRopeRowsDim);
  const auto [f_real, f_imag] = freq_ptr[lane_id];
  const auto row = pointer::offset(param.q, batch_id * param.q_stride_batch);
  PDLWaitPrimary<kUsePDL>();

  for (uint32_t base = warp_id; base < param.num_heads; base += kRopeRowsWarps * kRopeRowsHeadsInFlight) {
    DType2 data[kRopeRowsHeadsInFlight];
#pragma unroll
    for (uint32_t j = 0; j < kRopeRowsHeadsInFlight; ++j) {
      const auto head = base + j * kRopeRowsWarps;
      if (head < param.num_heads) {
        data[j] = static_cast<DType2*>(pointer::offset(row, head * param.q_stride_head))[lane_id];
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < kRopeRowsHeadsInFlight; ++j) {
      const auto head = base + j * kRopeRowsWarps;
      if (head < param.num_heads) {
        // The same expressions as deepseek_rope_kernel, so both kernels round alike.
        const auto [x_real, x_imag] = cast<fp32x2_t>(data[j]);
        fp32x2_t output;
        if constexpr (kInverse) {
          output = {
              x_real * f_real + x_imag * f_imag,
              x_imag * f_real - x_real * f_imag,
          };
        } else {
          output = {
              x_real * f_real - x_imag * f_imag,
              x_real * f_imag + x_imag * f_real,
          };
        }
        static_cast<DType2*>(pointer::offset(row, head * param.q_stride_head))[lane_id] = cast<DType2>(output);
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct RopeRowsKernel {
  static constexpr auto kernel_fwd_i32 = deepseek_rope_rows_kernel<kUsePDL, false, int32_t>;
  static constexpr auto kernel_fwd_i64 = deepseek_rope_rows_kernel<kUsePDL, false, int64_t>;
  static constexpr auto kernel_inv_i32 = deepseek_rope_rows_kernel<kUsePDL, true, int32_t>;
  static constexpr auto kernel_inv_i64 = deepseek_rope_rows_kernel<kUsePDL, true, int64_t>;

  static void forward(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      bool inverse) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto H = SymbolicSize{"num_heads"};
    constexpr auto D = kRopeRowsDim;
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();

    TensorMatcher({B, H, D})  //
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(q);
    TensorMatcher({-1, D})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);
    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);
    const bool pos_i32 = pos_dtype.is_type<int32_t>();

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto elem_size = static_cast<int64_t>(sizeof(bf16_t));
    const auto params = RopeRowsParams{
        .q = q.data_ptr(),
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .q_stride_batch = q.stride(0) * elem_size,
        .q_stride_head = q.stride(1) * elem_size,
        .num_heads = static_cast<uint32_t>(H.unwrap()),
    };

    using KernelType = decltype(kernel_fwd_i32);
    const KernelType kernel =
        inverse ? (pos_i32 ? kernel_inv_i32 : kernel_inv_i64) : (pos_i32 ? kernel_fwd_i32 : kernel_fwd_i64);
    LaunchKernel(batch_size, kRopeRowsBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
