import os
import tempfile
from contextlib import nullcontext

import torch
import torch.utils.cpp_extension
from packaging import version
from torch.cuda.memory import CUDAPluggableAllocator

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.server_args import get_global_server_args

after_2_8_0 = version.parse(torch.__version__) >= version.parse("2.8.0")

nccl_allocator_source = """

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

// copy from https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclRemoteError             =  6,
               ncclInProgress              =  7,
               ncclNumResults              =  8 } ncclResult_t;
typedef struct ncclComm* ncclComm_t;
typedef struct ncclWindow_vidmem* ncclWindow_t;
ncclResult_t  ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size, ncclWindow_t* win, int winFlags);
#define NCCL_WIN_COLL_SYMMETRIC 0x01

ncclResult_t  ncclMemAlloc(void** ptr, size_t size);
ncclResult_t  ncclMemFree(void *ptr);

static const char* ncclResultString(ncclResult_t result) {
  switch (result) {
    case ncclSuccess: return "ncclSuccess";
    case ncclUnhandledCudaError: return "ncclUnhandledCudaError";
    case ncclSystemError: return "ncclSystemError";
    case ncclInternalError: return "ncclInternalError";
    case ncclInvalidArgument: return "ncclInvalidArgument";
    case ncclInvalidUsage: return "ncclInvalidUsage";
    case ncclRemoteError: return "ncclRemoteError";
    case ncclInProgress: return "ncclInProgress";
    default: return "ncclUnknownError";
  }
}

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  void* ptr = NULL;
  // DEBUG: always print for large allocations during debugging
  int debug = 1;

  if (debug && size > 1024*1024*100) {  // >100MB
    fprintf(stderr, "[SYMM ALLOC] ncclMemAlloc request: size=%.2fGB\\n",
            (double)size / (1024*1024*1024));
    fflush(stderr);
  }

  ncclResult_t err = ncclMemAlloc(&ptr, size);
  if (err != ncclSuccess) {
    fprintf(stderr, "[SYMM ALLOC ERROR] ncclMemAlloc FAILED: size=%.2fGB, error=%s (%d)\\n",
            (double)size / (1024*1024*1024), ncclResultString(err), err);
    fflush(stderr);
    return NULL;  // Return NULL on failure
  }

  if (debug && size > 1024*1024*100) {
    fprintf(stderr, "[SYMM ALLOC] ncclMemAlloc success: ptr=%p, size=%.2fGB\\n",
            ptr, (double)size / (1024*1024*1024));
    fflush(stderr);
  }

  const char *str_val = getenv("SGLANG_TMP_NCCL_COMM_VALUE");
  if (!str_val) {
    fprintf(stderr, "[SYMM ALLOC ERROR] SGLANG_TMP_NCCL_COMM_VALUE not set!\\n");
    fflush(stderr);
    ncclMemFree(ptr);
    return NULL;
  }

  char *endptr;
  void* int_val = (void *)strtoull(str_val, &endptr, 0);
  ncclComm_t comm = (ncclComm_t)(int_val);
  ncclWindow_t win;

  ncclResult_t err2 = ncclCommWindowRegister(comm, ptr, size, &win, NCCL_WIN_COLL_SYMMETRIC);
  if (err2 != ncclSuccess) {
    fprintf(stderr, "[SYMM ALLOC ERROR] ncclCommWindowRegister FAILED: size=%.2fGB, error=%s (%d)\\n",
            (double)size / (1024*1024*1024), ncclResultString(err2), err2);
    fflush(stderr);
    ncclMemFree(ptr);
    return NULL;
  }

  if (debug && size > 1024*1024*100) {
    fprintf(stderr, "[SYMM ALLOC] ncclCommWindowRegister success\\n");
    fflush(stderr);
  }

  return ptr;
}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
  ncclResult_t err = ncclMemFree(ptr);
  if (err != ncclSuccess) {
    // DEBUG: always print errors during debugging
    fprintf(stderr, "[SYMM ALLOC ERROR] ncclMemFree FAILED: error=%s (%d)\\n",
            ncclResultString(err), err);
    fflush(stderr);
  }
}

}
"""

_allocator = None
_mem_pool = None
_graph_pool_id = None
_cur_device = None
_active_symmetric_memory_context = None


def is_symmetric_memory_enabled():
    return get_global_server_args().enable_symm_mem


def set_graph_pool_id(graph_pool_id):
    global _graph_pool_id
    _graph_pool_id = graph_pool_id


def disable_symmetric_memory_context():
    if _active_symmetric_memory_context is None:
        return None
    saved_context = _active_symmetric_memory_context
    saved_context.__exit__(None, None, None)
    return saved_context


def restore_symmetric_memory_context(saved_context):
    if saved_context is not None:
        saved_context.__enter__()


def get_nccl_mem_pool():
    global _allocator, _mem_pool, _cur_device
    if _mem_pool is None:
        out_dir = tempfile.gettempdir()
        nccl_allocator_libname = "nccl_allocator"
        torch.utils.cpp_extension.load_inline(
            name=nccl_allocator_libname,
            cpp_sources=nccl_allocator_source,
            with_cuda=True,
            extra_ldflags=["-lnccl"],
            verbose=True,
            is_python_module=False,
            build_directory=out_dir,
        )
        _allocator = CUDAPluggableAllocator(
            f"{out_dir}/{nccl_allocator_libname}.so",
            "nccl_alloc_plug",
            "nccl_free_plug",
        ).allocator()
        _mem_pool = torch.cuda.MemPool(_allocator)
        _cur_device = torch.cuda.current_device()
    return _mem_pool


class SymmetricMemoryContext:
    """
    Context manager for using symmetric memory with pynccl.

    To Utilize the symmetric memory feature in NCCL, the buffers need to be allocated
    by `ncclMemAlloc` and registered by `ncclCommWindowRegister`. Due to this, we introduce
    this context manager. All tensors created under this context will be correctly
    allocated and registered with a custom allocator.
    """

    def __init__(
        self,
        group_coordinator: GroupCoordinator,
    ):
        self.group_coordinator = group_coordinator
        self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
        self.is_graph_capture = torch.cuda.is_current_stream_capturing()
        self.exited = False

    def __enter__(self):
        assert (
            self.group_coordinator.pynccl_comm is not None
        ), f"Symmetric memory requires pynccl to be enabled in group '{self.group_coordinator.group_name}'"

        # Proactively release PyTorch cached memory if CUDA free memory is low
        # ncclMemAlloc bypasses PyTorch's allocator, so it needs raw CUDA memory
        # Threshold: 2GB (largest embedding allocation is ~1.4GB for 100K tokens)
        if not self.is_graph_capture:
            free_memory, total_memory = torch.cuda.mem_get_info()
            if free_memory < 2 * 1024 * 1024 * 1024:  # < 2GB free
                torch.cuda.empty_cache()

        if self.is_graph_capture:
            assert (
                _graph_pool_id is not None
            ), "graph_pool_id is not set under graph capture"
            # Pause graph memory pool to use symmetric memory with cuda graph
            if after_2_8_0:
                torch._C._cuda_endAllocateToPool(_cur_device, _graph_pool_id)
            else:
                torch._C._cuda_endAllocateCurrentStreamToPool(
                    _cur_device, _graph_pool_id
                )

        if self.exited:
            # mempool ctx (@contextlib.contextmanager) is not re-entrant
            self._mem_pool_ctx = torch.cuda.use_mem_pool(get_nccl_mem_pool())
            self.exited = False
        self._mem_pool_ctx.__enter__()

        # Set the env var to pass this argument to the C functions.
        os.environ["SGLANG_TMP_NCCL_COMM_VALUE"] = str(
            self.group_coordinator.pynccl_comm.comm.value
        )

        global _active_symmetric_memory_context
        _active_symmetric_memory_context = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mem_pool_ctx.__exit__(exc_type, exc_val, exc_tb)

        if self.is_graph_capture:
            if after_2_8_0:
                torch._C._cuda_beginAllocateCurrentThreadToPool(
                    _cur_device, _graph_pool_id
                )
            else:
                torch._C._cuda_beginAllocateToPool(_cur_device, _graph_pool_id)

        global _active_symmetric_memory_context
        _active_symmetric_memory_context = None

        self.exited = True


def use_symmetric_memory(group_coordinator: GroupCoordinator, disabled: bool = False):
    disabled = (
        not is_symmetric_memory_enabled()
        or disabled
        or group_coordinator.world_size == 1
    )
    return SymmetricMemoryContext(group_coordinator) if not disabled else nullcontext()
