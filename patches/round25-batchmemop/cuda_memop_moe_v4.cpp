// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>
#include <cuda.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <immintrin.h>
#include <cstdlib>

namespace {
using Forward = int (*)(void*, int, const float*, const int32_t*, const float*,
                        float*);
using Error = const char* (*)();

struct alignas(64) Doorbell {
  volatile uint32_t request;
  volatile uint32_t complete;
  volatile uint32_t timeout;
};

struct Task {
  Doorbell* host_bell = nullptr;
  Doorbell* device_bell = nullptr;
  void* moe;
  Forward forward;
  Error error;
  int tokens, hidden, topk;
  const float* input;
  const int32_t* ids;
  const float* routes;
  float* output;
  std::atomic<int> failed{0};
  char message[512]{};
  int device = 0;
  // Calls, tokens, native nanoseconds, out-of-range histogram IDs, 384 bins.
  std::array<uint64_t, 388> stats{};
  std::array<uint64_t, 4> thread_info{};
  uint64_t cached_routes = 0;
  uint64_t tail_skipped_routes = 0;
  uint64_t tail_skipped_experts = 0;
  uint64_t tail_candidate_routes = 0;
  int tail_skip_rank = 0;
  std::vector<int32_t> filtered_ids;
  std::vector<int> best_rank;

  const int32_t* filter_tail(int actual) {
    if (!tail_skip_rank) return ids;
    const int entries = actual * topk;
    for (int i = 0; i < entries; ++i)
      if (!std::isfinite(routes[i])) return ids;
    std::fill(best_rank.begin(), best_rank.end(), topk);
    for (int t = 0; t < actual; ++t) {
      for (int j = 0; j < topk; ++j) {
        const int i = t * topk + j;
        const int expert = ids[i];
        if (expert < 0 || size_t(expert) >= best_rank.size()) continue;
        int rank = 0;
        for (int q = 0; q < topk; ++q) {
          const float other = routes[t * topk + q];
          rank += other > routes[i] || (other == routes[i] && q < j);
        }
        best_rank[expert] = std::min(best_rank[expert], rank);
      }
    }
    std::copy_n(ids, entries, filtered_ids.begin());
    for (int i = 0; i < entries; ++i) {
      const int expert = ids[i];
      if (expert >= 0 && size_t(expert) < best_rank.size() &&
          best_rank[expert] >= tail_skip_rank)
        filtered_ids[i] = -2;
    }
    return filtered_ids.data();
  }

  void fail(const char* message_) {
    std::snprintf(message, sizeof(message), "%s", message_);
    failed.store(1, std::memory_order_release);
  }
};

// CUDA invokes this outside Python. All called functions are CPU-only: CUDA
// APIs are forbidden in a host callback. Stream ordering makes the preceding
// D2H copies visible here and delays the following H2D copy until completion.
void CUDART_CB run(void* ptr) {
  auto& task = *static_cast<Task*>(ptr);
  std::fill_n(task.output, size_t(task.tokens) * task.hidden, 0.0f);
  try {
    int actual = 0;
    while (actual < task.tokens && task.ids[actual * task.topk] != -1) ++actual;
    for (int t = actual; t < task.tokens; ++t) {
      if (task.ids[t * task.topk] != -1) {
        task.fail("CPU graph expects valid tokens followed by padding");
        break;
      }
    }
    if (!task.failed.load(std::memory_order_acquire) && actual) {
      const auto tid = uint64_t(syscall(SYS_gettid));
      if (!task.thread_info[0]) {
        task.thread_info[0] = tid;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
          task.thread_info[3] = CPU_COUNT(&mask);
        }
      } else if (task.thread_info[1] != tid) {
        ++task.thread_info[2];
      }
      task.thread_info[1] = tid;
      const auto start = std::chrono::steady_clock::now();
      const auto* filtered_ids = task.filter_tail(actual);
      const int status = task.forward(task.moe, actual, task.input,
                                      filtered_ids, task.routes, task.output);
      task.stats[0] += 1;
      task.stats[1] += actual;
      task.stats[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
      if (status) {
        task.fail(task.error());
      } else {
        if (filtered_ids != task.ids) {
          task.tail_skipped_experts += std::count_if(
              task.best_rank.begin(), task.best_rank.end(), [&](int rank) {
                return rank >= task.tail_skip_rank && rank < task.topk;
              });
        }
        for (int i = 0; i < actual * task.topk; ++i) {
          const int expert = task.ids[i];
          if (task.tail_skip_rank && expert >= 0) ++task.tail_candidate_routes;
          if (expert >= 0 && filtered_ids[i] == -2)
            ++task.tail_skipped_routes;
          else if (expert >= 0 && expert < 384)
            ++task.stats[4 + expert];
          else if (expert == -2)
            ++task.cached_routes;
          else
            ++task.stats[3];
        }
      }
    }
  } catch (const std::exception& error) {
    task.fail(error.what());
  } catch (...) {
    task.fail("Unknown failure in CPU MoE CUDA host callback");
  }
  if (task.failed.load(std::memory_order_acquire)) {
    std::fill_n(task.output, size_t(task.tokens) * task.hidden,
                std::numeric_limits<float>::quiet_NaN());
  }
}
// One CPU owner for all captured shapes/layers in this process. It executes
// the same forward routine and statistics as the original CUDA host callback.
// The owner never calls CUDA APIs; stream-ordered doorbells replace callbacks.
struct CPUOwner {
  std::mutex mutex;
  std::vector<Task*> tasks;
  static constexpr size_t slots = 4096;
  Doorbell* host_pool = nullptr;
  Doorbell* device_pool = nullptr;
  std::array<bool, slots> used{};
  std::atomic<bool> stopping{false};
  std::thread thread;
  CPUOwner() {
    if (cudaHostAlloc(reinterpret_cast<void**>(&host_pool), slots * sizeof(Doorbell),
                      cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) {
      if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_pool),
                                  host_pool, 0) != cudaSuccess) device_pool = nullptr;
    }
  }
  bool allocate(Task* task) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!host_pool || !device_pool) return false;
    for (size_t i = 0; i < slots; ++i) {
      if (!used[i]) {
        used[i] = true;
        task->host_bell = host_pool + i;
        task->device_bell = device_pool + i;
        task->host_bell->request = task->host_bell->complete = task->host_bell->timeout = 0;
        return true;
      }
    }
    return false;
  }
  ~CPUOwner() {
    stopping.store(true, std::memory_order_release);
    if (thread.joinable()) thread.join();
  }
  void loop() {
    while (!stopping.load(std::memory_order_acquire)) {
      bool worked = false;
      {
        std::lock_guard<std::mutex> lock(mutex);
        for (Task* task : tasks) {
          if (__atomic_load_n(&task->host_bell->request, __ATOMIC_ACQUIRE)) {
            run(task);
            __atomic_store_n(&task->host_bell->request, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&task->host_bell->complete, 1, __ATOMIC_RELEASE);
            worked = true;
          }
        }
      }
      if (!worked) _mm_pause();
    }
  }
  void add(Task* task) {
    std::lock_guard<std::mutex> lock(mutex);
    tasks.push_back(task);
    if (!thread.joinable()) thread = std::thread([this] { loop(); });
  }
  void remove(Task* task) {
    std::lock_guard<std::mutex> lock(mutex);
    tasks.erase(std::remove(tasks.begin(), tasks.end(), task), tasks.end());
    used[task->host_bell - host_pool] = false;
  }
};
CPUOwner& owner() { static CPUOwner instance; return instance; }
// CDLL loads during model construction, before graph capture. The entire
// pinned arena is created here; captured task creation only takes a free slot.
__attribute__((constructor)) void prepare_owner_arena() { (void)owner(); }

}  // namespace

extern "C" {
void* dsv41_cuda_task_create(void* moe, void* forward, void* error, int tokens,
                             int hidden, int topk, const float* input,
                             const int32_t* ids, const float* routes,
                             float* output) {
  auto task = std::make_unique<Task>();
  task->moe = moe;
  task->forward = reinterpret_cast<Forward>(forward);
  task->error = reinterpret_cast<Error>(error);
  task->tokens = tokens;
  task->hidden = hidden;
  task->topk = topk;
  task->input = input;
  task->ids = ids;
  task->routes = routes;
  task->output = output;
  if (cudaGetDevice(&task->device) != cudaSuccess) return nullptr;
  if (!owner().allocate(task.get())) return nullptr;
  owner().add(task.get());
  return task.release();
}

// The owner must complete its stream before changing a captured task's policy.
int dsv41_cuda_task_set_tail_skip(void* handle, int rank, int num_experts) {
  auto& task = *static_cast<Task*>(handle);
  if (rank < 0 || rank >= task.topk || num_experts < 1) return 1;
  task.tail_skip_rank = rank;
  if (rank) {
    task.filtered_ids.resize(size_t(task.tokens) * task.topk);
    task.best_rank.resize(num_experts);
  }
  return 0;
}

int dsv41_cuda_task_enqueue(void* handle, void* stream) {
  auto& task = *static_cast<Task*>(handle);
  auto driver_stream = reinterpret_cast<CUstream>(stream);
  CUstreamBatchMemOpParams operations[3]{};
  operations[0].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
  operations[0].writeValue.address = reinterpret_cast<CUdeviceptr>(&task.device_bell->complete);
  operations[0].writeValue.value = 0;
  operations[1].operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
  operations[1].writeValue.address = reinterpret_cast<CUdeviceptr>(&task.device_bell->request);
  operations[1].writeValue.value = 1;
  operations[2].operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
  operations[2].waitValue.address = reinterpret_cast<CUdeviceptr>(&task.device_bell->complete);
  operations[2].waitValue.value = 1;
  operations[2].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
  CUstreamCaptureStatus capture_status;
  auto status = cuStreamIsCapturing(driver_stream, &capture_status);
  if (status != CUDA_SUCCESS) return int(status);
  if (capture_status == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
    CUgraph graph;
    const CUgraphNode* dependencies;
    size_t count;
    status = cuStreamGetCaptureInfo(driver_stream, &capture_status, nullptr,
                                      &graph, &dependencies, nullptr, &count);
    if (status != CUDA_SUCCESS) return int(status);
    CUcontext context;
    status = cuCtxGetCurrent(&context);
    if (status != CUDA_SUCCESS) return int(status);
    CUDA_BATCH_MEM_OP_NODE_PARAMS params{};
    params.ctx = context;
    params.count = 3;
    params.paramArray = operations;
    CUgraphNode node;
    status = cuGraphAddBatchMemOpNode(&node, graph, dependencies, count, &params);
    if (status != CUDA_SUCCESS) return int(status);
    return int(cuStreamUpdateCaptureDependencies(driver_stream, &node, nullptr, 1,
                                                CU_STREAM_SET_CAPTURE_DEPENDENCIES));
  }
  return int(cuStreamBatchMemOp(driver_stream, 3, operations, 0));
}

const char* dsv41_cuda_task_error(void* handle) {
  auto& task = *static_cast<Task*>(handle);
  if (task.host_bell && task.host_bell->timeout) {
    task.fail("CPU owner handoff timed out");
  }
  return task.failed.load(std::memory_order_acquire) ? task.message : nullptr;
}

// The owner must finish its CUDA work before reading these serial-use counters.
void dsv41_cuda_task_stats(void* handle, uint64_t* output) {
  auto& task = *static_cast<Task*>(handle);
  std::copy(task.stats.begin(), task.stats.end(), output);
}

uint64_t dsv41_cuda_task_cached_routes(void* handle) {
  return static_cast<Task*>(handle)->cached_routes;
}

uint64_t dsv41_cuda_task_tail_skipped_routes(void* handle) {
  return static_cast<Task*>(handle)->tail_skipped_routes;
}

// Like task_stats, read only after the owning stream has completed its work.
void dsv41_cuda_tasks_route_counts(void* const* handles, int count,
                                   uint64_t* output) {
  output[0] = output[1] = 0;
  for (int i = 0; i < count; ++i) {
    const auto& task = *static_cast<Task*>(handles[i]);
    output[0] += task.cached_routes;
    for (size_t expert = 4; expert < task.stats.size(); ++expert) {
      output[1] += task.stats[expert];
    }
  }
}

// GPU routes, CPU routes, skipped routes, skipped expert computations, eligible
// CPU decode routes. The owner must finish its CUDA work before reading.
void dsv41_cuda_tasks_route_stats(void* const* handles, int count,
                                  uint64_t* output) {
  dsv41_cuda_tasks_route_counts(handles, count, output);
  output[2] = output[3] = output[4] = 0;
  for (int i = 0; i < count; ++i) {
    const auto& task = *static_cast<Task*>(handles[i]);
    output[2] += task.tail_skipped_routes;
    output[3] += task.tail_skipped_experts;
    output[4] += task.tail_candidate_routes;
  }
}

void dsv41_cuda_task_thread_info(void* handle, uint64_t* output) {
  auto& task = *static_cast<Task*>(handle);
  std::copy(task.thread_info.begin(), task.thread_info.end(), output);
}

void dsv41_cuda_task_destroy(void* handle) {
  auto* task = static_cast<Task*>(handle);
  if (!task) return;
  int previous = task->device;
  cudaGetDevice(&previous);
  cudaSetDevice(task->device);
  // A captured host node can replay on a different stream than capture used.
  cudaDeviceSynchronize();
  owner().remove(task);
  cudaSetDevice(previous);
  delete task;
}
}
