#pragma once
#include <memory>
#include "ixrt_common.h"

namespace iluvatar::inferrt {
// TODO: Add namespace for this libaray

class IxRT {
   public:
    IxRT();
    ~IxRT();
    void Init(const RuntimeConfig& config);
    void SaveEngine(const RuntimeConfig& config);
    void LoadEngine(const RuntimeConfig& config);                              // old engine, from file
    void LoadEngine(const RuntimeConfig& config, std::istringstream& engine);  // new engine, from buffer
    void InitEngine(const RuntimeConfig& config);                              // new engine, from file
    void Execute();                                                            // Synchronous run
    void Enqueue(bool start_enqueue = true);                                   // Asynchronous run
    void LoadInput(IOBuffers* inputs);
    void LoadInput();
    void FetchOutput(IOBuffers* outputs);
    void BindIOBuffers(void* io_buffers);
    TensorShapeMap GetInputShape();
    TensorShapeMap GetOutputShape();
    std::vector<std::string> GetInputNames();
    std::vector<std::string> GetOutputNames();
    void SetOutputCallback(const TaskMethod& task, void* outputs);
    void SetPostProcessMethod(const PostProcessMethod& method);
    void TimePerf();
    void SetDeviceID(uint32_t device_id);
    void SetBenchmarkOpMode(AlgoSelectMode algo_select_mode);

   protected:
    class Impl;
    std::shared_ptr<Impl> doer_;
};

}  // end of namespace iluvatar::inferrt
