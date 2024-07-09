
#pragma once

#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "cuda_runtime.h"

#include <fmt/format.h>

struct TimerEntry
{
    int id;
    std::string name;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float time_ms_accumulated;
    int n_executions;
    bool executed;

    void reset()
    {
        time_ms_accumulated = 0.0f;
        n_executions = 0;
        executed = false;
    }
    void start()
    {
        cudaEventRecord(start_event);
    }
    void stop()
    {
        cudaEventRecord(stop_event);
        executed = true;
    }
    float time()
    {
        float t = 0.0f;
        if (executed)
        {
            cudaEventElapsedTime(&t, start_event, stop_event);
            n_executions++;
            executed = false;
        }
        time_ms_accumulated += t;
        return t;
    }
};

template <bool IsActive>
struct CudaTimer
{
    CudaTimer(std::map<int, std::string> init_entries)
    {
        for (auto const& it : init_entries)
        {
            TimerEntry entry;
            entry.id = it.first;
            entry.name = it.second;

            if (IsActive)
            {
                cudaEventCreate(&(entry.start_event));
                cudaEventCreate(&(entry.stop_event));
            }            
            entry.reset();

            _ids.push_back(it.first);
            _timing_entries[it.first] = entry;
        }
        _running_id = -1;
    }

    ~CudaTimer()
    {
        if (!IsActive) return;
        for (int id : _ids)
        {
            TimerEntry entry = _timing_entries[id];

            cudaEventDestroy(entry.start_event);
            cudaEventDestroy(entry.stop_event);
        }
    }


    void start(int id)
    {
        if (!IsActive) return;
        if (_running_id >= 0) 
            stop(_running_id);

        _timing_entries[id].start();
        _running_id = id;
    }
    void stop(int id)
    {
        if (!IsActive) return;
        _timing_entries[id].stop();
        _running_id = -1;
    }

    void reset()
    {
        if (!IsActive) return;
        _running_id = -1;
        for (int id : _ids)
            _timing_entries[id].reset();
    }

    void syncElapsed()
    {
        if (!IsActive) return;
        if (_running_id >= 0) 
            stop(_running_id);

        cudaDeviceSynchronize();

        for (int id : _ids)
            _timing_entries[id].time();
    }

    void print(int n_iterations = 1)
    {
        std::cout << "Timings in ms:" << std::endl;
        for (int id : _ids)
        {
            TimerEntry entry = _timing_entries[id];
            if (entry.n_executions > 0)
            {
                fmt::print("- {0:30}: {1:7.03f}ms, N: {2:3}\n", entry.name, entry.time_ms_accumulated / n_iterations, entry.n_executions / n_iterations);
            }
        }
    }

    std::vector<TimerEntry> getTimings(int n_iterations = 1)
    {
        std::vector<TimerEntry> values(_timing_entries.size());
        std::transform(_timing_entries.begin(), _timing_entries.end(), values.begin(), [](auto pair){return pair.second;});

        for (auto& entry : values)
        {
            entry.time_ms_accumulated /= n_iterations; 
            entry.n_executions /= n_iterations;
        }
        return values;
    }

private:
    std::unordered_map<int, TimerEntry> _timing_entries;
    std::vector<int> _ids;
    int _running_id = -1;
};

static void to_json(nlohmann::json& j, const TimerEntry& t) 
{
	j = nlohmann::json{
		{"name", t.name},
        {"time_ms", t.time_ms_accumulated},
        {"n_executions", t.n_executions}
	};
}