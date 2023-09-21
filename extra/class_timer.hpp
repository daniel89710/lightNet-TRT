#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <numeric>

class Timer
{
public:
  Timer() : beg_(clock_::now())
  {
  }
  void reset()
  {
    beg_ = clock_::now();
  }

  double elapsed() const
  {
    return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
  }

  void out(std::string message = "")
  {
    double t = elapsed();
    durations.push_back(t);
    std::cout << message << " elasped time:" << t << "ms" << std::endl;
    std::cout << "avg time:" << std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size() << "ms" << std::endl;
    reset();
  }

  double get_duration() const
  {
    return elapsed();
  }

private:
  using clock_ = std::chrono::high_resolution_clock;
  using second_ = std::chrono::duration<double, std::milli>;
  std::chrono::time_point<clock_> beg_;
  static std::vector<double> durations;
};
