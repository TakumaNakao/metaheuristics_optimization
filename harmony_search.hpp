#pragma once

#include <array>
#include <vector>
#include <functional>
#include <optional>
#include <random>
#include <tuple>
#include <chrono>

#include <Eigen/Core>

namespace optimization
{

template <size_t D, size_t N>
class HarmonySearch
{
public:
    class Harmony
    {
    private:
        Eigen::Matrix<double, D, 1> x_ = Eigen::VectorXd::Zero(D);
        double cost_ = std::numeric_limits<double>::max();
    public:
        Harmony(const std::function<double(Eigen::Matrix<double, D, 1>)>& func, const Eigen::Matrix<double, D, 1>& init_x) : 
            x_(init_x)
        {
            cost_ = func(x_);
        }
        Eigen::Matrix<double, D, 1> get_x() const
        {
            return x_;
        }
        double get_cost() const
        {
            return cost_;
        }
    };
private:
    std::vector<Harmony> harmonies_;
    std::function<double(Eigen::Matrix<double, D, 1>)> func_;
    std::array<Eigen::Vector2d, D> range_;
    Eigen::Matrix<double, D, 1> best_x_ = Eigen::VectorXd::Zero(D);
    double best_cost_ = std::numeric_limits<double>::max();
    std::mt19937 mt_;
    std::uniform_int_distribution<> harmony_rand_ = std::uniform_int_distribution<>(0, N - 1);
    std::uniform_real_distribution<> probability_rand_ = std::uniform_real_distribution<>(0.0, 1.0);

    void update_best()
    {
        for(const auto& harmony : harmonies_){
            if(best_cost_ > harmony.get_cost()){
                best_x_ = harmony.get_x();
                best_cost_ = harmony.get_cost();
            }
        }
    }
public:
    HarmonySearch(std::function<double(Eigen::Matrix<double, D, 1>)> func, std::array<Eigen::Vector2d, D> range, std::optional<std::array<Eigen::Matrix<double, D, 1>, N>> init_x = std::nullopt) : 
        func_(func),
        range_(range)
    {
        std::random_device rnd;
        mt_ = std::mt19937(rnd());

        harmonies_.reserve(N);
        if(init_x){
            for(size_t i = 0; i < N; i++){
                harmonies_.push_back(Harmony(func_, init_x.value()[i]));
            }
        }
        else{
            for(size_t i = 0; i < N; i++){
                Eigen::Matrix<double, D, 1> x;
                for(size_t j = 0; j < D; j++){
                    x[j] = range_[j][0] + probability_rand_(mt_) * (range_[j][1] - range_[j][0]);
                }
                harmonies_.push_back(Harmony(func_, x));
            }
        }
        update_best();
    }
    void step(double bandwidth, double select_rate, double change_rate)
    {
        Eigen::Matrix<double, D, 1> x;
        for(size_t i = 0; i < D; i++){
            if(probability_rand_(mt_) < select_rate){
                auto select_harmony_x = harmonies_[harmony_rand_(mt_)].get_x();
                if(probability_rand_(mt_) < change_rate){
                    x[i] = std::clamp(select_harmony_x[i] + bandwidth * (probability_rand_(mt_) * 2.0 - 1.0) * (range_[i][1] - range_[i][0]), range_[i][0], range_[i][1]);
                }
                else{
                    x[i] = select_harmony_x[i];
                }
            }
            else{
                
                x[i] = range_[i][0] + probability_rand_(mt_) * (range_[i][1] - range_[i][0]);
            }
        }
        Harmony new_harmony(func_, x);
        std::sort(harmonies_.begin(), harmonies_.end(), [](const Harmony& a, const Harmony& b){ return b.get_cost() > a.get_cost(); });
        if(harmonies_.back().get_cost() > new_harmony.get_cost()){
            harmonies_.back() = new_harmony;
        }
        update_best();
    }
    Eigen::Matrix<double, D, 1> optimization(size_t loop_n, double bandwidth, double select_rate, double change_rate)
    {
        for(size_t i = 0; i < loop_n; i++){
            step(bandwidth, select_rate, change_rate);
        }
        return best_x_;
    }
    Eigen::Matrix<double, D, 1> optimization(std::chrono::nanoseconds loop_time, double bandwidth, double select_rate, double change_rate)
    {
        auto start_time = std::chrono::system_clock::now();
        while(std::chrono::system_clock::now() - start_time < loop_time){
            step(bandwidth, select_rate, change_rate);
        }
        return best_x_;
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>, std::vector<double>> optimization_log(size_t loop_n, double bandwidth, double select_rate, double change_rate)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> x_log;
        std::vector<double> cost_log;
        for(size_t i = 0; i < N; i++){
            x_log[i].push_back(harmonies_[i].get_x());
        }
        cost_log.push_back(best_cost_);
        for(size_t i = 0; i < loop_n; i++){
            step(bandwidth, select_rate, change_rate);
            for(size_t j = 0; j < N; j++){
                x_log[j].push_back(harmonies_[j].get_x());
            }
            cost_log.push_back(best_cost_);
        }

        return {best_x_, x_log, cost_log};
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>, std::vector<double>> optimization_log(std::chrono::nanoseconds loop_time, double bandwidth, double select_rate, double change_rate)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> x_log;
        std::vector<double> cost_log;
        for(size_t i = 0; i < N; i++){
            x_log[i].push_back(harmonies_[i].get_x());
        }
        cost_log.push_back(best_cost_);
        auto start_time = std::chrono::system_clock::now();
        while(std::chrono::system_clock::now() - start_time < loop_time){
            step(bandwidth, select_rate, change_rate);
            for(size_t j = 0; j < N; j++){
                x_log[j].push_back(harmonies_[j].get_x());
            }
            cost_log.push_back(best_cost_);
        }

        return {best_x_, x_log, cost_log};
    }
};

}