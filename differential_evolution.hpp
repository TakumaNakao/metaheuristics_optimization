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
class DifferentialEvolution
{
public:
    class Individual
    {
    private:
        inline static std::random_device rnd_;
        inline static std::mt19937 mt_ = std::mt19937(rnd_());
        const std::function<double(Eigen::Matrix<double, D, 1>)>& func_;
        Eigen::Matrix<double, D, 1> x_ = Eigen::VectorXd::Zero(D);
        double cost_ = std::numeric_limits<double>::max();
        std::array<Eigen::Vector2d, D> range_;
        std::uniform_real_distribution<> rand_ = std::uniform_real_distribution<>(0.0, 1.0);
    public:
        Individual(const std::function<double(Eigen::Matrix<double, D, 1>)>& func, std::array<Eigen::Vector2d, D> range, std::optional<Eigen::Matrix<double, D, 1>> init_x = std::nullopt) : 
            func_(func),
            range_(range)
        {
            if(init_x){
                set_x(init_x.value());
            }
            else{
                Eigen::Matrix<double, D, 1> x;
                for(size_t i = 0; i < D; i++){
                    std::uniform_real_distribution<> rand(range_[i][0], range_[i][1]);
                    x[i] = rand(mt_);
                }
                set_x(x);
            }
        }
        void update(Eigen::Matrix<double, D, 1> x)
        {
            for(size_t i = 0; i < D; i++){
                x[i] = std::clamp(x[i], range_[i][0], range_[i][1]);
            }
            double cost = func_(x);
            if(cost_ > cost){
                set_x(x, cost);
            }
        }
        void set_x(const Eigen::Matrix<double, D, 1>& x, std::optional<double> cost = std::nullopt)
        {
            for(size_t i = 0; i < D; i++){
                x_[i] = std::clamp(x[i], range_[i][0], range_[i][1]);
            }
            if(cost){
                cost_ = cost.value();
            }
            else{
                cost_ = func_(x_);
            }
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
    std::vector<Individual> individuals_;
    std::function<double(Eigen::Matrix<double, D, 1>)> func_;
    std::array<Eigen::Vector2d, D> range_;
    Eigen::Matrix<double, D, 1> best_x_ = Eigen::VectorXd::Zero(D);
    double best_cost_ = std::numeric_limits<double>::max();
    std::mt19937 mt_;
    std::uniform_int_distribution<> dim_rand_ = std::uniform_int_distribution<>(0, D - 1);
    std::uniform_int_distribution<> individual_rand_ = std::uniform_int_distribution<>(0, N - 1);
    std::uniform_real_distribution<> probability_rand_ = std::uniform_real_distribution<>(0.0, 1.0);

    int gen_rand_individual_id(int exclusion)
    {
        while(1)
        {
            int id = individual_rand_(mt_);
            if(id != exclusion){
                return id;
            }
        }
    }
    void update_best()
    {
        for(const auto& individual : individuals_){
            if(best_cost_ > individual.get_cost()){
                best_x_ = individual.get_x();
                best_cost_ = individual.get_cost();
            }
        }
    }
public:
    DifferentialEvolution(std::function<double(Eigen::Matrix<double, D, 1>)> func, std::array<Eigen::Vector2d, D> range, std::optional<std::array<Eigen::Matrix<double, D, 1>, N>> init_x = std::nullopt) : 
        func_(func),
        range_(range)
    {
        std::random_device rnd;
        mt_ = std::mt19937(rnd());

        individuals_.reserve(N);
        if(init_x){
            for(size_t i = 0; i < N; i++){
                individuals_.push_back(Individual(func_, range_, init_x.value()[i]));
            }
        }
        else{
            for(size_t i = 0; i < N; i++){
                Eigen::Matrix<double, D, 1> x;
                for(size_t j = 0; j < D; j++){
                    x[j] = range_[j][0] + probability_rand_(mt_) * (range_[j][1] - range_[j][0]);
                }
                individuals_.push_back(Individual(func_, range_, x));
            }
        }
        update_best();
    }
    void step(double crossover_rate, double scaling)
    {
        for(size_t i = 0; i < N; i++){
            Eigen::Matrix<double, D, 1> x1 = individuals_[gen_rand_individual_id(i)].get_x();
            Eigen::Matrix<double, D, 1> x2 = individuals_[gen_rand_individual_id(i)].get_x();
            Eigen::Matrix<double, D, 1> x3 = individuals_[gen_rand_individual_id(i)].get_x();
            Eigen::Matrix<double, D, 1> v = x1 + scaling * (x2 - x3);
            Eigen::Matrix<double, D, 1> x = individuals_[i].get_x();
            int change_dim = dim_rand_(mt_);
            for(size_t j = 0; j < D; j++){
                if(j == change_dim || probability_rand_(mt_) < crossover_rate){
                    x[j] = v[j];
                }
            }
            individuals_[i].update(x);
        }
        update_best();
    }
    Eigen::Matrix<double, D, 1> optimization(size_t loop_n, double crossover_rate, double scaling)
    {
        for(size_t i = 0; i < loop_n; i++){
            step(crossover_rate, scaling);
        }
        return best_x_;
    }
    Eigen::Matrix<double, D, 1> optimization(std::chrono::nanoseconds loop_time, double crossover_rate, double scaling)
    {
        auto start_time = std::chrono::system_clock::now();
        while(std::chrono::system_clock::now() - start_time < loop_time){
            step(crossover_rate, scaling);
        }
        return best_x_;
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>, std::vector<double>> optimization_log(size_t loop_n, double crossover_rate, double scaling)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> x_log;
        std::vector<double> cost_log;
        for(size_t i = 0; i < N; i++){
            x_log[i].push_back(individuals_[i].get_x());
        }
        cost_log.push_back(best_cost_);
        for(size_t i = 0; i < loop_n; i++){
            step(crossover_rate, scaling);
            for(size_t j = 0; j < N; j++){
                x_log[j].push_back(individuals_[j].get_x());
            }
            cost_log.push_back(best_cost_);
        }

        return {best_x_, x_log, cost_log};
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>, std::vector<double>> optimization_log(std::chrono::nanoseconds loop_time, double crossover_rate, double scaling)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> x_log;
        std::vector<double> cost_log;
        for(size_t i = 0; i < N; i++){
            x_log[i].push_back(individuals_[i].get_x());
        }
        cost_log.push_back(best_cost_);
        auto start_time = std::chrono::system_clock::now();
        while(std::chrono::system_clock::now() - start_time < loop_time){
            step(crossover_rate, scaling);
            for(size_t j = 0; j < N; j++){
                x_log[j].push_back(individuals_[j].get_x());
            }
            cost_log.push_back(best_cost_);
        }

        return {best_x_, x_log, cost_log};
    }
};

}