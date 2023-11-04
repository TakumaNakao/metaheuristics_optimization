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
class GeneticAlgorithm
{
public:
    class Individual
    {
    private:
        Eigen::Matrix<double, D, 1> x_ = Eigen::VectorXd::Zero(D);
        double fitness_ = 0;

        double calc_fitness(Eigen::Matrix<double, D, 1> x, const std::function<double(Eigen::Matrix<double, D, 1>)>& func)
        {
            double cost = func(x);
            if(cost >= 0){
                return 1.0 / (1.0 + cost);
            }
            else{
                return 1.0 + std::fabs(cost);
            }
        }
    public:
        Individual(const std::function<double(Eigen::Matrix<double, D, 1>)>& func, const Eigen::Matrix<double, D, 1>& init_x) : 
            x_(init_x)
        {
            fitness_ = calc_fitness(x_, func);
        }
        Eigen::Matrix<double, D, 1> get_x() const
        {
            return x_;
        }
        double get_fitness() const
        {
            return fitness_;
        }
    };
private:
    std::vector<Individual> parent_individuals_;
    std::vector<Individual> childe_individuals_;
    std::function<double(Eigen::Matrix<double, D, 1>)> func_;
    std::array<Eigen::Vector2d, D> range_;
    Eigen::Matrix<double, D, 1> best_x_ = Eigen::VectorXd::Zero(D);
    double best_fitness_ = 0;
    std::mt19937 mt_;
    std::uniform_real_distribution<> rand_ = std::uniform_real_distribution<>(0.0, 1.0);

    Individual roulette()
    {
        double fitness_sum = 0;
        for(const auto& indivisual : parent_individuals_){
            fitness_sum += indivisual.get_fitness();
        }
        double r = rand_(mt_) * fitness_sum;
        for(const auto& individual : parent_individuals_){
            if(r > individual.get_fitness()){
                return individual;
            }
        }
        return parent_individuals_.back();
    }
    Individual random_cross(Individual parent_a, Individual parent_b, double mutation_ratio)
    {
        Eigen::Matrix<double, D, 1> parent_a_x = parent_a.get_x();
        Eigen::Matrix<double, D, 1> parent_b_x = parent_b.get_x();
        Eigen::Matrix<double, D, 1> childe_x;
        for(size_t i = 0; i < D; i++){
            if(rand_(mt_) < mutation_ratio){
                childe_x[i] = range_[i][0] + rand_(mt_) * (range_[i][1] - range_[i][0]);
            }
            else if(rand_(mt_) > 0.5){
                childe_x[i] = parent_a_x[i];
            }
            else{
                childe_x[i] = parent_b_x[i];
            }
        }
        return Individual(func_, childe_x);
    }
    void update_best()
    {
        for(const auto& indivisual : parent_individuals_){
            if(indivisual.get_fitness() > best_fitness_){
                best_x_ = indivisual.get_x();
                best_fitness_ = indivisual.get_fitness();
            }
        }
    }
    double fitness_to_cost(double fitness)
    {
        if(fitness > 1.0){
            return 1.0 - fitness;
        }
        else{
            return 1.0 / fitness - 1.0;
        }
    }
public:
    GeneticAlgorithm(std::function<double(Eigen::Matrix<double, D, 1>)> func, std::array<Eigen::Vector2d, D> range, std::optional<std::array<Eigen::Matrix<double, D, 1>, N>> init_x = std::nullopt) : 
        func_(func),
        range_(range)
    {
        std::random_device rnd;
        mt_ = std::mt19937(rnd());

        parent_individuals_.reserve(N);
        if(init_x){
            for(size_t i = 0; i < N; i++){
                parent_individuals_.push_back(Individual(func_, init_x.value()[i]));
            }
        }
        else{
            for(size_t i = 0; i < N; i++){
                Eigen::Matrix<double, D, 1> x;
                for(size_t j = 0; j < D; j++){
                    x[j] = range_[j][0] + rand_(mt_) * (range_[j][1] - range_[j][0]);
                }
                parent_individuals_.push_back(Individual(func_, x));
            }
        }
        update_best();
    }
    void step(double mutation_ratio)
    {
        childe_individuals_.clear();
        childe_individuals_.reserve(N);
        std::sort(parent_individuals_.begin(), parent_individuals_.end(), [](const Individual& a, const Individual& b){ return a.get_fitness() > b.get_fitness(); });
        for(size_t i = 0; i < N; i++){
            childe_individuals_.push_back(random_cross(roulette(), roulette(), mutation_ratio));
        }
        parent_individuals_ = childe_individuals_;
        update_best();
    }
    Eigen::Matrix<double, D, 1> optimization(size_t loop_n, double mutation_ratio)
    {
        for(size_t i = 0; i < loop_n; i++){
            step(mutation_ratio);
        }
        return best_x_;
    }
    Eigen::Matrix<double, D, 1> optimization(std::chrono::nanoseconds loop_time, double mutation_ratio)
    {
        auto start_time = std::chrono::system_clock::now();
        while(std::chrono::system_clock::now() - start_time < loop_time){
            step(mutation_ratio);
        }
        return best_x_;
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>, std::vector<double>> optimization_log(size_t loop_n, double mutation_ratio)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> x_log;
        std::vector<double> cost_log;
        for(size_t i = 0; i < N; i++){
            x_log[i].push_back(parent_individuals_[i].get_x());
        }
        cost_log.push_back(fitness_to_cost(best_fitness_));
        for(size_t i = 0; i < loop_n; i++){
            step(mutation_ratio);
            for(size_t j = 0; j < N; j++){
                x_log[j].push_back(parent_individuals_[j].get_x());
            }
            cost_log.push_back(fitness_to_cost(best_fitness_));
        }

        return {best_x_, x_log, cost_log};
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>, std::vector<double>> optimization_log(std::chrono::nanoseconds loop_time, double mutation_ratio)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> x_log;
        std::vector<double> cost_log;
        for(size_t i = 0; i < N; i++){
            x_log[i].push_back(parent_individuals_[i].get_x());
        }
        cost_log.push_back(fitness_to_cost(best_fitness_));
        auto start_time = std::chrono::system_clock::now();
        while(std::chrono::system_clock::now() - start_time < loop_time){
            step(mutation_ratio);
            for(size_t j = 0; j < N; j++){
                x_log[j].push_back(parent_individuals_[j].get_x());
            }
            cost_log.push_back(fitness_to_cost(best_fitness_));
        }

        return {best_x_, x_log, cost_log};
    }
};

}