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
class ArtificialBeeColony
{
public:
    class Bee
    {
    private:
        inline static std::random_device rnd_;
        inline static std::mt19937 mt_ = std::mt19937(rnd_());
        const std::function<double(Eigen::Matrix<double, D, 1>)>& func_;
        Eigen::Matrix<double, D, 1> x_ = Eigen::VectorXd::Zero(D);
        double fitness_ = 0;
        size_t count_ = 0;
        std::array<Eigen::Vector2d, D> range_;
        std::uniform_int_distribution<> dim_rand_ = std::uniform_int_distribution<>(0, D - 1);
        std::uniform_real_distribution<> probability_rand_ = std::uniform_real_distribution<>(0.0, 1.0);

        double calc_fitness(Eigen::Matrix<double, D, 1> x)
        {
            double cost = func_(x);
            if(cost >= 0){
                return 1.0 / (1.0 + cost);
            }
            else{
                return 1.0 + std::fabs(cost);
            }
        }
    public:
        Bee(const std::function<double(Eigen::Matrix<double, D, 1>)>& func, std::array<Eigen::Vector2d, D> range, std::optional<Eigen::Matrix<double, D, 1>> init_x = std::nullopt) : 
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
        void set_x(const Eigen::Matrix<double, D, 1>& x, std::optional<double> fitness = std::nullopt)
        {
            for(size_t i = 0; i < D; i++){
                x_[i] = std::clamp(x[i], range_[i][0], range_[i][1]);
            }
            if(fitness){
                fitness_ = fitness.value();
            }
            else{
                fitness_ = calc_fitness(x_);
            }
        }
        void move_bee(const Eigen::Matrix<double, D, 1>& rand_bee_x, double ga_rate)
        {
            int update_dim = dim_rand_(mt_);
            Eigen::Matrix<double, D, 1> v = x_;
            for(size_t i = 0; i < D; i++){
                if(i == update_dim || probability_rand_(mt_) < ga_rate){
                    v[i] = x_[i] + (probability_rand_(mt_) * 2.0 - 1.0) * (x_[i] - rand_bee_x[i]);
                }
            }
            double v_fitness = calc_fitness(v);
            if(v_fitness > fitness_){
                set_x(v, v_fitness);
                count_ = 0;
            }
            else{
                count_++;
            }
        }
        void scout_bee(size_t count_limit)
        {
            if(count_ > count_limit){
                Eigen::Matrix<double, D, 1> x;
                for(size_t i = 0; i < D; i++){
                    x[i] = range_[i][0] + probability_rand_(mt_) * (range_[i][1] - range_[i][0]);
                }
                set_x(x);
                count_ = 0;
            }
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
    std::vector<Bee> bees_;
    std::function<double(Eigen::Matrix<double, D, 1>)> func_;
    Eigen::Matrix<double, D, 1> best_x_ = Eigen::VectorXd::Zero(D);
    double best_fitness_ = 0;
    std::mt19937 mt_;
    std::uniform_int_distribution<> bee_rand_ = std::uniform_int_distribution<>(0, N - 1);
    std::uniform_real_distribution<> probability_rand_ = std::uniform_real_distribution<>(0.0, 1.0);

    int gen_rand_bee_id(int exclusion)
    {
        while(1)
        {
            int id = bee_rand_(mt_);
            if(id != exclusion){
                return id;
            }
        }
    }
    void update_best()
    {
        for(const auto& bee : bees_){
            if(bee.get_fitness() > best_fitness_){
                best_x_ = bee.get_x();
                best_fitness_ = bee.get_fitness();
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
    ArtificialBeeColony(std::function<double(Eigen::Matrix<double, D, 1>)> func, std::array<Eigen::Vector2d, D> range, std::optional<std::array<Eigen::Matrix<double, D, 1>, N>> init_x = std::nullopt) : 
        func_(func)
    {
        std::random_device rnd;
        mt_ = std::mt19937(rnd());

        bees_.reserve(N);
        if(init_x){
            for(size_t i = 0; i < N; i++){
                bees_.push_back(Bee(func_, range, init_x.value()[i]));
            }
        }
        else{
            for(size_t i = 0; i < N; i++){
                bees_.push_back(Bee(func_, range));
            }
        }
        update_best();
    }
    void step(size_t count_limit, double ga_rate = 0.0)
    {
        double fitness_sum = 0;
        for(size_t i = 0; i < N; i++){
            bees_[i].move_bee(bees_[gen_rand_bee_id(i)].get_x(), ga_rate);
            fitness_sum += bees_[i].get_fitness();
        }
        for(size_t i = 0; i < N; i++){
            if(bees_[i].get_fitness() > probability_rand_(mt_) * fitness_sum){
                bees_[i].move_bee(bees_[gen_rand_bee_id(i)].get_x(), ga_rate);
            }
        }
        for(auto& bee : bees_){
            bee.scout_bee(count_limit);
        }
        update_best();
    }
    Eigen::Matrix<double, D, 1> optimization(size_t loop_n, size_t count_limit, double ga_rate = 0.0)
    {
        for(size_t i = 0; i < loop_n; i++){
            step(count_limit, ga_rate);
        }
        return best_x_;
    }
    Eigen::Matrix<double, D, 1> optimization(std::chrono::nanoseconds loop_time, size_t count_limit, double ga_rate = 0.0)
    {
        auto start_time = std::chrono::system_clock::now();
        while(std::chrono::system_clock::now() - start_time < loop_time){
            step(count_limit, ga_rate);
        }
        return best_x_;
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>, std::vector<double>> optimization_log(size_t loop_n, size_t count_limit, double ga_rate = 0.0)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> x_log;
        std::vector<double> cost_log;
        for(size_t i = 0; i < N; i++){
            x_log[i].push_back(bees_[i].get_x());
        }
        cost_log.push_back(fitness_to_cost(best_fitness_));
        for(size_t i = 0; i < loop_n; i++){
            step(count_limit, ga_rate);
            for(size_t j = 0; j < N; j++){
                x_log[j].push_back(bees_[j].get_x());
            }
            cost_log.push_back(fitness_to_cost(best_fitness_));
        }

        return {best_x_, x_log, cost_log};
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>, std::vector<double>> optimization_log(std::chrono::nanoseconds loop_time, size_t count_limit, double ga_rate = 0.0)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> x_log;
        std::vector<double> cost_log;
        for(size_t i = 0; i < N; i++){
            x_log[i].push_back(bees_[i].get_x());
        }
        cost_log.push_back(fitness_to_cost(best_fitness_));
        auto start_time = std::chrono::system_clock::now();
        while(std::chrono::system_clock::now() - start_time < loop_time){
            step(count_limit, ga_rate);
            for(size_t j = 0; j < N; j++){
                x_log[j].push_back(bees_[j].get_x());
            }
            cost_log.push_back(fitness_to_cost(best_fitness_));
        }

        return {best_x_, x_log, cost_log};
    }
};

}