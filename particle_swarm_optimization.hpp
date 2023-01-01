#pragma once

#include <array>
#include <vector>
#include <functional>
#include <optional>
#include <random>
#include <tuple>

#include <Eigen/Core>

namespace optimization
{

template <size_t D, size_t N>
class ParticleSwarmOptimization
{
public:
    class Particle
    {
    private:
        inline static std::random_device rnd_;
        inline static std::mt19937 mt_ = std::mt19937(rnd_());
        const std::function<double(Eigen::Matrix<double, D, 1>)>& func_;
        Eigen::Matrix<double, D, 1> x_ = Eigen::VectorXd::Zero(D);
        Eigen::Matrix<double, D, 1> v_ = Eigen::VectorXd::Zero(D);
        Eigen::Matrix<double, D, 1> best_x_ = Eigen::VectorXd::Zero(D);
        double best_cost_ = std::numeric_limits<double>::max();
        std::array<Eigen::Vector2d, D> range_;
        std::uniform_real_distribution<> rand_ = std::uniform_real_distribution<>(0.0, 1.0);
        void update_best()
        {
            double cost = func_(x_);
            if(cost < best_cost_){
                best_x_ = x_;
                best_cost_ = cost;
            }
        }
    public:
        Particle(const std::function<double(Eigen::Matrix<double, D, 1>)>& func, std::array<Eigen::Vector2d, D> range, std::optional<Eigen::Matrix<double, D, 1>> init_x = std::nullopt) : 
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
        void set_x(const Eigen::Matrix<double, D, 1>& x)
        {
            for(size_t i = 0; i < D; i++){
                x_[i] = std::clamp(x[i], range_[i][0], range_[i][1]);
            }
            update_best();
        }
        void update_state(const Eigen::Matrix<double, D, 1>& g_best_x, double w, double c1, double c2)
        {
            v_ = w * v_ + c1 * rand_(mt_) * (best_x_ - x_) + c2 * rand_(mt_) * (g_best_x - x_);
            set_x(x_ + v_);
        }
        Eigen::Matrix<double, D, 1> get_best_x() const
        {
            return best_x_;
        }
        double get_best_cost() const
        {
            return best_cost_;
        }
    };
private:
    std::vector<Particle> particles_;
    std::function<double(Eigen::Matrix<double, D, 1>)> func_;
    Eigen::Matrix<double, D, 1> g_best_x_ = Eigen::VectorXd::Zero(D);
    double g_best_cost_ = std::numeric_limits<double>::max();

    void update_particle(double w, double c1, double c2)
    {
        for(auto& particle : particles_){
            particle.update_state(g_best_x_, w, c1, c2);
        }
        update_g_best();
    }
    void update_g_best()
    {
        for(const auto& particle : particles_){
            if(particle.get_best_cost() < g_best_cost_){
                g_best_x_ = particle.get_best_x();
                g_best_cost_ = particle.get_best_cost();
            }
        }
    }
public:
    ParticleSwarmOptimization(std::function<double(Eigen::Matrix<double, D, 1>)> func, std::array<Eigen::Vector2d, D> range, std::optional<std::array<Eigen::Matrix<double, D, 1>, N>> init_x = std::nullopt) : 
        func_(func)
    {
        particles_.reserve(N);
        if(init_x){
            for(size_t i = 0; i < N; i++){
                particles_.push_back(Particle(func_, range, init_x.value()[i]));
            }
        }
        else{
            for(size_t i = 0; i < N; i++){
                particles_.push_back(Particle(func_, range));
            }
        }
        update_g_best();
    }
    Eigen::Matrix<double, D, 1> optimization(size_t loop_n, double w, double c1, double c2)
    {
        for(size_t i = 0; i < loop_n; i++){
            update_particle(w, c1, c2);
        }
        return g_best_x_;
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>, std::vector<double>> optimization_log(size_t loop_n, double w, double c1, double c2)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> x_log;
        std::vector<double> cost_log;
        for(size_t i = 0; i < N; i++){
            x_log[i].push_back(particles_[i].get_best_x());
        }
        cost_log.push_back(g_best_cost_);
        for(size_t i = 0; i < loop_n; i++){
            update_particle(w, c1, c2);
            for(size_t j = 0; j < N; j++){
                x_log[j].push_back(particles_[j].get_best_x());
            }
            cost_log.push_back(g_best_cost_);
        }

        return {g_best_x_, x_log, cost_log};
    }
};

}