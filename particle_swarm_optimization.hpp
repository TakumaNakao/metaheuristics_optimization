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
        const std::function<double(Eigen::Matrix<double, D, 1>)>& func_;
        Eigen::Matrix<double, D, 1> x_ = Eigen::VectorXd::Zero(D);
        Eigen::Matrix<double, D, 1> v_ = Eigen::VectorXd::Zero(D);
        Eigen::Matrix<double, D, 1> best_x_ = Eigen::VectorXd::Zero(D);
        double best_score_ = std::numeric_limits<double>::max();
        std::array<Eigen::Vector2d, D> range_;
        std::mt19937 mt_;
        std::uniform_real_distribution<> rand_ = std::uniform_real_distribution<>(0.0, 1.0);
        void update_best()
        {
            double score = func_(x_);
            if(score < best_score_){
                best_x_ = x_;
                best_score_ = score;
            }
        }
    public:
        Particle(const std::function<double(Eigen::Matrix<double, D, 1>)>& func, std::array<Eigen::Vector2d, D> range, std::optional<Eigen::Matrix<double, D, 1>> init_x = std::nullopt) : 
            func_(func),
            range_(range)
        {
            std::random_device rnd;
            mt_ = std::mt19937(rnd());

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
        Eigen::VectorXd get_best_x() const
        {
            return best_x_;
        }
        double get_best_score() const
        {
            return best_score_;
        }
    };
private:
    std::vector<Particle> particles_;
    std::function<double(Eigen::Matrix<double, D, 1>)> func_;
    Eigen::Matrix<double, D, 1> g_best_x_ = Eigen::VectorXd::Zero(D);
    double g_best_score_ = std::numeric_limits<double>::max();

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
            if(particle.get_best_score() < g_best_score_){
                g_best_x_ = particle.get_best_x();
                g_best_score_ = particle.get_best_score();
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
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>> optimization_log(size_t loop_n, double w, double c1, double c2)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> log;
        for(size_t i = 0; i < N; i++){
            log[i].push_back(particles_[i].get_best_x());
        }
        for(size_t i = 0; i < loop_n; i++){
            update_particle(w, c1, c2);
            for(size_t j = 0; j < N; j++){
                log[j].push_back(particles_[j].get_best_x());
            }
        }

        return {g_best_x_, log};
    }
};

}