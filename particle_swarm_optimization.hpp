#pragma once

#include <array>
#include <functional>
#include <random>

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
        const std::function<double(Eigen::VectorXd)>& func_;
        Eigen::VectorXd x_ = Eigen::VectorXd::Zero(D);
        Eigen::VectorXd v_ = Eigen::VectorXd::Zero(D);
        Eigen::VectorXd best_x_ = Eigen::VectorXd::Zero(D);
        double best_score_ = std::numeric_limits<double>::max();
        std::mt19937 mt;
        std::uniform_real_distribution<> rand = std::uniform_real_distribution<>(0.0, 1.0);
        void update_best()
        {
            double score = func_(x_);
            if(score < best_score_){
                best_x_ = x_;
                best_score_ = score;
            }
        }
    public:
        Particle(const std::function<double(Eigen::VectorXd)>& func) : 
            func_(func)
        {
            std::random_device rnd;
            mt = std::mt19937(rnd());
        }
        void set_x(const Eigen::VectorXd& x)
        {
            x_ = x;
            update_best();
        }
        void update_state(const Eigen::VectorXd& g_best_x, double w, double c1, double c2)
        {
            v_ = w * v_ + c1 * rand(mt) * (best_x_ - x_) + c2 * rand(mt) * (g_best_x - x_);
            x_ += v_;
            update_best();
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
    std::function<double(Eigen::VectorXd)> func_;
    Eigen::VectorXd g_best_x_ = Eigen::VectorXd::Zero(D);
    double g_best_score_ = std::numeric_limits<double>::max();;

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
    ParticleSwarmOptimization(std::function<double(Eigen::VectorXd)> func, std::array<Eigen::VectorXd, N> init_x) : 
        func_(func)
    {
        particles_.reserve(N);
        for(size_t i = 0; i < N; i++){
            particles_.push_back(Particle(func_));
            particles_[i].set_x(init_x[i]);
        }
        update_g_best();
    }
    Eigen::VectorXd optimization(size_t loop_n, double w, double c1, double c2)
    {
        for(size_t i = 0; i < loop_n; i++){
            update_particle(w, c1, c2);
        }
        return g_best_x_;
    }
};

}