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
class ArtificialBeeColony
{
public:
    class Bee
    {
    private:
        const std::function<double(Eigen::Matrix<double, D, 1>)>& func_;
        Eigen::Matrix<double, D, 1> x_ = Eigen::VectorXd::Zero(D);
        double score_ = std::numeric_limits<double>::max();
        size_t count_ = 0;
        std::array<Eigen::Vector2d, D> range_;
        std::mt19937 mt_;
        std::uniform_real_distribution<> employed_rand_ = std::uniform_real_distribution<>(-1.0, 1.0);
        std::uniform_real_distribution<> scout_rand_ = std::uniform_real_distribution<>(0.0, 1.0);
    public:
        Bee(const std::function<double(Eigen::Matrix<double, D, 1>)>& func, std::array<Eigen::Vector2d, D> range, std::optional<Eigen::Matrix<double, D, 1>> init_x = std::nullopt) : 
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
        void set_x(const Eigen::Matrix<double, D, 1>& x, std::optional<double> score = std::nullopt)
        {
            for(size_t i = 0; i < D; i++){
                x_[i] = std::clamp(x[i], range_[i][0], range_[i][1]);
            }
            if(score){
                score_ = score.value();
            }
            else{
                score_ = func_(x_);
            }
        }
        void employed_bee(int update_dim, const Eigen::Matrix<double, D, 1>& rand_bee_x)
        {
            double vd = x_[update_dim] + employed_rand_(mt_) * (x_[update_dim] - rand_bee_x[update_dim]);
            Eigen::Matrix<double, D, 1> v = x_;
            v[update_dim] = vd;
            double v_score = func_(v);
            if(v_score < score_){
                set_x(v, v_score);
                count_ = 0;
            }
            else{
                count_++;
            }
        }
        void onlooker_bee()
        {
            count_++;
        }
        void scout_bee(size_t count_limit)
        {
            if(count_ > count_limit){
                Eigen::Matrix<double, D, 1> x;
                for(size_t i = 0; i < D; i++){
                    x[i] = range_[i][0] + scout_rand_(mt_) * (range_[i][1] - range_[i][0]);
                }
                set_x(x);
            }
        }
        Eigen::VectorXd get_x() const
        {
            return x_;
        }
        double get_score() const
        {
            return score_;
        }
    };
private:
    std::vector<Bee> bees_;
    std::function<double(Eigen::Matrix<double, D, 1>)> func_;
    Eigen::Matrix<double, D, 1> best_x_ = Eigen::VectorXd::Zero(D);
    double best_score_ = std::numeric_limits<double>::max();
    std::mt19937 mt_;
    std::uniform_int_distribution<> dim_rand_ = std::uniform_int_distribution<>(0, D - 1);
    std::uniform_int_distribution<> bee_rand_ = std::uniform_int_distribution<>(0, N - 1);
    std::uniform_real_distribution<> onlooker_rand_ = std::uniform_real_distribution<>(0.0, 1.0);

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
    void update_bee(size_t count_limit)
    {
        int update_dim = dim_rand_(mt_);
        double score_sum = 0;
        for(size_t i = 0; i < N; i++){
            bees_[i].employed_bee(update_dim, bees_[gen_rand_bee_id(i)].get_x());
            score_sum += bees_[i].get_score();
        }
        if(score_sum != 0){
            for(size_t i = 0; i < N; i++){
                if(bees_[i].get_score() / score_sum > onlooker_rand_(mt_)){
                    // bees_[i].onlooker_bee();
                    bees_[i].employed_bee(update_dim, bees_[gen_rand_bee_id(i)].get_x());
                }
            }
        }
        for(auto& bee : bees_){
            bee.scout_bee(count_limit);
        }
        update_best();
    }
    void update_best()
    {
        for(const auto& bee : bees_){
            if(bee.get_score() < best_score_){
                best_x_ = bee.get_x();
                best_score_ = bee.get_score();
            }
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
    Eigen::Matrix<double, D, 1> optimization(size_t loop_n, size_t count_limit)
    {
        for(size_t i = 0; i < loop_n; i++){
            update_bee(count_limit);
        }
        return best_x_;
    }
    std::tuple<Eigen::Matrix<double, D, 1>, std::array<std::vector<Eigen::Matrix<double, D, 1>>, N>> optimization_log(size_t loop_n, size_t count_limit)
    {
        std::array<std::vector<Eigen::Matrix<double, D, 1>>, N> log;
        for(size_t i = 0; i < N; i++){
            log[i].push_back(bees_[i].get_x());
        }
        for(size_t i = 0; i < loop_n; i++){
            update_bee(count_limit);
            for(size_t j = 0; j < N; j++){
                log[j].push_back(bees_[j].get_x());
            }
        }

        return {best_x_, log};
    }
};

}