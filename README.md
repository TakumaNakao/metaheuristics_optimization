# optimization

## 概要
目的関数の最小化を行うライブラリ  

## 環境
* C++17
* Eigen

## サンプルコード
matplotlibcppを使用
```c++
#include <iostream>
#include <cmath>
#include <chrono>

#include "matplotlibcpp.hpp"
#include "optimization/particle_swarm_optimization.hpp"
#include "optimization/artificial_bee_colony.hpp"
#include "optimization/genetic_algorithm.hpp"
#include "optimization/harmony_search.hpp"
#include "optimization/differential_evolution.hpp"

namespace plt = matplotlibcpp;

constexpr size_t dim = 2;
constexpr size_t element_num = 30;
constexpr size_t loop_num = 50;
constexpr std::chrono::duration loop_time = std::chrono::milliseconds(5);

namespace pso
{
    constexpr double w = 1.0;
    constexpr double c1 = 1.0;
    constexpr double c2 = 1.0;
}
namespace abc
{
    constexpr size_t count_limit = 10;
    constexpr double ga_rate = 0.0;
}
namespace ga
{
    constexpr double mutation_ratio = 0.2;
}
namespace hs
{
    constexpr double bandwidth = 0.1;
    constexpr double select_rate = 0.9;
    constexpr double change_rate = 0.5;
}
namespace de
{
    constexpr double crossover_rate = 0.5;
    constexpr double scaling = 0.6;
}

int main()
{
    using namespace std::chrono_literals;

    auto f = [](Eigen::Matrix<double, dim, 1> x) -> double {
        double x2_sum = 0;
        double cosx_sum = 0;
        for(size_t i = 0; i < dim; i++){
            x2_sum += x[i] * x[i];
            cosx_sum += std::cos(2 * M_PI * x[i]);
        }
        double t1 = 20;
        double t2 = -20 * std::exp(-0.2 * std::sqrt((1.0 / dim) * x2_sum));
        double t3 = std::exp(1.0);
        double t4 = -std::exp((1.0 / dim) * cosx_sum);
        return t1 + t2 + t3 + t4;
    };
    std::array<Eigen::Vector2d, dim> range;
    for(auto& r : range){
        r << -32.0 , 32.0;
    }

    // std::vector<std::vector<double>> x,y,z;
    // for(double i = range[0][0]; i <= range[0][1]; i += 0.5){
    //     std::vector<double> x_row,y_row,z_row;
    //     for(double j = range[0][0]; j <= range[0][1]; j += 0.5){
    //         Eigen::VectorXd v(2);
    //         v << i , j;
    //         x_row.push_back(i);
    //         y_row.push_back(j);
    //         z_row.push_back(f(v));
    //     }
    //     x.push_back(x_row);
    //     y.push_back(y_row);
    //     z.push_back(z_row);
    // }
    // plt::plot_surface(x, y, z);
    // plt::show();

    {
        optimization::ParticleSwarmOptimization<dim, element_num> pso(f, range);
        auto [result, x_log, cost_log] = pso.optimization_log(loop_time, pso::w, pso::c1, pso::c2);
        plt::named_plot("ParticleSwarmOptimization", cost_log);
    }
    {
        optimization::ArtificialBeeColony<dim, element_num> abc(f, range);
        auto [result, x_log, cost_log] = abc.optimization_log(loop_time, abc::count_limit, abc::ga_rate);
        plt::named_plot("ArtificialBeeColony", cost_log);
    }
    {
        optimization::GeneticAlgorithm<dim, element_num> ga(f, range);
        auto [result, x_log, cost_log] = ga.optimization_log(loop_time, ga::mutation_ratio);
        plt::named_plot("GeneticAlgorithm", cost_log);
    }
    {
        optimization::HarmonySearch<dim, element_num> hs(f, range);
        auto [result, x_log, cost_log] = hs.optimization_log(loop_time, hs::bandwidth, hs::select_rate, hs::change_rate);
        plt::named_plot("HarmonySearch", cost_log);
    }
    {
        optimization::DifferentialEvolution<dim, element_num> de(f, range);
        auto [result, x_log, cost_log] = de.optimization_log(loop_time, de::crossover_rate, de::scaling);
        plt::named_plot("DifferentialEvolution", cost_log);
    }
    plt::legend();
    plt::show();

    // for(size_t i = 0; i < cost_log.size(); i++){
    //     plt::clf();
    //     for(const auto& l : x_log){
    //         std::vector<double> x, y;
    //         x.push_back(l[i][0]);
    //         y.push_back(l[i][1]);
    //         plt::plot(x, y, ".");
    //         plt::text(-30, -30, std::to_string(i + 1) + "/" + std::to_string(cost_log.size()));
    //         plt::xlim(range[0][0], range[0][1]);
    //         plt::ylim(range[0][0], range[0][1]);
    //     }
    //     plt::pause(0.05);
    // }
}
```