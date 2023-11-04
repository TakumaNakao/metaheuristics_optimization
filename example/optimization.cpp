#include <chrono>
#include <cmath>
#include <iostream>
#include <unordered_map>

#include <Eigen/Dense>
#include <matplot/matplot.h>

#include "artificial_bee_colony.hpp"
#include "differential_evolution.hpp"
#include "genetic_algorithm.hpp"
#include "harmony_search.hpp"
#include "particle_swarm_optimization.hpp"

namespace plt = matplot;

constexpr size_t dim = 2;
constexpr size_t element_num = 30;
constexpr size_t loop_num = 30;
constexpr std::chrono::duration loop_time = std::chrono::milliseconds(5);

namespace pso
{
constexpr double w = 1.0;
constexpr double c1 = 1.0;
constexpr double c2 = 1.0;
} // namespace pso
namespace abc
{
constexpr size_t count_limit = 10;
constexpr double ga_rate = 0.0;
} // namespace abc
namespace ga
{
constexpr double mutation_ratio = 0.2;
}
namespace hs
{
constexpr double bandwidth = 0.1;
constexpr double select_rate = 0.9;
constexpr double change_rate = 0.5;
} // namespace hs
namespace de
{
constexpr double crossover_rate = 0.5;
constexpr double scaling = 0.6;
} // namespace de

int main()
{
    using namespace std::chrono_literals;

    auto f = [](Eigen::Matrix<double, dim, 1> x) -> double {
        double x2_sum = 0;
        double cosx_sum = 0;
        for (size_t i = 0; i < dim; i++) {
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
    for (auto& r : range) {
        r << -32.0, 32.0;
    }

    std::unordered_map<std::string, std::array<std::vector<Eigen::Matrix<double, dim, 1>>, element_num>> x_log_map;
    std::unordered_map<std::string, std::vector<double>> cost_log_map;

    plt::hold(plt::on);
    {
        optimization::ParticleSwarmOptimization<dim, element_num> pso(f, range);
        auto [result, x_log, cost_log] = pso.optimization_log(loop_num, pso::w, pso::c1, pso::c2);
        x_log_map["particle swarm optimization"] = x_log;
        cost_log_map["particle swarm optimization"] = cost_log;
        plt::plot(cost_log)->line_width(3).display_name("ParticleSwarmOptimization");
    }
    {
        optimization::ArtificialBeeColony<dim, element_num> abc(f, range);
        auto [result, x_log, cost_log] = abc.optimization_log(loop_num, abc::count_limit, abc::ga_rate);
        x_log_map["artificial bee colony"] = x_log;
        cost_log_map["artificial bee colony"] = cost_log;
        plt::plot(cost_log)->line_width(3).display_name("ArtificialBeeColony");
    }
    {
        optimization::GeneticAlgorithm<dim, element_num> ga(f, range);
        auto [result, x_log, cost_log] = ga.optimization_log(loop_num, ga::mutation_ratio);
        x_log_map["genetic algorithm"] = x_log;
        cost_log_map["genetic algorithm"] = cost_log;
        plt::plot(cost_log)->line_width(3).display_name("GeneticAlgorithm");
    }
    {
        optimization::HarmonySearch<dim, element_num> hs(f, range);
        auto [result, x_log, cost_log] = hs.optimization_log(loop_num, hs::bandwidth, hs::select_rate, hs::change_rate);
        x_log_map["harmony search"] = x_log;
        cost_log_map["harmony search"] = cost_log;
        plt::plot(cost_log)->line_width(3).display_name("HarmonySearch");
    }
    {
        optimization::DifferentialEvolution<dim, element_num> de(f, range);
        auto [result, x_log, cost_log] = de.optimization_log(loop_num, de::crossover_rate, de::scaling);
        x_log_map["differential evolution"] = x_log;
        cost_log_map["differential evolution"] = cost_log;
        plt::plot(cost_log)->line_width(3).display_name("DifferentialEvolution");
    }
    plt::legend({});
    plt::xlabel("iteration");
    plt::ylabel("cost");
    plt::hold(plt::off);
    plt::save("img/cost.png");
    plt::cla();

    for (const auto& [key, cost_log] : cost_log_map) {
        for (size_t i = 0; i < cost_log.size(); i++) {
            std::vector<double> x, y;
            for (const auto& l : x_log_map[key]) {
                x.push_back(l[i][0]);
                y.push_back(l[i][1]);
            }
            plt::hold(plt::on);
            plt::plot(x, y, "b.");
            plt::plot({0.0}, {0.0}, "r*");
            plt::text(-30, -30, key + " : " + std::to_string(i + 1) + "/" + std::to_string(cost_log.size()));
            plt::xlim({range[0][0], range[0][1]});
            plt::ylim({range[1][0], range[1][1]});
            plt::hold(plt::off);
            plt::save("img/" + key + "/" + std::to_string(i) + ".png");
            plt::cla();
        }
    }
}