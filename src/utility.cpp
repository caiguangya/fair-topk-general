/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include "utility.h"
#include <limits>
#include <iostream>
#include <string_view>
#include <algorithm>

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include "CLI/CLI.hpp"

#include "tie_breaking.h"

namespace FairTopK {

bool checkFairness(const std::vector<Eigen::VectorXd> &points,
    const std::vector<Groups>& groups, const Eigen::VectorXd& weights,
    int k, int pGroup, int pGroupLowerBound, int pGroupUpperBound, double epsilon) {
    std::vector<std::pair<double, Groups> > pts;
    int count = points.size();

    auto pGroupBitmask = getGroupsMask(pGroup);

    pts.reserve(count);
    for (int i = 0; i < count; i++) {
        pts.emplace_back(weights.dot(points[i]), groups[i]);
    }

    std::nth_element(pts.begin(), pts.begin() + (k - 1), pts.end(),
        [](const auto& p0, const auto& p1) { return p0.first > p1.first; });

    double kthScore = pts[k - 1].first;

    int pGroupBaseCount = 0;
    int vacant = 0;
    int tieProtected = 0;
    int tieOther = 0;
    for (int i = 0; i < k; i++) {
        const auto [score, group] = pts[i];

        int isProtected = ((group & pGroupBitmask) != 0);
        if (score - kthScore > epsilon) {
            pGroupBaseCount += isProtected;
        }
        else {
            vacant += 1;
            tieProtected += isProtected;
            tieOther += 1 - isProtected;
        }
    }

    for (int i = k; i < count; i++) {
        const auto [score, group] = pts[i];

        if (kthScore - score <= epsilon) {
            int isProtected = ((group & pGroupBitmask) != 0);
            tieProtected += isProtected;
            tieOther += 1 - isProtected;
        }
    }

    int pGroupLowerCount = pGroupBaseCount + std::max(0, vacant - tieOther);
    int pGroupUpperCount = pGroupBaseCount + vacant - std::max(0, vacant - tieProtected);

    return std::max(pGroupLowerCount, pGroupLowerBound) <= std::min(pGroupUpperCount, pGroupUpperBound);
}

bool checkFairness(const std::vector<Eigen::VectorXd> &points,
    const std::vector<Groups>& groups, const Eigen::VectorXd& weights,
    int k, GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, double epsilon) {
    std::vector<std::pair<double, Groups> > pts;
    int count = points.size();

    pts.reserve(count);
    for (int i = 0; i < count; i++) {
        pts.emplace_back(weights.dot(points[i]), groups[i]);
    }

    std::nth_element(pts.begin(), pts.begin() + (k - 1), pts.end(),
        [](const auto& p0, const auto& p1) { return p0.first > p1.first; });

    double kthScore = pts[k - 1].first;

    int numPGroups = pGroupsBounds.size();
    auto pGroupsVec = getPGroupsVec(pGroups);

    std::array<int, maxNumGroups> pGroupsBaseCounts{};
    
    std::vector<Groups> ties;
    ties.reserve(count);

    int vacant = 0;
    
    for (int i = 0; i < k; i++) {
        const auto [score, groups] = pts[i];

        if (score - kthScore > epsilon) {
            if ((groups & pGroups) != 0)
                updatePGroupsCounts<1>(numPGroups, groups, pGroupsVec, pGroupsBaseCounts);
        }
        else {
            ties.push_back(groups & pGroups);
            vacant += 1;
        }
    }

    for (int i = k; i < count; i++) {
        const auto [score, groups] = pts[i];

        if (kthScore - score <= epsilon)
            ties.push_back(groups & pGroups);
    }

    return FairTopK::searchFairSelection(vacant, pGroupsVec, pGroupsBounds, pGroupsBaseCounts, ties);
}

Eigen::VectorXd getRandomWeightVector(int dimension, std::default_random_engine& rand) {
    std::vector<double> rngGenNums(dimension, 0.0);
    std::uniform_real_distribution<double> dis(0.0, 1.0 + std::numeric_limits<double>::epsilon());
    for (int i = 0; i < dimension; i++) {
        rngGenNums[i] = dis(rand);
    }
    std::sort(rngGenNums.begin(), rngGenNums.end());

    Eigen::VectorXd weights(dimension);
    weights(0) = rngGenNums[0];
    for (int i = 1; i < dimension - 1; i++) {
        weights(i) = rngGenNums[i] - rngGenNums[i - 1];
    }
    weights(dimension - 1) = 1.0 - rngGenNums[dimension - 1];

    weights = weights.cwiseAbs();
    weights /= weights.sum();

    return weights;
}

std::vector<Eigen::VectorXd> getRandomWeightVectors(int count, int dimension) {
    std::vector<Eigen::VectorXd> vectors;
    vectors.reserve(count);

    std::default_random_engine rand(2024);

    for (int i = 0; i < count; i++) {
        Eigen::VectorXd weights = getRandomWeightVector(dimension, rand);

        vectors.push_back(std::move(weights));
    }

    return vectors;
}

std::vector<Eigen::VectorXd> getRandomWeightVectors(int count, const std::vector<Eigen::VectorXd> &points, 
    const std::vector<Groups>& groups, int k, GroupsMask pGroups,
    const std::vector<std::pair<int, int> >& pGroupsBounds, bool pessimistic) {
    std::vector<Eigen::VectorXd> vectors;
    vectors.reserve(count);

    std::default_random_engine rand(2024);

    int dimension = points[0].rows();
    int numPGroups = pGroupsBounds.size();
    auto pGroupsVec = getPGroupsVec(pGroups);
    while (vectors.size() < count) {
        Eigen::VectorXd weights = getRandomWeightVector(dimension, rand);

        bool unfair = true;
        if (pessimistic) {
            for (int i = 0; i < numPGroups; i++) {
                auto [lowerBound, upperBound] = pGroupsBounds[i];
                int pGroup = pGroupsVec[i];
                
                if (checkFairness(points, groups, weights, k, pGroup, lowerBound, upperBound)) {
                    unfair = false;
                    break;
                }
            }
        }
        else {
            unfair = !checkFairness(points, groups, weights, k, pGroups, pGroupsBounds);
        }

        if (unfair) vectors.push_back(std::move(weights));
    }    

    return vectors;
}

namespace {

void printInputInfos(int k, double margin, 
    const std::vector<boost::multiprecision::cpp_dec_float_50> &pGroupLowerBounds,
    const std::vector<boost::multiprecision::cpp_dec_float_50> &pGroupUpperBounds,
    Optimization opt, int threadCount) {
    std::cout << "k: " << k;

    std::cout << " | Protected Group Proportion Bounds: ";
    int numPGroups = std::max(pGroupLowerBounds.size(), pGroupUpperBounds.size());
    for (int i = 0; i < numPGroups; i++)
        std::cout << "[" << pGroupLowerBounds[i] << ", " << pGroupUpperBounds[i] << "] ";

    std::cout << "| Epsilon: " << margin;

    std::cout << " | Optimization Goal: ";
    if (opt == Optimization::Utility) {
        std::cout << "Utility";
    }
    else if (opt == Optimization::WeightsDifference) {
        std::cout << "Weights Difference";
    }
    else if (opt == Optimization::StableUtility) {
        std::cout << "Stable Utility";
    }
    else {
        std::cout << "None";
    }

    if (threadCount > 0)
        std::cout << " | Number of Threads: " << threadCount;
        
    std::cout << std::endl;
}

}

std::pair<std::string, InputParams> parseCommandLine(int argc, char* argv[]) {
    CLI::App app;
    std::string file;
    InputParams inputParams;
    constexpr double epsilon = std::numeric_limits<double>::epsilon();

    std::vector<std::string> pGroupLowerBoundRatioStrs;
    std::vector<std::string> pGroupUpperBoundRatioStrs;

    bool optUtility = false;
    bool optWeights = false;

    std::string optOption;

    app.allow_non_standard_option_names();

    app.add_option("-f", file, "File");
    app.add_option("-k", inputParams.k, "k");
    app.add_option("-eps", inputParams.margin, "Epsilon");
    app.add_option("-plb", pGroupLowerBoundRatioStrs, "Protected Group lower bound");
    app.add_option("-pub", pGroupUpperBoundRatioStrs, "Protected Group upper bound");
    app.add_option("-nt", inputParams.threadCount, "Number of threads");
    app.add_option("-ns", inputParams.sampleCount, "Number of samples");
    app.add_option("-sol", inputParams.solver, "MILP Solver");

    app.add_flag("-t", inputParams.runtime, "Runtime");
    app.add_flag("-q", inputParams.quality, "Evaluate Quality");
    app.add_flag("-us", inputParams.uniformSampling, "Uniform sampling method");
    app.add_flag("-ps", inputParams.pessSampling, "Pessimistic sampling method");

    app.add_option("-opt", optOption, "Optimization option");

    app.parse(argc, argv);

    boost::algorithm::to_lower(inputParams.solver);

    int boundCounts = std::max(pGroupLowerBoundRatioStrs.size(), pGroupUpperBoundRatioStrs.size());
    inputParams.pGroupsBounds.reserve(boundCounts);

    while (pGroupLowerBoundRatioStrs.size() < boundCounts) pGroupLowerBoundRatioStrs.push_back("0.0");
    while (pGroupUpperBoundRatioStrs.size() < boundCounts) pGroupUpperBoundRatioStrs.push_back("1.0");

    using cpp_dec_float_50 = boost::multiprecision::cpp_dec_float_50;
    std::vector<cpp_dec_float_50> pGroupLowerBoundRatios(pGroupLowerBoundRatioStrs.cbegin(), pGroupLowerBoundRatioStrs.cend());
    std::vector<cpp_dec_float_50> pGroupUpperBoundRatios(pGroupUpperBoundRatioStrs.cbegin(), pGroupUpperBoundRatioStrs.cend());

    for (int i = 0; i < boundCounts; i++) {
        int pGroupLowerBound = (int)std::floor((pGroupLowerBoundRatios[i] * inputParams.k).convert_to<double>());
        int pGroupUpperBound = (int)std::ceil((pGroupUpperBoundRatios[i] * inputParams.k).convert_to<double>());

        inputParams.pGroupsBounds.emplace_back(pGroupLowerBound, pGroupUpperBound);
    }

    constexpr std::array<std::string_view, 3> utility = {"u", "util", "utility" };
    constexpr std::array<std::string_view, 3> weightsDiff = { "wd", "wtsdiff", "weightsdifference" };
    constexpr std::array<std::string_view, 3> stbUtility = {"su", "stbutil", "stableutility" };

    boost::algorithm::to_lower(optOption);
    if (std::find(utility.cbegin(), utility.cend(), optOption) != utility.cend()) {
        inputParams.opt = Optimization::Utility;
    }
    else if (std::find(weightsDiff.cbegin(), weightsDiff.cend(), optOption) != weightsDiff.cend()) {
        inputParams.opt = Optimization::WeightsDifference;
    }
    else if (std::find(stbUtility.cbegin(), stbUtility.cend(), optOption) != stbUtility.cend()) {
        inputParams.opt = Optimization::StableUtility;
    }
    else {
        if (!optOption.empty()) {
            std::cout << "Unsupported optimization option. Defaulting to None" << std::endl;
        }
    }

    printInputInfos(inputParams.k, inputParams.margin, pGroupLowerBoundRatios, pGroupUpperBoundRatios, 
        inputParams.opt, inputParams.threadCount);

    return { std::move(file), std::move(inputParams) };
}

bool checkProtectedGroupsBoundsCount(GroupsMask pGroups, const std::vector<std::pair<int, int> > &pGroupsBounds) {
    if (pGroupsBounds.empty()) {
        std::cerr << "Error: Protected group bounds are not specified." << std::endl;
        return false;
    }

    GroupsMask pGroupsBitmask = pGroups;
    
    int pGroupCount = 0;
    while (pGroupsBitmask != 0) {
        if ((pGroupsBitmask & 1) != 0)
            pGroupCount += 1;
        
        pGroupsBitmask = pGroupsBitmask >> 1;
    }
    int pGroupsBoundsCount = pGroupsBounds.size();

    bool valid = (pGroupCount >= pGroupsBoundsCount);

    if (!valid) {
        std::cerr << "Error: The number of protected groups (" << pGroupCount << ")" <<
            " is smaller than the number of their bounds (" << pGroupsBoundsCount << ")." << std::endl;
    }

    return valid;
}

}