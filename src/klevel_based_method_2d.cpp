/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <limits>
#include <boost/mp11/algorithm.hpp>
#include <utility>

#include <Eigen/Dense>

#include "utility.h"
#include "data_loader.h"
#include "experiments.h"
#include "tourney_tree.h"
#include "tie_breaking.h"
#include "stabilization.h"

struct GroupedLine {
    double k;
    double b;
    FairTopK::Groups maskedGroups;
};

struct GroupedScoredLine {
    double k;
    double b;
    FairTopK::Groups maskedGroups;
    double score;
};

template <std::size_t numGroups>
bool isFair(const std::vector<GroupedLine>& lines, int k, double time,
    const boost::container::small_vector<int, numGroups>& pGroupsVec, const std::vector<std::pair<int , int> >& pGroupsBounds,
    double epsilon) {
    int numPGroups = pGroupsBounds.size();

    int count = lines.size();

    std::vector<FairTopK::Groups> ties;
    ties.reserve(count);

    const GroupedLine& kthLine = lines[k - 1];
    double kthScore = kthLine.k * time + kthLine.b;

    std::array<int, FairTopK::maxNumGroups> pGroupsBaseCounts{};

    int vacant = 0;
    
    for (int i = 0; i < k; i++) {
        const GroupedLine& line = lines[i];
        double score = line.k * time + line.b;
        FairTopK::Groups maskedGroups = line.maskedGroups;

        if (score - kthScore > epsilon) {
            if (maskedGroups != 0)
                FairTopK::updatePGroupsCounts<1>(numPGroups, maskedGroups, pGroupsVec, pGroupsBaseCounts);
        }
        else {
            ties.push_back(maskedGroups);
            vacant += 1;
        }
    }

    for (int i = k; i < count; i++) {
        const GroupedLine& line = lines[i];
        double score = line.k * time + line.b;

        if (kthScore - score <= epsilon)
            ties.push_back(line.maskedGroups);
    }

    return FairTopK::searchFairSelection(vacant, pGroupsVec, pGroupsBounds, pGroupsBaseCounts, ties);
}

template <std::size_t numGroups>
std::pair<bool, double> tryObtainFairSelection(const std::vector<GroupedScoredLine>& lines, int k, double time,
    const boost::container::small_vector<int, numGroups>& pGroupsVec, const std::vector<std::pair<int, int> >& pGroupsBounds,
    double epsilon) {
    int count = lines.size();
    int numPGroups = pGroupsBounds.size();

    std::vector<FairTopK::Groups> tiesGroups; 
    std::vector<double> tiesScore;
    tiesGroups.reserve(count);
    tiesScore.reserve(count);

    const GroupedScoredLine& kthLine = lines[k - 1];
    double kthScore = kthLine.k * time + kthLine.b;

    std::array<int, FairTopK::maxNumGroups> pGroupsBaseCounts{};

    double utility = 0.0;
    int vacant = 0;
    
    for (int i = 0; i < k; i++) {
        const GroupedScoredLine& line = lines[i];
        double score = line.k * time + line.b;
        FairTopK::Groups maskedGroups = line.maskedGroups;

        if (score - kthScore > epsilon) {
            utility += line.score;
            if (maskedGroups != 0)
                FairTopK::updatePGroupsCounts<1>(numPGroups, maskedGroups, pGroupsVec, pGroupsBaseCounts);
        }
        else {
            tiesGroups.push_back(maskedGroups);
            tiesScore.push_back(line.score);

            vacant += 1;
        }
    }

    for (int i = k; i < count; i++) {
        const GroupedScoredLine& line = lines[i];
        double score = line.k * time + line.b;

        if (kthScore - score <= epsilon) {
            tiesGroups.push_back(line.maskedGroups);
            tiesScore.push_back(line.score);
        }
    }

    auto [fair, tiesUtility] = FairTopK::searchFairSelectionWithLargestUtility(vacant, pGroupsVec, pGroupsBounds,
        tiesGroups, tiesScore, pGroupsBaseCounts);

    return { fair, utility + tiesUtility };
}

template <class TopKTreeType, class OtherTreeType, class EquivCompare, std::size_t numGroups> 
inline void processTies(int numPGroups, const boost::container::small_vector<int, numGroups>& pGroupsVec, 
    EquivCompare &&testEquiv, TopKTreeType &topKTree, OtherTreeType &otherTree,
    int& vacant, std::array<int, numGroups>& pGroupsBaseCounts, std::vector<FairTopK::Groups>& tiesGroups) {
    
    auto handleTopKTops = [numPGroups, &tiesGroups, &pGroupsVec, &pGroupsBaseCounts, &vacant](const GroupedLine& line) {
        FairTopK::Groups maskedGroups = line.maskedGroups;
        tiesGroups.push_back(maskedGroups);

        if (maskedGroups != 0) {
            FairTopK::updatePGroupsCounts<-1>(numPGroups, maskedGroups, pGroupsVec, pGroupsBaseCounts);
        }

        vacant += 1;
    };

    auto handleOtherTops = [&tiesGroups](const GroupedLine& line) {
        tiesGroups.push_back(line.maskedGroups);
    };

    topKTree.applyToTopEquivs(std::forward<EquivCompare>(testEquiv), handleTopKTops);
    otherTree.applyToTopEquivs(std::forward<EquivCompare>(testEquiv), handleOtherTops);
}

template <class TopKTreeType, class OtherTreeType, class EquivCompare, std::size_t numGroups> 
inline void processTies(int numPGroups, const boost::container::small_vector<int, numGroups>& pGroupsVec, 
    EquivCompare &&testEquiv, TopKTreeType &topKTree, OtherTreeType &otherTree,
    int& vacant, std::array<int, numGroups>& pGroupsBaseCounts, double& baseUtility,
    std::vector<FairTopK::Groups>& tiesGroups, std::vector<double>& tiesScore) {
    
    auto handleTopKTops = [numPGroups, &tiesGroups, &tiesScore, &pGroupsVec, &pGroupsBaseCounts, 
        &baseUtility, &vacant](const GroupedScoredLine& line) {
        FairTopK::Groups maskedGroups = line.maskedGroups;
        tiesGroups.push_back(maskedGroups);
        tiesScore.push_back(line.score);

        if (maskedGroups != 0) {
            FairTopK::updatePGroupsCounts<-1>(numPGroups, maskedGroups, pGroupsVec, pGroupsBaseCounts);
        }

        baseUtility -= line.score;

        vacant += 1;
    };

    auto handleOtherTops = [&tiesGroups, &tiesScore](const GroupedScoredLine& line) {
        tiesGroups.push_back(line.maskedGroups);
        tiesScore.push_back(line.score);
    };

    topKTree.applyToTopEquivs(std::forward<EquivCompare>(testEquiv), handleTopKTops);
    otherTree.applyToTopEquivs(std::forward<EquivCompare>(testEquiv), handleOtherTops);
}

template <FairTopK::Optimization opt>
bool solve(const std::vector<Eigen::VectorXd> &points, const std::vector<FairTopK::Groups>& groups, 
    int k, FairTopK::GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, double margin,
    Eigen::VectorXd& weights) {
    constexpr double epsilon = 1e-8;

    double timeLower = std::max({ 0.0, weights(0) - margin, 1.0 - weights(1) - margin });
    double timeUpper = std::min({ 1.0, weights(0) + margin, 1.0 - weights(1) + margin });

    double inputTime = weights(0);
    Eigen::VectorXd refWeights = weights;

    constexpr bool optUtility = (opt == FairTopK::Optimization::Utility || opt == FairTopK::Optimization::StableUtility);

    using Line = std::conditional<optUtility, GroupedScoredLine, GroupedLine>::type;

    bool fair = false;
    double time = std::numeric_limits<double>::lowest();
    double maxUtility = std::numeric_limits<double>::lowest();

    int count = points.size();
    int numPGroups = pGroupsBounds.size();

    std::vector<Line> lines;
    lines.reserve(count);

    for (int i = 0; i < count; i++) {
        const auto &point = points[i];
        FairTopK::Groups maskedCandidateGroups = (groups[i] & pGroups);
        if constexpr (optUtility) {
            double score = point.dot(weights);
            lines.emplace_back(point(0) - point(1), point(1), maskedCandidateGroups, score);
        }
        else {
            lines.emplace_back(point(0) - point(1), point(1), maskedCandidateGroups);
        }
    }

    std::nth_element(lines.begin(), lines.begin() + (k - 1), lines.end(),
        [timeLower](const auto& l1, const auto& l2) { 
            double score1 = l1.k * timeLower + l1.b;
            double score2 = l2.k * timeLower + l2.b;
            double diff = score1 - score2;

            return std::abs(diff) > epsilon ? (diff > 0.0) : 
                (l1.k != l2.k ? l1.k > l2.k : l1.b > l2.b);
    });

    auto pGroupsVec = FairTopK::getPGroupsVec(pGroups);

    if constexpr (optUtility) {
        auto [isFair, utility] = tryObtainFairSelection(lines, k, timeLower, pGroupsVec, pGroupsBounds, epsilon);
        if (isFair) {
            fair = true;
            maxUtility = utility;

            time = timeLower;
        }
    }
    else if constexpr (opt == FairTopK::Optimization::WeightsDifference) {
        if (isFair(lines, k, timeLower, pGroupsVec, pGroupsBounds, epsilon)) {
            fair = true;
            time = timeLower;
        }
    }
    else {
        if (isFair(lines, k, timeLower, pGroupsVec, pGroupsBounds, epsilon)) {
            weights(0) = timeLower; 
            weights(1) = 1.0 - weights(0);
            
            return true;
        }
    }

    auto less = [](const Line& left, const Line& right, double time) noexcept {
        double score1 = left.k * time + left.b;
        double score2 = right.k * time + right.b;
        double diff = score1 - score2;

        return std::abs(diff) > epsilon ? (diff < 0.0) : 
            (left.k != right.k ? left.k < right.k : left.b < right.b);
    };

    auto greater = [](const Line& left, const Line& right, double time) noexcept {
        double score1 = left.k * time + left.b;
        double score2 = right.k * time + right.b;
        double diff = score1 - score2;

        return std::abs(diff) > epsilon ? (diff > 0.0) : 
            (left.k != right.k ? left.k > right.k : left.b > right.b);
    };

    auto crossCompute = [](const Line& left, const Line& right) noexcept -> double {
        if (left.k == right.k) return std::numeric_limits<double>::infinity();

        return (right.b - left.b) / (left.k - right.k);
    };

    using TopKTreeType = FairTopK::KineticTourneyLineTree<Line, decltype(less), decltype(crossCompute)>;
    using OtherTreeType = FairTopK::KineticTourneyLineTree<Line, decltype(greater), decltype(crossCompute)>;

    TopKTreeType topKTree(lines.cbegin(), lines.cbegin() + k, timeLower, timeUpper, less, crossCompute);
    OtherTreeType otherTree(lines.cbegin() + k, lines.cend(), timeLower, timeUpper, greater, crossCompute);

    std::array<int, FairTopK::maxNumGroups> pGroupsCounts{};
    for (int i = 0; i < k; i++) {
        FairTopK::Groups maskedGroups = lines[i].maskedGroups;
        if (maskedGroups != 0) {
            FairTopK::updatePGroupsCounts<1>(numPGroups, maskedGroups, pGroupsVec, pGroupsCounts);
        }
    }

    double utility = 0.0;
    if constexpr (optUtility) {
        for (int i = 0; i < k; i++) utility += lines[i].score;
    }

    std::vector<FairTopK::Groups> tiesGroups;
    tiesGroups.reserve(count);
    
    std::vector<double> tiesScore;
    if constexpr (optUtility) {
        tiesScore.reserve(count);
    }
    
    enum class AdvanceType { Exchange, TopK, Other };

    double curTime = timeLower;
    bool exchanged = false;
    while (true) {
        auto topKMin = topKTree.Top();
        auto otherMax = otherTree.Top();

        double exchangeTime = std::numeric_limits<double>::max();
        if (!exchanged && topKMin.k != otherMax.k && greater(otherMax, topKMin, timeUpper)) {
            exchangeTime = std::min((otherMax.b - topKMin.b) / (topKMin.k - otherMax.k), timeUpper);
        }

        double topKNextEventTime = topKTree.getNextEventTime();
        double otherNextEventTime = otherTree.getNextEventTime();

        double nextTime = exchangeTime;
        AdvanceType advanceType = AdvanceType::Exchange;
        if (topKNextEventTime < nextTime && !greater(otherMax, topKMin, topKNextEventTime)) {
            nextTime = topKNextEventTime;
            advanceType = AdvanceType::TopK;
        }
        if (otherNextEventTime < nextTime && !greater(otherMax, topKMin, otherNextEventTime)) {
            nextTime = otherNextEventTime;
            advanceType = AdvanceType::Other;
        }

        if (nextTime > timeUpper) break;

        double prevTime = curTime;

        curTime = std::max(curTime, nextTime);

        if (advanceType == AdvanceType::Exchange) {
            int vacant = 0;
            auto pGroupsBaseCounts = pGroupsCounts;

            double baseUtility = 0.0;
            if constexpr (optUtility) {
                baseUtility = utility;
            }
      
            while (topKTree.getNextEventTime() <= curTime + epsilon) {
                topKTree.Advance();
            }
            
            while (otherTree.getNextEventTime() <= curTime + epsilon) {
                otherTree.Advance();
            }

            auto testEquiv = [curTime](const Line& left, const Line& right) noexcept {
                double score1 = left.k * curTime + left.b;
                double score2 = right.k * curTime + right.b;
                
                return std::abs(score1 - score2) <= epsilon;
            };

            tiesGroups.clear();
            if constexpr (optUtility) {
                tiesScore.clear();
                processTies(numPGroups, pGroupsVec, testEquiv, topKTree, otherTree, 
                    vacant, pGroupsBaseCounts, baseUtility, tiesGroups, tiesScore);
            }
            else {
                processTies(numPGroups, pGroupsVec, testEquiv, topKTree, otherTree, 
                    vacant, pGroupsBaseCounts, tiesGroups);
            }

            if constexpr (optUtility) {
                auto [isFair, tiesUtility] = FairTopK::searchFairSelectionWithLargestUtility(vacant,
                     pGroupsVec, pGroupsBounds, tiesGroups, tiesScore, pGroupsBaseCounts);
                
                if (isFair) {
                    fair = true;
                    double curUtility = baseUtility + tiesUtility;
                    if (curUtility > maxUtility) {
                        time = curTime;
                        maxUtility = curUtility;
                    }
                    
                    if (curTime > inputTime) break;
                }
            }
            else if constexpr (opt == FairTopK::Optimization::WeightsDifference) {
                if (FairTopK::searchFairSelection(vacant, pGroupsVec, pGroupsBounds, pGroupsBaseCounts, tiesGroups)) {
                    fair = true;
                    if (std::abs(curTime - inputTime) < std::abs(time - inputTime)) {
                        time = curTime;
                    }

                    if (curTime > inputTime) break;
                }
            }
            else {
                if (FairTopK::searchFairSelection(vacant, pGroupsVec, pGroupsBounds, pGroupsBaseCounts, tiesGroups)) {
                    fair = true;
                    time = curTime; 

                    break;
                } 
            }

            topKMin = topKTree.Top();
            otherMax = otherTree.Top();

            if ((topKMin.maskedGroups & pGroups) != 0) {
                FairTopK::updatePGroupsCounts<-1>(numPGroups, topKMin.maskedGroups, pGroupsVec, pGroupsCounts);
            }
            if ((otherMax.maskedGroups & pGroups) != 0) {
                FairTopK::updatePGroupsCounts<1>(numPGroups, otherMax.maskedGroups, pGroupsVec, pGroupsCounts);
            }

            if constexpr (optUtility) {
                utility = utility - topKMin.score + otherMax.score;
            }

            bool topKMinChange = topKTree.replaceTop(otherMax);
            bool otherMaxChange = otherTree.replaceTop(topKMin);

            exchanged = true;

            bool topKTreeChange = true;
            bool otherTreeChange = true;
            while (topKTreeChange || otherTreeChange) {
                while (topKMinChange || otherMaxChange) {
                    topKMin = topKTree.Top();
                    otherMax = otherTree.Top();
                    if (greater(otherMax, topKMin, curTime)) {
                        if ((topKMin.maskedGroups & pGroups) != 0) {
                            FairTopK::updatePGroupsCounts<-1>(numPGroups, topKMin.maskedGroups, pGroupsVec, pGroupsCounts);
                        }         
                        if ((otherMax.maskedGroups & pGroups) != 0) {
                            FairTopK::updatePGroupsCounts<1>(numPGroups, otherMax.maskedGroups, pGroupsVec, pGroupsCounts);
                        }

                        if constexpr (optUtility) {
                            utility = utility - topKMin.score + otherMax.score;
                        }

                        topKMinChange = topKTree.replaceTop(otherMax);
                        otherMaxChange = otherTree.replaceTop(topKMin);
                    }
                    else {
                        topKMinChange = false;
                        otherMaxChange = false;
                    }
                }

                topKTreeChange = false;
                if (topKTree.getNextEventTime() <= curTime + epsilon) {
                    topKMinChange = topKTree.Advance();
                    topKTreeChange = true;
                }
                otherTreeChange = false;
                if (otherTree.getNextEventTime() <= curTime + epsilon) {
                    otherMaxChange = otherTree.Advance();
                    otherTreeChange = true;
                }
            }
            
            curTime += epsilon;
        }
        else if (advanceType == AdvanceType::TopK) {
            bool topChanged = topKTree.Advance();
            if (exchanged && topChanged) exchanged = false;
        }
        else {
            bool topChanged = otherTree.Advance();
            if (exchanged && topChanged) exchanged = false;
        }
    }

    if (fair) {
        weights(0) = time;
        weights(1) = 1.0 - time;

        if constexpr (opt == FairTopK::Optimization::StableUtility) {
            FairTopK::stabilizeFairWeightVector<2>(points, groups, k, pGroups, pGroupsBounds, refWeights, margin, epsilon, weights);
        }
    }

    return fair;
}

int main(int argc, char* argv[]) {
    std::vector<Eigen::VectorXd> points;
    std::vector<FairTopK::Groups> groups;
    FairTopK::GroupsMask protectedGroups = 0;

    auto [fileName, params] = FairTopK::parseCommandLine(argc, argv);

    bool success = FairTopK::DataLoader::readPreprocessedDataset(fileName, points, groups, protectedGroups);
    if (!success) return -1;

    int dimension = points[0].rows();
    if (dimension != 2) {
        std::cerr << "Error: Do not support datasets with dimensions != 2" << std::endl;
        return -1;
    }

    auto solveFunc = boost::mp11::mp_with_index<(std::size_t)FairTopK::Optimization::NumOptions>((std::size_t)params.opt,
        [](auto opt) { return solve<FairTopK::Optimization(opt())>; });

    FairTopK::fairTopkExperiments(points, groups, protectedGroups, params, solveFunc);

    return 0;
}