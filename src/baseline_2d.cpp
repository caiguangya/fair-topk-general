/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <queue>
#include <tuple>
#include <limits>
#include <cmath>
#include <boost/mp11/algorithm.hpp>
#include <utility>
#include <algorithm>

#include <Eigen/Dense>

#include "utility.h"
#include "data_loader.h"
#include "experiments.h"
#include "memory.h"
#include "tie_breaking.h"
#include "stabilization.h"

template <typename Line>
concept SlopeInterceptForm = requires(Line line) {
    { line.k } -> std::convertible_to<double>;
    { line.b } -> std::convertible_to<double>;
};

struct GroupedLine {
    double k;
    double b;
    int rank;
    int count;
    int *pGroupsCounts;
    FairTopK::Groups *maskedGroups;
};

struct GroupedScoredLine {
    double k;
    double b;
    int rank;
    int count;
    int *pGroupsCounts;
    FairTopK::Groups *maskedGroups;
    double score;
};

template <SlopeInterceptForm Line>
inline double computeIntersect(const Line& front, const Line& back, double upperLimit) noexcept {
    double scoreFront = front.k * upperLimit + front.b;
    double scoreBack = back.k * upperLimit + back.b;
    if (scoreFront < scoreBack && front.k != back.k) {
        return (front.b - back.b) / (back.k - front.k);
    }

    return std::numeric_limits<double>::infinity();
}

template <bool inc, std::size_t numGroups>
inline void updatePGroupsBaseCounts(int numPGroups,
    const int *pGroupsCounts, std::array<int, numGroups>& pGroupsBaseCounts) {
    for (int i = 0; i < numPGroups; i++) {
        if constexpr (inc) pGroupsBaseCounts[i] += pGroupsCounts[i];
        else pGroupsBaseCounts[i] -= pGroupsCounts[i];
    }
}

template <std::size_t numGroups>
bool isFair(const std::vector<GroupedLine *>& lines, int k, double sweepLinePos, int kthIdx, int kthTieCount, 
    const std::array<int, numGroups>& pGroupsBaseCounts, const boost::container::small_vector<int, numGroups>& pGroupsVec, 
    const std::vector<std::pair<int , int> >& pGroupsBounds, double epsilon) {
    int numPGroups = pGroupsBounds.size();
    const GroupedLine* kthLine = lines[kthIdx];

    std::array<int, numGroups> curPGroupsBaseCounts = pGroupsBaseCounts;

    std::vector<FairTopK::Groups> ties;
    double kthScore = kthLine->k * sweepLinePos + kthLine->b;

    std::copy(kthLine->maskedGroups, kthLine->maskedGroups + kthLine->count, std::back_inserter(ties));

    int vacant = kthTieCount;
    int idx = kthIdx - 1;
    while (idx >= 0) {
        const GroupedLine* line = lines[idx];
        double score = line->k * sweepLinePos + line->b;
        if (std::abs(score - kthScore) > epsilon) {
            break;
        }

        updatePGroupsBaseCounts<false>(numPGroups, line->pGroupsCounts, curPGroupsBaseCounts);

        const auto maskedGroups = line->maskedGroups;
        std::copy(maskedGroups, maskedGroups + line->count, std::back_inserter(ties));

        vacant += line->count;
        idx -= 1;
    }

    idx = kthIdx + 1;
    int count = lines.size();
    while (idx < count) {
        const GroupedLine* line = lines[idx];
        double score = line->k * sweepLinePos + line->b;
        if (std::abs(score - kthScore) > epsilon) {
            break;
        }

        const auto maskedGroups = line->maskedGroups;
        std::copy(maskedGroups, maskedGroups + line->count, std::back_inserter(ties));

        idx += 1;
    }

    return FairTopK::searchFairSelection(vacant, pGroupsVec, pGroupsBounds, curPGroupsBaseCounts, ties);
}

template <std::size_t numGroups>
std::pair<bool, double> tryObtainFairSelection(const std::vector<GroupedScoredLine *>& lines, int k, double sweepLinePos, 
    int kthIdx, int kthTieCount, const std::array<int, numGroups>& pGroupsBaseCounts,
    const boost::container::small_vector<int, numGroups>& pGroupsVec, 
    const std::vector<std::pair<int , int> >& pGroupsBounds, double baseUtility, double epsilon) {
    int numPGroups = pGroupsBounds.size();
    const GroupedScoredLine* kthLine = lines[kthIdx];

    std::array<int, numGroups> curPGroupsBaseCounts = pGroupsBaseCounts;
    double curBaseUtility = baseUtility;

    std::vector<FairTopK::Groups> tiesGroups;
    std::vector<double> tiesScore;
    double kthScore = kthLine->k * sweepLinePos + kthLine->b;

    std::copy(kthLine->maskedGroups, kthLine->maskedGroups + kthLine->count, std::back_inserter(tiesGroups));
    std::fill_n(std::back_inserter(tiesScore), kthLine->count, kthLine->score);

    int vacant = kthTieCount;
    int idx = kthIdx - 1;
    while (idx >= 0) {
        const GroupedScoredLine* line = lines[idx];
        double score = line->k * sweepLinePos + line->b;
        if (std::abs(score - kthScore) > epsilon) {
            break;
        }

        updatePGroupsBaseCounts<false>(numPGroups, line->pGroupsCounts, curPGroupsBaseCounts);
        curBaseUtility -= line->score * line->count;

        const auto maskedGroups = line->maskedGroups;
        int count = line->count;
        std::copy(maskedGroups, maskedGroups + count, std::back_inserter(tiesGroups));
        std::fill_n(std::back_inserter(tiesScore), count, line->score);

        vacant += line->count;
        idx -= 1;
    }

    idx = kthIdx + 1;
    int count = lines.size();
    while (idx < count) {
        const GroupedScoredLine* line = lines[idx];
        double score = line->k * sweepLinePos + line->b;
        if (std::abs(score - kthScore) > epsilon) {
            break;
        }

        const auto maskedGroups = line->maskedGroups;
        int count = line->count;
        std::copy(maskedGroups, maskedGroups + count, std::back_inserter(tiesGroups));
        std::fill_n(std::back_inserter(tiesScore), count, line->score);

        idx += 1;
    }

    auto [fair, tiesUtility] = FairTopK::searchFairSelectionWithLargestUtility(vacant, pGroupsVec, pGroupsBounds,
        tiesGroups, tiesScore, curPGroupsBaseCounts);

    return { fair, curBaseUtility + tiesUtility };
}

template <FairTopK::Optimization opt>
bool solve(const std::vector<Eigen::VectorXd> &points, const std::vector<FairTopK::Groups>& groups, int k, 
    FairTopK::GroupsMask pGroups, const std::vector<std::pair<int , int> >& pGroupsBounds, double margin, 
    Eigen::VectorXd& weights) {
    constexpr double epsilon = 1e-8;

    double sweepLower = std::max({ 0.0, weights(0) - margin, 1.0 - weights(1) - margin });
    double sweepUpper = std::min({ 1.0, weights(0) + margin, 1.0 - weights(1) + margin });

    double inputPos = weights(0);
    Eigen::VectorXd refWeights = weights;

    constexpr bool optUtility = (opt == FairTopK::Optimization::Utility || opt == FairTopK::Optimization::StableUtility);

    using Line = std::conditional<optUtility, GroupedScoredLine, GroupedLine>::type;

    bool fair = false;
    double optPos = std::numeric_limits<double>::lowest();
    double maxUtility = std::numeric_limits<double>::lowest();

    auto pGroupsVec = FairTopK::getPGroupsVec(pGroups); 
    int numPGroups = pGroupsBounds.size();

    int pointCount = points.size();

    std::vector<std::tuple<double, double, int> > rawLines;
    rawLines.reserve(pointCount);

    for (int i = 0; i < pointCount; i++) {
        const auto &point = points[i];
        rawLines.emplace_back(point(0) - point(1), point(1), i);
    }

    std::sort(rawLines.begin(), rawLines.end(),
        [sweepLower](const auto& l1, const auto& l2) { 
            double k1 = std::get<0>(l1);
            double b1 = std::get<1>(l1);
            double k2 = std::get<0>(l2);
            double b2 = std::get<1>(l2);

            double s1 = k1 * sweepLower + b1;
            double s2 = k2 * sweepLower + b2;

            return s1 != s2 ? (s1 > s2) : (k1 != k2 ? (k1 > k2) : (b1 > b2));
    });

    FairTopK::MemoryArena<Line> linePool(pointCount);
    FairTopK::MemoryArena<int> pgcPool(pointCount * numPGroups);
    FairTopK::MemoryArena<FairTopK::Groups> groupsPool(pointCount);
    std::vector<Line *> lines;
    lines.reserve(pointCount);

    {
        auto [k, b, idx] = rawLines[0];
        FairTopK::Groups candidateGroups = groups[idx];
        int *pGroupsCounts = pgcPool.Alloc(numPGroups, std::true_type{});
        FairTopK::updatePGroupsCounts<1>(numPGroups, candidateGroups, pGroupsVec, pGroupsCounts);
        if constexpr (optUtility) {
            lines.emplace_back(linePool.Alloc(1, std::true_type{}, k, b, 0, 1, pGroupsCounts, nullptr, 
                weights.dot(points[idx])));
        }
        else {
            lines.emplace_back(linePool.Alloc(1, std::true_type{}, k, b, 0, 1, pGroupsCounts, nullptr));
        }
    }

    for (int i = 1; i < pointCount; i++) {
        auto [k, b, idx] = rawLines[i];
        auto preLine = lines.back();
        FairTopK::Groups candidateGroups = groups[idx];

        if (k == preLine->k && b == preLine->b) {
            preLine->count += 1;
            FairTopK::updatePGroupsCounts<1>(numPGroups, candidateGroups, pGroupsVec, preLine->pGroupsCounts);
        }
        else {
            int *pGroupsCounts = pgcPool.Alloc(numPGroups, std::true_type{});
            FairTopK::updatePGroupsCounts<1>(numPGroups, candidateGroups, pGroupsVec, pGroupsCounts);
            if constexpr (optUtility) {
                lines.emplace_back(linePool.Alloc(1, std::true_type{}, k, b, preLine->rank + 1, 1, pGroupsCounts, nullptr, 
                    weights.dot(points[idx])));
            }
            else {
                lines.push_back(linePool.Alloc(1, std::true_type{}, k, b, preLine->rank + 1, 1, pGroupsCounts, nullptr));
            }
        }
    }

    int lineCount = lines.size();

    {
        int pointIdx = 0;
        for (int i = 0; i < lineCount; i++) {
            auto &line = lines[i];
            int count = line->count;
            FairTopK::Groups *maskedProupsPerLine = groupsPool.Alloc(count);
            for (int i = 0; i < count; i++) {
                int idx = std::get<2>(rawLines[pointIdx + i]);
                maskedProupsPerLine[i] = (groups[idx] & pGroups);
            }
            line->maskedGroups = maskedProupsPerLine;
            pointIdx += count;
        }
    }

    int kthIdx = 0;
    int kthTieCount = 0;
    {
        int count = lines[0]->count;
        while (count < k) {
            count += lines[++kthIdx]->count;
        }
        kthTieCount = k - (count - lines[kthIdx]->count); 
    }

    std::array<int, FairTopK::maxNumGroups> pGroupsBaseCounts{};
    for (int i = 0; i < kthIdx; i++) {
        updatePGroupsBaseCounts<true>(numPGroups, lines[i]->pGroupsCounts, pGroupsBaseCounts);
    }

    double baseUtility = 0.0;
    if constexpr (optUtility) {
        for (int i = 0; i < kthIdx; i++) {
            const auto &line = lines[i];
            baseUtility += line->score * line->count;
        }
    }

    auto checkFairness = [&lines, k, inputPos, &kthIdx, &kthTieCount, &pGroupsBaseCounts, &pGroupsVec, &pGroupsBounds,
                          &baseUtility, &maxUtility, &optPos](double pos) {
        bool fair = false;
        if constexpr (optUtility) {
            auto [isFair, utility] = tryObtainFairSelection(lines, k, pos, kthIdx, kthTieCount, 
                pGroupsBaseCounts, pGroupsVec, pGroupsBounds, baseUtility, epsilon);
            fair = isFair;
                
            if (fair) {
                if (utility > maxUtility) {
                    optPos = pos;
                    maxUtility = utility;
                }
            }
        }
        else if constexpr (opt == FairTopK::Optimization::WeightsDifference) {
            fair = isFair(lines, k, pos, kthIdx, kthTieCount, pGroupsBaseCounts, pGroupsVec, pGroupsBounds, epsilon);
            if (fair) {
                if (std::abs((pos - inputPos) < std::abs(optPos - inputPos))) {
                    optPos = pos;
                }
            }
        }
        else {
            fair = isFair(lines, k, pos, kthIdx, kthTieCount, pGroupsBaseCounts, pGroupsVec, pGroupsBounds, epsilon);
        }
        return fair;
    };

    {
        bool isFair = checkFairness(sweepLower);
        if constexpr (opt != FairTopK::Optimization::None) {
            if (isFair) fair = true;
        }
        else {
            if (isFair) {
                weights(0) = sweepLower;
                weights(1) = 1.0 - sweepLower;
                return true;
            }
        }
    }

    auto compare = [](const auto& intersect0, const auto& intersect1) noexcept {
        return std::get<0>(intersect0) > std::get<0>(intersect1);
    };

    using IntersectTuple = std::tuple<double, Line*, Line*>;
    using IntPriorityQueue = std::priority_queue<IntersectTuple, std::vector<IntersectTuple>, decltype(compare)>;
    std::vector<IntersectTuple> queueContainer;
    queueContainer.reserve(2 * lineCount);

    IntPriorityQueue intersectQueue(compare, std::move(queueContainer));

    for (int i = 0; i < lineCount - 1; i++) {
        double intersect = computeIntersect(*lines[i], *lines[i + 1], sweepUpper);
        if (!std::isinf(intersect)) {
            intersectQueue.emplace(intersect, lines[i], lines[i + 1]);
        }
    }

    auto insertNewOrderingChange = [&intersectQueue, sweepUpper](Line* front, Line* back) {
        double intersect = computeIntersect(*front, *back, sweepUpper);
        if (!std::isinf(intersect)) {
            intersectQueue.emplace(intersect, front, back);
        }
    };

    while (!intersectQueue.empty()) {
        auto [intersect, frontLine, backLine] = intersectQueue.top();
        intersectQueue.pop();

        int frontRank = frontLine->rank;
        int backRank = backLine->rank;

        if (backRank < frontRank) {
            continue;
        }

        std::swap(lines[frontLine->rank], lines[backLine->rank]);
        std::swap(frontLine->rank, backLine->rank);

        if (frontRank <= kthIdx && backRank > kthIdx) {
            int kBaseCount = k - kthTieCount;
            if (frontRank != kthIdx) {
                updatePGroupsBaseCounts<false>(numPGroups, frontLine->pGroupsCounts, pGroupsBaseCounts);
                updatePGroupsBaseCounts<true>(numPGroups, backLine->pGroupsCounts, pGroupsBaseCounts);

                if constexpr (optUtility) {
                    baseUtility -= frontLine->count * frontLine->score;
                    baseUtility += backLine->count * backLine->score;
                }

                kBaseCount -= frontLine->count;
                kBaseCount += backLine->count;
            }
            kthTieCount = k - kBaseCount;
            
            int count = kBaseCount + lines[kthIdx]->count;
            if (kBaseCount >= k) {
                int newKthIdx = kthIdx - 1;
                count = kBaseCount;
                while (newKthIdx >= 0 && count >= k) {
                    count -= lines[newKthIdx--]->count;
                }
                newKthIdx += 1;
                
                for (int i = newKthIdx; i < kthIdx; i++)
                    updatePGroupsBaseCounts<false>(numPGroups, lines[i]->pGroupsCounts, pGroupsBaseCounts);

                if constexpr (optUtility) {
                    for (int i = newKthIdx; i < kthIdx; i++) {
                        const auto& line = lines[i];
                        baseUtility -= line->count * line->score;
                    }
                }

                kthIdx = newKthIdx;
                kthTieCount = k - count;
            }
            else if (count < k) {
                int newKthIdx = kthIdx;
                do {
                    count += lines[++newKthIdx]->count;
                } while (count < k);

                for (int i = kthIdx; i < newKthIdx; i++)
                    updatePGroupsBaseCounts<true>(numPGroups, lines[i]->pGroupsCounts, pGroupsBaseCounts);

                if constexpr (optUtility) {
                    for (int i = kthIdx; i < newKthIdx; i++) {
                        const auto& line = lines[i];
                        baseUtility += line->count * line->score;
                    }
                }

                kthIdx = newKthIdx;
                kthTieCount = k - (count - lines[kthIdx]->count);
            }

            bool isFair = checkFairness(intersect);
            if constexpr (opt != FairTopK::Optimization::None) {
                if (isFair) fair = true;
            }
            else {
                if (isFair) {
                    weights(0) = intersect;
                    weights(1) = 1.0 - intersect;
                    return true;
                }
            }
        }
        else if (backRank == kthIdx) {
            int kBaseCount = k - kthTieCount;

            updatePGroupsBaseCounts<false>(numPGroups, frontLine->pGroupsCounts, pGroupsBaseCounts);
            updatePGroupsBaseCounts<true>(numPGroups, backLine->pGroupsCounts, pGroupsBaseCounts);

            if constexpr (optUtility) {
                baseUtility -= frontLine->count * frontLine->score;
                baseUtility += backLine->count * backLine->score;
            }

            kBaseCount -= frontLine->count;
            kBaseCount += backLine->count;

            kthTieCount = k - kBaseCount;

            if (kBaseCount >= k) {
                int newKthIdx = kthIdx - 1;
                int count = kBaseCount;
                while (newKthIdx >= 0 && count >= k) {
                    count -= lines[newKthIdx--]->count;
                }
                newKthIdx += 1;

                for (int i = newKthIdx; i < kthIdx; i++)
                    updatePGroupsBaseCounts<false>(numPGroups, lines[i]->pGroupsCounts, pGroupsBaseCounts);

                if constexpr (optUtility) {
                    for (int i = newKthIdx; i < kthIdx; i++) {
                        const auto& line = lines[i];
                        baseUtility -= line->count * line->score;
                    }
                }

                kthIdx = newKthIdx;
                kthTieCount = k - count;
            }

            bool isFair = checkFairness(intersect);
            if constexpr (opt != FairTopK::Optimization::None) {
                if (isFair) fair = true;
            }
            else {
                if (isFair) {
                    weights(0) = intersect;
                    weights(1) = 1.0 - intersect;
                    return true;
                }
            }
        }

        if (frontRank > 0) {
            insertNewOrderingChange(lines[frontRank - 1], backLine);
        }
        if (backRank < lineCount - 1) {
            insertNewOrderingChange(frontLine, lines[backRank + 1]);
        }

        if (backRank - frontRank > 1) {
            insertNewOrderingChange(backLine, lines[frontRank + 1]);
            insertNewOrderingChange(lines[backRank - 1], frontLine);
        }
    }

    {
        bool isFair = checkFairness(sweepUpper);
        if constexpr (opt != FairTopK::Optimization::None) {
            if (isFair) fair = true;
        }
        else {
            if (isFair) {
                weights(0) = sweepUpper;
                weights(1) = 1.0 - sweepUpper;
                return true;
            }
        }
    }

    if (fair) {
        weights(0) = optPos;
        weights(1) = 1.0 - optPos;

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
