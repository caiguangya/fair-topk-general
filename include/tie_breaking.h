/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef FAIR_TOPK_TIEBREAKING_H
#define FAIR_TOPK_TIEBREAKING_H

#include <vector>
#include <stack>
#include <limits>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <tuple>

#include "utility.h"

namespace FairTopK {

namespace Detail {
inline bool searchFairSelectionSingleProtected(int vacant, GroupsMask pGroupMask, 
    int pGroupLowerBound, int pGroupUpperBound, int pGroupBaseCount,
    const std::vector<Groups>& ties);

template <std::size_t numGroups>
inline bool searchFairSelectionSingleGroups(int vacant, const boost::container::small_vector<int, numGroups>& pGroupsVec,
    const std::vector<std::pair<int , int> >& pGroupsBounds, const std::array<int, numGroups>& pGroupsCounts,
    Groups groups, int groupsCount);

template <std::size_t numGroups>
bool fastFairnessNACheck(int vacant, int tiesCount, const boost::container::small_vector<int, numGroups>& pGroupsVec,
    const std::vector<std::pair<int , int> >& pGroupsBounds, const std::array<int, numGroups>& pGroupsCounts,
    const std::vector<std::pair<Groups, int> >& groupsCounts);

template <std::size_t numGroups, class UtililtyCalculator = std::nullptr_t>
bool searchFairSelectionBacktrack(int vacant, const boost::container::small_vector<int, numGroups>& pGroupsVec,
    const std::vector<std::pair<int , int> >& pGroupsBounds, const std::vector<std::pair<Groups, int> >& groupsCounts,
    std::array<int, numGroups>& pGroupsCounts,
    UtililtyCalculator&& utililtyCalculator = nullptr);
}

template <std::size_t numGroups>
bool searchFairSelection(int vacant, const boost::container::small_vector<int, numGroups>& pGroupsVec,
    const std::vector<std::pair<int , int> >& pGroupsBounds, std::array<int, numGroups>& pGroupsCounts,
    std::vector<Groups>& ties) {

    int numPGroups = pGroupsBounds.size();
    
    if (vacant <= 1) {
        if (vacant <= 0) return false;

        for (auto candidateGroups : ties) {
            bool isFair = true;

            for (int i = 0; i < numPGroups; i++) {
                auto pGroupMask = getGroupsMask(pGroupsVec[i]);

                int isProtected = ((candidateGroups & pGroupMask) != 0);
                int pGroupCount = pGroupsCounts[i] + isProtected;

                auto [pGroupLowerBound, pGroupUpperBound] = pGroupsBounds[i];
                if (pGroupCount < pGroupLowerBound || pGroupCount > pGroupUpperBound) {
                    isFair = false;
                    break;
                }
            }

            if (isFair) return true;
        }

        return false;
    }

    if (numPGroups <= 1) {
        auto pGroupMask = getGroupsMask(pGroupsVec[0]);
        int pGroupBaseCount = pGroupsCounts[0];
        auto [pGroupLowerBound, pGroupUpperBound] = pGroupsBounds[0];

        return Detail::searchFairSelectionSingleProtected(vacant, pGroupMask, pGroupLowerBound, 
            pGroupUpperBound, pGroupBaseCount, ties);
    }

    std::sort(ties.begin(), ties.end());

    int count = ties.size();

    std::vector<std::pair<Groups, int> > groupsCounts;
    groupsCounts.reserve(count);

    groupsCounts.emplace_back(ties[0], 1);

    for (int i = 1; i < count; i++) {
        Groups groups = ties[i];

        if (groups == ties[i - 1]) {
            groupsCounts.back().second += 1;
        }
        else {
            groupsCounts.emplace_back(groups, 1);
        }
    }

    if (groupsCounts.size() <= 1) {
        auto [groups, groupsCount] = groupsCounts[0];
        
        return Detail::searchFairSelectionSingleGroups(vacant, pGroupsVec, pGroupsBounds, pGroupsCounts,
            groups, groupsCount);
    }

    if (Detail::fastFairnessNACheck(vacant, count, pGroupsVec, pGroupsBounds, pGroupsCounts, groupsCounts)) {
        return false;
    }

    return Detail::searchFairSelectionBacktrack(vacant, pGroupsVec, pGroupsBounds, groupsCounts, pGroupsCounts);
}

template <std::size_t numGroups>
std::pair<bool, double> searchFairSelectionWithLargestUtility(int vacant,
    const boost::container::small_vector<int, numGroups>& pGroupsVec, const std::vector<std::pair<int, int> >& pGroupsBounds,
    const std::vector<Groups>& tiesGroups, const std::vector<double>& tiesScore,
    std::array<int, numGroups>& pGroupsCounts) {

    int numPGroups = pGroupsBounds.size();
    int count = tiesGroups.size();

    if (vacant <= 1) {
        if (vacant <= 0) return { false, std::numeric_limits<double>::lowest() };

        bool fair = false;
        double maxUtility = std::numeric_limits<double>::lowest();
        for (int i = 0; i < count; i++) {
            bool isFair = true;

            Groups candidateGroups = tiesGroups[i];
            for (int j = 0; j < numPGroups; j++) {
                auto pGroupMask = getGroupsMask(pGroupsVec[j]);

                int isProtected = ((candidateGroups & pGroupMask) != 0);
                int pGroupCount = pGroupsCounts[j] + isProtected;

                auto [pGroupLowerBound, pGroupUpperBound] = pGroupsBounds[j];
                if (pGroupCount < pGroupLowerBound || pGroupCount > pGroupUpperBound) {
                    isFair = false;
                    break;
                }
            }

            if (isFair) {
                fair = true;
                maxUtility = std::max(maxUtility, tiesScore[i]);
            }
        }

        return { fair, maxUtility };
    }

    if (numPGroups <= 1) {
        auto pGroupMask = getGroupsMask(pGroupsVec[0]);
        int pGroupBaseCount = pGroupsCounts[0];
        auto [pGroupLowerBound, pGroupUpperBound] = pGroupsBounds[0];
        
        bool fair = Detail::searchFairSelectionSingleProtected(vacant, pGroupMask, pGroupLowerBound, 
            pGroupUpperBound, pGroupBaseCount, tiesGroups);

        double maxUtility = std::numeric_limits<double>::lowest();
        if (fair) {
            std::vector<double> protectedScores;
            std::vector<double> otherScores;
            protectedScores.reserve(count / 2);
            otherScores.reserve(count / 2);

            for (int i = 0; i < count; i++) {
                Groups groups = tiesGroups[i];
                double score = tiesScore[i];
                if ((groups & pGroupMask) != 0) {
                    protectedScores.push_back(score);
                }
                else {
                    otherScores.push_back(score);
                }
            }

            int protectedMaxCount = std::min(vacant, (int)protectedScores.size());
            int otherMaxCount = std::min(vacant, (int)otherScores.size());

            std::partial_sort(protectedScores.begin(), protectedScores.begin() + protectedMaxCount, protectedScores.end(),
                std::greater<double>());
            std::partial_sort(otherScores.begin(), otherScores.begin() + otherMaxCount, otherScores.end(),
                std::greater<double>());

            maxUtility = 0.0;
            int pGroupCount = pGroupBaseCount;
            int pScoreIdx = 0, oScoreIdx = 0;
            for (int i = 0; i < vacant; i++) {
                double pScore = (pScoreIdx < protectedMaxCount) ? protectedScores[pScoreIdx] :
                                                                  std::numeric_limits<double>::lowest();
                double oScore = (oScoreIdx < otherMaxCount) ? otherScores[oScoreIdx] :
                                                              std::numeric_limits<double>::lowest();
                if (pGroupCount < pGroupLowerBound) {
                    maxUtility += pScore;
                    pScoreIdx += 1;

                    pGroupCount += 1;
                }
                else if (pGroupCount >= pGroupUpperBound) {
                    maxUtility += oScore;
                    oScoreIdx += 1;
                }
                else {
                    if (pScore > oScore) {
                        maxUtility += pScore;
                        pScoreIdx += 1;
                        
                        pGroupCount += 1;
                    }
                    else {
                        maxUtility += oScore;
                        oScoreIdx += 1;
                    }
                }
            }
        }

        return { fair, maxUtility };
    }

    std::vector<std::pair<Groups, double> > ties;
    ties.reserve(count);
    for (int i = 0; i < count; i++) {
        ties.emplace_back(tiesGroups[i], tiesScore[i]);
    }

    std::sort(ties.begin(), ties.end(), [](const auto& t1, const auto& t2) { return t1.first < t2.first; });

    std::vector<std::vector<double> > perDistGroupAccumScores;

    std::vector<std::pair<Groups, int> > groupsCounts;
    groupsCounts.reserve(count);

    {
        auto [groups, score] = ties[0];
        groupsCounts.emplace_back(groups, 1);
        
        std::vector<double> scores;
        scores.reserve(count / numGroups);
        scores.push_back(score);
        perDistGroupAccumScores.push_back(std::move(scores));
    }

    for (int i = 1; i < count; i++) {
        auto [groups, score] = ties[i];

        if (groups == ties[i - 1].first) {
            groupsCounts.back().second += 1;
            perDistGroupAccumScores.back().push_back(score);
        }
        else {
            groupsCounts.emplace_back(groups, 1);
            std::vector<double> scores;
            scores.reserve(count / numGroups);
            scores.push_back(score);
            perDistGroupAccumScores.push_back(std::move(scores));
        }
    }

    for (auto& accumScores : perDistGroupAccumScores) {
        int effSize = std::min(vacant, (int)accumScores.size());
        std::partial_sort(accumScores.begin(), accumScores.begin() + effSize, accumScores.end(),
            std::greater<double>());

        for (int i = 1; i < effSize; i++) {
            accumScores[i] += accumScores[i - 1];
        }
    }

    if (groupsCounts.size() <= 1) {
        auto [groups, groupsCount] = groupsCounts[0];
        
        bool fair = Detail::searchFairSelectionSingleGroups(vacant, pGroupsVec, pGroupsBounds, pGroupsCounts,
            groups, groupsCount);

        double maxUtility = std::numeric_limits<double>::lowest();
        if (fair) {
            maxUtility = perDistGroupAccumScores[0][vacant - 1];
        }
        
        return { fair, maxUtility };
    }

    if (Detail::fastFairnessNACheck(vacant, count, pGroupsVec, pGroupsBounds, pGroupsCounts, groupsCounts)) {
        return { false, std::numeric_limits<double>::lowest() };
    }

    double maxUtility = std::numeric_limits<double>::lowest();
    auto calculateUtility = [&maxUtility, &perDistGroupAccumScores](const std::vector<int>& states) {
        double utility = 0.0;
        int numDistGroupMasks = states.size();
        for (int i = 0; i < numDistGroupMasks; i++) {
            int groupsCount = states[i];
            if (groupsCount > 0) {
                utility += perDistGroupAccumScores[i][groupsCount - 1];
            }
        }

        if (utility > maxUtility) maxUtility = utility;
    };

    bool fair = Detail::searchFairSelectionBacktrack(vacant, pGroupsVec, pGroupsBounds, 
        groupsCounts, pGroupsCounts, calculateUtility);

    return { fair, maxUtility };
}

template <std::size_t numGroups>
bool obtainFairSelectionWithLargestUtility(int vacant,
    const boost::container::small_vector<int, numGroups>& pGroupsVec, const std::vector<std::pair<int, int> >& pGroupsBounds,
    const std::vector<Groups>& tiesGroups, const std::vector<double>& tiesScore, const std::vector<int>& tiesIndices, 
    std::array<int, numGroups>& pGroupsCounts, std::vector<int>& selectedIndices) {

    int numPGroups = pGroupsBounds.size();
    int count = tiesGroups.size();

    if (vacant <= 1) {
        if (vacant <= 0) return false;

        bool fair = false;
        double maxUtility = std::numeric_limits<double>::lowest();
        int selectedIdx = -1;
        for (int i = 0; i < count; i++) {
            bool isFair = true;

            Groups candidateGroups = tiesGroups[i];
            for (int j = 0; j < numPGroups; j++) {
                auto pGroupMask = getGroupsMask(pGroupsVec[j]);

                int isProtected = ((candidateGroups & pGroupMask) != 0);
                int pGroupCount = pGroupsCounts[j] + isProtected;

                auto [pGroupLowerBound, pGroupUpperBound] = pGroupsBounds[j];
                if (pGroupCount < pGroupLowerBound || pGroupCount > pGroupUpperBound) {
                    isFair = false;
                    break;
                }
            }

            if (isFair) {
                fair = true;
                double score = tiesScore[i];
                if (score > maxUtility) {
                    selectedIdx = tiesIndices[i];
                    maxUtility = score;
                }
            }
        }

        if (fair) selectedIndices.push_back(selectedIdx);

        return fair;
    }

    if (numPGroups <= 1) {
        auto pGroupMask = getGroupsMask(pGroupsVec[0]);
        int pGroupBaseCount = pGroupsCounts[0];
        auto [pGroupLowerBound, pGroupUpperBound] = pGroupsBounds[0];
        
        bool fair = Detail::searchFairSelectionSingleProtected(vacant, pGroupMask, pGroupLowerBound, 
            pGroupUpperBound, pGroupBaseCount, tiesGroups);

        if (fair) {
            std::vector<std::pair<double, int> > protectedScoresPair;
            std::vector<std::pair<double, int> > otherScoresPair;
            protectedScoresPair.reserve(count / 2);
            otherScoresPair.reserve(count / 2);

            for (int i = 0; i < count; i++) {
                Groups groups = tiesGroups[i];
                double score = tiesScore[i];
                int idx = tiesIndices[i];
                if ((groups & pGroupMask) != 0) {
                    protectedScoresPair.emplace_back(score, idx);
                }
                else {
                    otherScoresPair.emplace_back(score, idx);
                }
            }

            int protectedMaxCount = std::min(vacant, (int)protectedScoresPair.size());
            int otherMaxCount = std::min(vacant, (int)otherScoresPair.size());

            std::partial_sort(protectedScoresPair.begin(), protectedScoresPair.begin() + protectedMaxCount, protectedScoresPair.end(),
                [](const auto& t0, const auto& t1) { return t0.first > t1.first; });
            std::partial_sort(otherScoresPair.begin(), otherScoresPair.begin() + otherMaxCount, otherScoresPair.end(),
                [](const auto& t0, const auto& t1) { return t0.first > t1.first; });

            constexpr std::pair<double, int> dummy(std::numeric_limits<double>::lowest(), -1);

            int pGroupCount = pGroupBaseCount;
            int pScoreIdx = 0, oScoreIdx = 0;
            for (int i = 0; i < vacant; i++) {
                auto [pScore, pIdx] = (pScoreIdx < protectedMaxCount) ? protectedScoresPair[pScoreIdx] : dummy;
                auto [oScore, oIdx] = (oScoreIdx < otherMaxCount) ? otherScoresPair[oScoreIdx] : dummy;

                if (pGroupCount < pGroupLowerBound) {
                    pScoreIdx += 1;
                    selectedIndices.push_back(pIdx);

                    pGroupCount += 1;
                }
                else if (pGroupCount >= pGroupUpperBound) {
                    oScoreIdx += 1;

                    selectedIndices.push_back(oIdx);
                }
                else {
                    if (pScore > oScore) {
                        pScoreIdx += 1;
                        selectedIndices.push_back(pIdx);

                        pGroupCount += 1;
                    }
                    else {
                        oScoreIdx += 1;
                        selectedIndices.push_back(oIdx);
                    }
                }
            }
        }

        return fair;
    }

    std::vector<std::tuple<Groups, double, int> > ties;
    ties.reserve(count);
    for (int i = 0; i < count; i++) {
        ties.emplace_back(tiesGroups[i], tiesScore[i], tiesIndices[i]);
    }

    std::sort(ties.begin(), ties.end(), [](const auto& t1, const auto& t2) { return std::get<0>(t1) < std::get<0>(t2); });

    std::vector<std::vector<std::pair<double, int> > > perDistGroupAccumScoresPair;

    std::vector<std::pair<Groups, int> > groupsCounts;
    groupsCounts.reserve(count);

    {
        auto [groups, score, idx] = ties[0];
        groupsCounts.emplace_back(groups, 1);
        
        std::vector<std::pair<double, int> > scoresPair;
        scoresPair.reserve(count / numGroups);
        scoresPair.emplace_back(score, idx);
        perDistGroupAccumScoresPair.push_back(std::move(scoresPair));
    }

    for (int i = 1; i < count; i++) {
        auto [groups, score, idx] = ties[i];

        if (groups == std::get<0>(ties[i - 1])) {
            groupsCounts.back().second += 1;
            perDistGroupAccumScoresPair.back().emplace_back(score, idx);
        }
        else {
            groupsCounts.emplace_back(groups, 1);
            std::vector<std::pair<double, int> > scoresPair;
            scoresPair.reserve(count / numGroups);
            scoresPair.emplace_back(score, idx);
            perDistGroupAccumScoresPair.push_back(std::move(scoresPair));
        }
    }

    for (auto& accumScores : perDistGroupAccumScoresPair) {
        int effSize = std::min(vacant, (int)accumScores.size());
        std::partial_sort(accumScores.begin(), accumScores.begin() + effSize, accumScores.end(),
            [](const auto& t0, const auto& t1) { return t0.first > t1.first; });

        for (int i = 1; i < effSize; i++) {
            accumScores[i].first += accumScores[i - 1].first;
        }
    }

    if (groupsCounts.size() <= 1) {
        auto [groups, groupsCount] = groupsCounts[0];
        
        bool fair = Detail::searchFairSelectionSingleGroups(vacant, pGroupsVec, pGroupsBounds, pGroupsCounts,
            groups, groupsCount);

        if (fair) {
            const auto &sortedIndices = perDistGroupAccumScoresPair[0];
            for (int i = 0; i < vacant; i++) {
                selectedIndices.push_back(sortedIndices[i].second);
            }
        }
        
        return fair;
    }

    std::vector<int> maxUtilityStates;
    maxUtilityStates.reserve(groupsCounts.size());

    double maxUtility = std::numeric_limits<double>::lowest();
    auto trackUtility = [&maxUtility, &perDistGroupAccumScoresPair, &maxUtilityStates](const std::vector<int>& states) {
        double utility = 0.0;
        int numDistGroupMasks = states.size();
        for (int i = 0; i < numDistGroupMasks; i++) {
            int groupsCount = states[i];
            if (groupsCount > 0) {
                utility += perDistGroupAccumScoresPair[i][groupsCount - 1].first;
            }
        }

        if (utility > maxUtility) { 
            maxUtility = utility;
            maxUtilityStates = states;
        }
    };

    bool fair = Detail::searchFairSelectionBacktrack(vacant, pGroupsVec, pGroupsBounds, 
        groupsCounts, pGroupsCounts, trackUtility);

    if (fair) {
        int numDistGroupMasks = maxUtilityStates.size();
        for (int i = 0; i < numDistGroupMasks; i++) {
            int groupsCount = maxUtilityStates[i];

            const auto &sortedIndices = perDistGroupAccumScoresPair[i];
            for (int j = 0; j < groupsCount; j++) {
                selectedIndices.push_back(sortedIndices[j].second);
            }
        }
    }

    return fair;
}

namespace Detail {

inline bool searchFairSelectionSingleProtected(int vacant, GroupsMask pGroupMask, 
    int pGroupLowerBound, int pGroupUpperBound, int pGroupBaseCount,
    const std::vector<Groups>& ties) {
    int tieProtected = 0;
    int tieOther = 0;
    for (auto groups : ties) {
        int isProtected = ((groups & pGroupMask) != 0);
        tieProtected += isProtected;
        tieOther += 1 - isProtected;
    }

    int pGroupLowerCount = pGroupBaseCount + std::max(0, vacant - tieOther);
    int pGroupUpperCount = pGroupBaseCount + vacant - std::max(0, vacant - tieProtected);
        
    return std::max(pGroupLowerCount, pGroupLowerBound) <= std::min(pGroupUpperCount, pGroupUpperBound);
}

template <std::size_t numGroups>
inline bool searchFairSelectionSingleGroups(int vacant, const boost::container::small_vector<int, numGroups>& pGroupsVec,
    const std::vector<std::pair<int, int> >& pGroupsBounds, const std::array<int, numGroups>& pGroupsCounts,
    Groups groups, int groupsCount) {
    if (vacant > groupsCount) return false;

    int numPGroups = pGroupsBounds.size();

    bool fair = true;
    for (int i = 0; i < numPGroups; i++) {
        auto [lowerBound, upperBound] = pGroupsBounds[i];

        auto mask = getGroupsMask(pGroupsVec[i]);
        int pGroupCount = pGroupsCounts[i];
        int isProtected = ((groups & mask) != 0);
        pGroupCount += vacant * isProtected;

        if (pGroupCount < lowerBound || pGroupCount > upperBound) {
            fair = false;
            break;
        }
    }

    return fair;
}

template <std::size_t numGroups>
bool fastFairnessNACheck(int vacant, int tiesCount, const boost::container::small_vector<int, numGroups>& pGroupsVec,
    const std::vector<std::pair<int , int> >& pGroupsBounds, const std::array<int, numGroups>& pGroupsCounts,
    const std::vector<std::pair<Groups, int> >& groupsCounts) {
    int numPGroups = pGroupsBounds.size();

    std::array<int, numGroups> pGroupsDeltas{};

    for (auto [groups, groupsCount] : groupsCounts) {
        for (int i = 0; i < numPGroups; i++) {
            auto mask = getGroupsMask(pGroupsVec[i]);
            if ((groups & mask) != 0) 
                pGroupsDeltas[i] += groupsCount;
        }
    }

    for (int i = 0; i < numPGroups; i++) {
        auto [lowerBound, upperBound] = pGroupsBounds[i];
        int pGroupsCount = pGroupsCounts[i];
        if (pGroupsCount < lowerBound) {
            int delta = lowerBound - pGroupsCount;
            int deltaMax = std::min(vacant, pGroupsDeltas[i]);

            if (delta > deltaMax) return true; 
        }
        else if (pGroupsCount <= upperBound) {
            int delta = upperBound - pGroupsCount;
            int deltaPotMax = pGroupsDeltas[i];
            int deltaMin = std::min(vacant - (tiesCount - deltaPotMax), deltaPotMax);

            if (delta < deltaMin) return true;
        }
        else {
            return true;
        }
    }

    return false;
}

template <std::size_t numGroups, class UtililtyCalculator>
bool searchFairSelectionBacktrack(int vacant, const boost::container::small_vector<int, numGroups>& pGroupsVec,
    const std::vector<std::pair<int , int> >& pGroupsBounds, const std::vector<std::pair<Groups, int> >& groupsCounts,
    std::array<int, numGroups>& pGroupsCounts,
    UtililtyCalculator&& utililtyCalculator) {
    int numPGroups = pGroupsBounds.size();
    int numDistGroupMasks = groupsCounts.size();

    std::vector<int> states(numDistGroupMasks, -1);
    std::vector<int> acctBounds(numDistGroupMasks - 1, 0);

    acctBounds[numDistGroupMasks - 2] = groupsCounts[numDistGroupMasks - 1].second;
    for (int i = numDistGroupMasks - 3; i >= 0; i--)
        acctBounds[i] = acctBounds[i + 1] + groupsCounts[i + 1].second;

    std::vector<std::pair<int, int> > stackContainer;
    stackContainer.reserve(numDistGroupMasks);
    std::stack<std::pair<int, int>, std::vector<std::pair<int, int> > > stack(std::move(stackContainer));

    stack.emplace(vacant, 0);

    bool fair = false;
    while (!stack.empty()) {
        auto [remained, idx] = stack.top();

        int preCount = std::max(states[idx], 0);
        auto [groups, bound] = groupsCounts[idx];

        int inc = 0;
        if (idx < numDistGroupMasks - 1)  {
            if (states[idx] < 0) {
                states[idx] = std::max(0, remained - acctBounds[idx]);
                inc = states[idx];
            }
            else {
                states[idx] += 1;
                inc = 1;
            }
            remained -= states[idx];
        }
        else {
            inc = remained;
            states[idx] = remained;
            remained = 0;
        }

        bool outOfBound = states[idx] > bound;

        if (remained < 0 || outOfBound) {
            states[idx] = -1;
            for (int i = 0; i < numPGroups; i++) {
                auto mask = getGroupsMask(pGroupsVec[i]);
                int isProtected = ((groups & mask) != 0);
                pGroupsCounts[i] -= preCount * isProtected;
            }
            stack.pop();
            continue;
        }

        for (int i = 0; i < numPGroups; i++) {
            auto mask = getGroupsMask(pGroupsVec[i]);
            int isProtected = ((groups & mask) != 0);
            pGroupsCounts[i] += inc * isProtected;
        }

        if (remained == 0) {
            bool isFair = true;
            for (int i = 0; i < numPGroups; i++) {
                auto [lowerBound, upperBound] = pGroupsBounds[i];
                int pGroupCount = pGroupsCounts[i];
                if (pGroupCount < lowerBound || pGroupCount > upperBound) {
                    isFair = false;
                    break;
                }
            }
            if (isFair) {
                fair = true;
                if constexpr (std::is_same<UtililtyCalculator, std::nullptr_t>()) {
                    break;
                }
                else {
                    std::forward<UtililtyCalculator>(utililtyCalculator)(states);
                }
            }
            for (int i = 0; i < numPGroups; i++) {
                auto mask = getGroupsMask(pGroupsVec[i]);
                int isProtected = ((groups & mask) != 0);
                pGroupsCounts[i] -= states[idx] * isProtected;
            }
            states[idx] = -1;
            stack.pop();
        }
        else {
            stack.emplace(remained, idx + 1);
        }
    }

    return fair;
}

}

}

#endif