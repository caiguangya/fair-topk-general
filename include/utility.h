/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#ifndef FAIR_TOPK_UTILITY_H
#define FAIR_TOPK_UTILITY_H

#include <vector>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <climits>

#include <boost/predef.h>
#include <boost/container/small_vector.hpp>

#include <Eigen/Dense>

#if (BOOST_COMP_GNUC || BOOST_COMP_CLANG)
#define FAIRTOPK_ALWAYS_INLINE __attribute__((always_inline))
#elif BOOST_COMP_MSVC
#define FAIRTOPK_ALWAYS_INLINE __forceinline
#else
#define FAIRTOPK_ALWAYS_INLINE
#endif

namespace FairTopK {

enum class Optimization : unsigned int { None = 0, Utility = 1, WeightsDifference = 2, StableUtility = 3, NumOptions = 4 };

using Groups = std::uint64_t;
using GroupsMask = std::uint64_t;
static constexpr std::size_t maxNumGroups = sizeof(Groups) * CHAR_BIT;

inline FAIRTOPK_ALWAYS_INLINE GroupsMask getGroupsMask(int group) noexcept {
    return GroupsMask(1) << group;
}

template <std::size_t numGroups = maxNumGroups> requires (numGroups <= maxNumGroups)
inline boost::container::small_vector<int, numGroups> getPGroupsVec(GroupsMask pGroups) {
    boost::container::small_vector<int, numGroups> pGroupsVec;
    GroupsMask pGroupsBitmask = pGroups;

    int pGroup = 0;
    while (pGroupsBitmask != 0) {
        if ((pGroupsBitmask & GroupsMask(1)) != 0)
            pGroupsVec.push_back(pGroup); 

        pGroup += 1;
        pGroupsBitmask = pGroupsBitmask >> 1;
    }
    
    return pGroupsVec;
}

template <typename T>
concept IntSubscriptable = requires(T array) {
    { array[0] } -> std::convertible_to<int>;
};

template <int delta, std::size_t numGroups = maxNumGroups, IntSubscriptable ArrayType> requires (numGroups <= maxNumGroups)
inline void updatePGroupsCounts(int numPGroups, Groups groups, 
    const boost::container::small_vector<int, numGroups>& pGroupsVec,
    ArrayType& pGroupsCounts) {
    for (int i = 0; i < numPGroups; i++) {
        GroupsMask mask = getGroupsMask(pGroupsVec[i]);
        if ((groups & mask) != 0)
            pGroupsCounts[i] += delta;
    }
}

bool checkFairness(const std::vector<Eigen::VectorXd> &points, const std::vector<Groups>& groups, 
    const Eigen::VectorXd& weights, int k, int pGroup, int pGroupLowerBound, int pGroupUpperBound, double epsilon = 1e-8);

bool checkFairness(const std::vector<Eigen::VectorXd> &points, const std::vector<Groups>& groups, 
    const Eigen::VectorXd& weights, int k, GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, 
    double epsilon = 1e-8);

Eigen::VectorXd getRandomWeightVector(int dimension, std::default_random_engine& rand);
std::vector<Eigen::VectorXd> getRandomWeightVectors(int count, int dimension);
std::vector<Eigen::VectorXd> getRandomWeightVectors(int count, const std::vector<Eigen::VectorXd> &points, 
    const std::vector<Groups>& groups, int k, GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, 
    bool pessimistic);

template <int d>
struct Plane {
    using NormalVector = Eigen::Matrix<double, d, 1>;
    NormalVector normal; //Unnormalized
    double constant;
};

struct InputParams {
    int k = 0;
    double margin = 0.0;
    int threadCount = 0;
    int sampleCount = 0;
    bool uniformSampling = false;
    bool pessSampling = false;
    bool runtime = false;
    bool quality = false;
    std::string solver = "gurobi";
    Optimization opt = Optimization::None;
    std::vector<std::pair<int, int> > pGroupsBounds;

    InputParams() = default;
    InputParams(InputParams&) = default;
    InputParams(InputParams&&) = default;
    InputParams& operator=(InputParams&) = default;
    InputParams& operator=(InputParams&&) = default;
};

bool checkProtectedGroupsBoundsCount(GroupsMask pGroups, const std::vector<std::pair<int, int> > &pGroupsBounds);

std::pair<std::string, InputParams> parseCommandLine(int argc, char* argv[]);

template <int dimension>
std::vector<Eigen::Matrix<double, dimension - 1, 1> > computeExtremePoints(const Eigen::VectorXd& weights, 
    double margin, double epsilon) {
    using VectorDd = Eigen::Matrix<double, dimension - 1, 1>;

    std::vector<VectorDd> points;
    points.reserve(1 << dimension);

    VectorDd lbs, ubs;
    for (int i = 0; i < dimension - 1; i++) {
        double lb = std::max(0.0, weights(i) - margin);
        double ub = std::min(1.0, weights(i) + margin);

        lbs(i) = lb;
        ubs(i) = ub; 
    }
    double lastLb = std::max(0.0, weights(dimension - 1) - margin);
    double lastUb = std::min(1.0, weights(dimension - 1) + margin); 

    VectorDd point;
    constexpr int count = 1 << (dimension - 1);
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < dimension - 1; j++) {
            unsigned int mask = 1 << j;
            point(j) = ((i & mask) == 0) ? lbs(j) : ubs(j); 
        }

        double lastCoord = 1.0 - point.sum();
        if (lastLb - lastCoord <= epsilon && lastCoord - lastUb <= epsilon) {
            points.push_back(point);
        }

        for (int j = 0; j < dimension - 1; j++) {
            double lb = lbs(j);
            double ub = ubs(j);

            double preCoord = point(j);
            point(j) = 0.0;

            double sum = point.sum();
            unsigned int mask = 1 << j;
            double coord = ((i & mask) == 0) ? 1.0 - lastLb - sum : 1.0 - lastUb - sum;

            if (lb - coord <= epsilon && coord - ub <= epsilon) {
                point(j) = coord; 
                points.push_back(point);
            }

            point(j) = preCoord;
        }
    }

    return points;
}

template <int dimension>
bool testIntersection(const Plane<dimension>& plane, const std::vector<Eigen::Matrix<double, dimension, 1> >& extremePoints,
    double epsilon) {
    if (extremePoints.empty()) return false;

    enum class IntersectionStatus : int { Intersect = 0, Positive = 1, Negative = 2 };
    
    IntersectionStatus status = IntersectionStatus::Intersect;
    for (const auto& point : extremePoints) {
        double diff = plane.normal.dot(point) - plane.constant;

        if (diff > epsilon && status != IntersectionStatus::Negative) {
            status = IntersectionStatus::Positive;
        }
        else if (diff < -epsilon && status != IntersectionStatus::Positive) {
            status = IntersectionStatus::Negative;
        }
        else {
            status = IntersectionStatus::Intersect;
            break;
        }
    }

    return (int)status <= 0;
}

}

#endif
