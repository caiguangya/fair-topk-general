/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <algorithm>
#include <random>
#include <utility>
#include <boost/mp11/algorithm.hpp>

#include <Eigen/Dense>
#include <soplex.h>

#include "utility.h"
#include "experiments.h"
#include "data_loader.h"
#include "bsp_tree.h"
#include "weights_difference.h"
#include "tie_breaking.h"
#include "stabilization.h"

template <int dimension>
inline FairTopK::Plane<dimension - 1> getProjectedIntersection(const Eigen::VectorXd& diff) {
    using ProjPlaneNormalVector = FairTopK::Plane<dimension - 1>::NormalVector;
    FairTopK::Plane<dimension - 1> projectedPlane;
    
    projectedPlane.normal = ProjPlaneNormalVector::Map(diff.data());
    projectedPlane.normal -= diff(dimension - 1) * ProjPlaneNormalVector::Ones();
    projectedPlane.constant = -diff(dimension - 1);

    return projectedPlane;
}

template <int projDimension>
bool findValidWeightVector(const std::vector<std::pair<FairTopK::Plane<projDimension>, bool> >& halfSpaces,
    const Eigen::VectorXd& refWeights, double margin, Eigen::VectorXd& weights) {
    using LPConstrsMat = Eigen::Matrix<double, projDimension + 1, -1>;
    using LPVector = Eigen::Matrix<double, projDimension + 1, 1>;
    using ColVector = Eigen::Matrix<double, projDimension, 1>;

    int count = halfSpaces.size();
    int addConstrsCount = 2 * (projDimension + 1);
    LPConstrsMat mat = LPConstrsMat::Zero(projDimension + 1, count + addConstrsCount);
    Eigen::VectorXd rhs(count + addConstrsCount);

    LPVector objCoeffs = LPVector::Zero();
    objCoeffs(projDimension) = -1.0;
    LPVector results;

    for (int i = 0; i < projDimension; i++) {
        double lb = std::max(0.0, refWeights(i) - margin);
        double ub = std::min(1.0, refWeights(i) + margin);
            
        mat(i, 2 * i) = 1.0;
        rhs(2 * i) = ub;
        mat(i, 2 * i + 1) = -1.0;
        rhs(2 * i + 1) = -lb;
    }

    {
        int lastTwoOffset = 2 * projDimension;

        double lb = std::max(0.0, refWeights(projDimension) - margin);
        double ub = std::min(1.0, refWeights(projDimension) + margin);

        mat.col(lastTwoOffset).template head<projDimension>() = -ColVector::Ones();
        rhs(lastTwoOffset) = ub - 1.0;
        mat.col(lastTwoOffset + 1).template head<projDimension>() = ColVector::Ones();
        rhs(lastTwoOffset + 1) = 1.0 - lb;
    }

    int offset = addConstrsCount;
    for (int i = 0; i < count; i++) {
        const auto& [halfSpacePlane, isPositive] = halfSpaces[i];
        int colIdx = offset + i;
        if (isPositive) {
            mat.col(colIdx).template head<projDimension>() = 
                -ColVector::Map(halfSpacePlane.normal.data());
            rhs(colIdx) = -halfSpacePlane.constant;
        }
        else {
            mat.col(colIdx).template head<projDimension>() = 
                ColVector::Map(halfSpacePlane.normal.data());
            rhs(colIdx) = halfSpacePlane.constant;
        }
        mat(projDimension, colIdx) = 1.0;
    }

    double val = sdlp::linprog<projDimension + 1>(objCoeffs, mat, rhs, results);

    if (val >= 0.0) 
        return false;

    results(projDimension) = 1.0 - results.template head<projDimension>().sum();
    weights = Eigen::VectorXd::Map(results.data(), projDimension + 1);
    return true;
}

std::pair<bool, double> tryObtainFairSelection(const std::vector<Eigen::VectorXd> &points, 
    const std::vector<FairTopK::Groups>& groups, const Eigen::VectorXd& weightVector, int k, 
    FairTopK::GroupsMask pGroups, const std::vector<std::pair<int, int> >& pGroupsBounds, 
    const std::vector<double>& scores, double epsilon) {
    std::vector<std::pair<double, int> > pts;
    int count = points.size();

    pts.reserve(count);
    for (int i = 0; i < count; i++) {
        pts.emplace_back(weightVector.dot(points[i]), i);
    }

    std::nth_element(pts.begin(), pts.begin() + (k - 1), pts.end(),
        [](const auto& p0, const auto& p1) { return p0.first > p1.first; });

    double utility = 0.0;
    double kthScore = pts[k - 1].first;

    std::vector<FairTopK::Groups> tiesGroups; 
    std::vector<double> tiesScore;

    int vacant = 0;

    int numPGroups = pGroupsBounds.size();
    auto pGroupsVec = FairTopK::getPGroupsVec(pGroups);
    std::array<int, FairTopK::maxNumGroups> pGroupsBaseCounts{};

    for (int i = 0; i < k; i++) {
        const auto [score, idx] = pts[i];
        FairTopK::Groups candiateGroups = groups[idx];
        if (score - kthScore > epsilon) {
            utility += scores[idx];
            if ((candiateGroups & pGroups) != 0)
                FairTopK::updatePGroupsCounts<1>(numPGroups, candiateGroups, pGroupsVec, pGroupsBaseCounts);
        }
        else {
            vacant += 1;

            tiesGroups.push_back(candiateGroups & pGroups);
            tiesScore.push_back(scores[idx]);
        }
    }

    for (int i = k; i < count; i++) {
        const auto [score, idx] = pts[i];
        if (kthScore - score <= epsilon) {
            tiesGroups.push_back(groups[idx] & pGroups);
            tiesScore.push_back(scores[idx]);
        }
    }

    auto [fair, tiesUtility] = FairTopK::searchFairSelectionWithLargestUtility(vacant, pGroupsVec, pGroupsBounds, 
        tiesGroups, tiesScore, pGroupsBaseCounts);

    utility += tiesUtility;

    return { fair, utility };
}

template <int projDimension>
void setupWeightsDiffLPSolver(const std::vector<std::pair<FairTopK::Plane<projDimension>, bool> >& halfSpaces,
    const FairTopK::WeightsDiffLPAuxVector<projDimension + 1>& weightsDiffLPAuxVector, soplex::SoPlex& soplexSolver) {
    constexpr int dimension = projDimension + 1;

    soplexSolver.setIntParam(soplex::SoPlex::OBJSENSE, soplex::SoPlex::OBJSENSE_MINIMIZE);
    soplexSolver.setRealParam(soplex::SoPlex::RealParam::FEASTOL, FairTopK::soplexEpsilon);
    soplexSolver.setRealParam(soplex::SoPlex::RealParam::FPFEASTOL, FairTopK::soplexEpsilon);

    soplexSolver.spxout.setVerbosity(soplex::SPxOut::ERROR);

    soplex::DSVector dummyCol(0);
    
    FairTopK::setUpWeightsDiffLPWeightVars<dimension>(weightsDiffLPAuxVector, dummyCol, soplexSolver);

    for (int i = 0; i < dimension; i++) {
        soplexSolver.addColReal(soplex::LPCol(1.0, dummyCol, soplex::infinity, 0.0));
    }

    constexpr std::array<int, projDimension> indexArray = 
        FairTopK::getIndexSequence(std::make_integer_sequence<int, projDimension>{});

    soplex::DSVector row(projDimension + dimension);

    FairTopK::setUpWeightsDiffLPConstrs<projDimension + dimension, dimension>(weightsDiffLPAuxVector, row, soplexSolver);

    for (const auto &[halfSpacePlane, isPositive] : halfSpaces) {
        row.add(projDimension, indexArray.data(), halfSpacePlane.normal.data());
        if (isPositive) {
            soplexSolver.addRowReal(soplex::LPRow(row, soplex::LPRow::GREATER_EQUAL, halfSpacePlane.constant));
        }
        else {
            soplexSolver.addRowReal(soplex::LPRow(row, soplex::LPRow::LESS_EQUAL, halfSpacePlane.constant));
        }
        row.clear();
    }
}

template <int projDimension>
void findVectorWithMinimumWeightsDifference(const std::vector<std::pair<FairTopK::Plane<projDimension>, bool> >& halfSpaces,
    const FairTopK::WeightsDiffLPAuxVector<projDimension + 1>& weightsDiffLPAuxVector, double& optScore, 
    Eigen::VectorXd& weights) {
    constexpr int dimension = projDimension + 1;

    soplex::SoPlex soplexSolver;
    setupWeightsDiffLPSolver(halfSpaces, weightsDiffLPAuxVector, soplexSolver);

    soplex::SPxSolver::Status stat = soplexSolver.solve();
    if (stat == soplex::SPxSolver::OPTIMAL) {
        double negWeightsDifference = -soplexSolver.objValueReal();

        if (negWeightsDifference > optScore) {
            optScore = negWeightsDifference;

            soplex::DVector vars(dimension + dimension);
            soplexSolver.getPrimal(vars);

            weights.head(dimension - 1) = Eigen::VectorXd::Map(vars.get_const_ptr(), dimension - 1);
            weights(dimension - 1) = 1.0 - weights.head(dimension - 1).sum();
        }
    }
}

template <int dimension, FairTopK::Optimization opt>
bool solve(const std::vector<Eigen::VectorXd> &points, const std::vector<FairTopK::Groups>& groups,
    int k, FairTopK::GroupsMask pGroups, const std::vector<std::pair<int , int> >& pGroupsBounds, double margin,
    Eigen::VectorXd& weights) {
    using DualPlane = FairTopK::Plane<dimension>;
    using PlaneNormalVector = FairTopK::Plane<dimension>::NormalVector;

    constexpr double epsilon = 1e-8;

    int count = points.size();

    auto refWeights = weights;

    auto sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(), [](const auto& p1, const auto& p2) {
        for (int i = 0; i < dimension; i++) {
            if (p1(i) < p2(i)) return true;
            else if (p1(i) > p2(i)) return false;
        }
        return false;
    });

    std::vector<std::pair<Eigen::VectorXd, DualPlane> > pointPlanePairs;
    pointPlanePairs.reserve(count);
    for (int i = 0; i < count; i++) {
        const auto& pt = sortedPoints[i];

        if (i > 0 && (pt.array() == sortedPoints[i - 1].array()).all())
            continue;

        DualPlane plane;
        plane.normal = PlaneNormalVector::Map(pt.data());
        plane.normal -= pt(dimension - 1) * PlaneNormalVector::Ones();
        plane.normal(dimension - 1) = -1.0;
        plane.constant = -pt(dimension - 1);

        pointPlanePairs.emplace_back(pt, std::move(plane));
    }

    std::default_random_engine rand(2024);
    std::shuffle(pointPlanePairs.begin(), pointPlanePairs.end(), rand);

    auto extremePoints = FairTopK::computeExtremePoints<dimension>(weights, margin, epsilon);

    Eigen::VectorXd trialWeights(dimension);
    auto fairnessChecker = [&points, &groups, &refWeights, margin, pGroups, k, &pGroupsBounds, &trialWeights
                           ](const std::vector<std::pair<FairTopK::Plane<dimension - 1>, bool> >& halfSpaces) {
        bool found = findValidWeightVector(halfSpaces, refWeights, margin, trialWeights);
        if (found && FairTopK::checkFairness(points, groups, trialWeights, k, pGroups, pGroupsBounds, epsilon)) {
            return true;
        }

        return false;
    };

    FairTopK::BSPTree<dimension - 1> tree;

    int distinctCount = pointPlanePairs.size();
    for (int i = 0; i < distinctCount - 1; i++) {
        const auto &[pt1, plane1] = pointPlanePairs[i];

        for (int j = i + 1; j < distinctCount; j++) {
            const auto &[pt2, plane2] = pointPlanePairs[j];
            auto diff = pt1 - pt2;
            const auto& diffArray = diff.array();
            if ((diffArray >= 0.0).all() || (diffArray <= 0.0).all()) {
                continue;
            }

            auto prjoectedPlane = getProjectedIntersection<dimension>(diff);

            if (FairTopK::testIntersection(prjoectedPlane, extremePoints, epsilon)) {
                if constexpr (opt != FairTopK::Optimization::None) {
                    tree.insert(prjoectedPlane);
                }
                else {
                    bool found = tree.insert(prjoectedPlane, fairnessChecker);
                    
                    if (found) {
                        weights = std::move(trialWeights);
                        return true;
                    }
                }
            }
        }
    }

    if constexpr (opt == FairTopK::Optimization::None)
        return false;

    std::vector<double> scores;
    if constexpr (opt == FairTopK::Optimization::Utility || opt == FairTopK::Optimization::StableUtility) {
        scores = std::vector<double>(count, 0.0);
        std::transform(points.cbegin(), points.cend(), scores.begin(),  
            [&weights](const auto& point) -> double { return point.dot(weights); } );
    }
        
    FairTopK::WeightsDiffLPAuxVector<dimension> weightsDiffLPAuxVector;
    if constexpr (opt == FairTopK::Optimization::WeightsDifference) {
        weightsDiffLPAuxVector = FairTopK::setupWeightsDiffLPAuxVector<dimension>(weights, margin);
    }

    bool fair = false;
    double optScore = std::numeric_limits<double>::lowest();
    auto regionIter = tree.cbegin();
    while (regionIter != nullptr) {
        bool found = findValidWeightVector(*regionIter, weights, margin, trialWeights);

        if (found) {
            if constexpr (opt == FairTopK::Optimization::Utility || opt == FairTopK::Optimization::StableUtility) {
                auto [isFair, utility] = tryObtainFairSelection(points, groups, trialWeights, k, 
                    pGroups, pGroupsBounds, scores, epsilon);
                    
                if (isFair) {
                    fair = true;
                    if (utility > optScore) {
                        optScore = utility;
                        weights = trialWeights;
                    }
                }
            }
            else {
                bool isFair = FairTopK::checkFairness(points, groups, trialWeights, k, pGroups, pGroupsBounds, epsilon);
                
                if (isFair) {
                    fair = true;
                    findVectorWithMinimumWeightsDifference(*regionIter, weightsDiffLPAuxVector, optScore, weights);
                }
            }
        }

        ++regionIter;
    }

    if constexpr (opt == FairTopK::Optimization::StableUtility) {
        if (fair) {
            FairTopK::stabilizeFairWeightVector<dimension>(points, groups, k, pGroups, pGroupsBounds, 
                refWeights, margin, epsilon, weights);
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
    constexpr int minDimension = 3;
    constexpr int maxDimension = 6;

    if (dimension < minDimension || dimension > maxDimension) {
        std::cerr << "Error: Only support datasets with 3 <= dimensions <= 6" << std::endl;
        return -1;
    }

    constexpr int dimCount = maxDimension - minDimension + 1;
    int dimDiff = dimension - minDimension;

    auto solveFunc = boost::mp11::mp_with_index<(std::size_t)FairTopK::Optimization::NumOptions>((std::size_t)params.opt,
        [dimDiff](auto opt) {
            return boost::mp11::mp_with_index<dimCount>(dimDiff,
                [opt](auto dimDiff) { return solve<dimDiff() + minDimension, FairTopK::Optimization(opt())>; });
        });

    FairTopK::fairTopkExperiments(points, groups, protectedGroups, params, solveFunc);

    return 0;
}