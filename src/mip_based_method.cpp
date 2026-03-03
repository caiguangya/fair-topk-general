/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <utility>
#include <boost/mp11/algorithm.hpp>

#include <Eigen/Dense>
#include <gurobi/gurobi_c++.h>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "utility.h"
#include "stabilization.h"
#include "data_loader.h"
#include "experiments.h"

bool checkGurobiLicense() {
    try {
        GRBEnv env = GRBEnv(true);
        env.set("OutputFlag", "0");
        env.start();
        return true;
    } catch (const GRBException& e) {
        std::cerr << e.getMessage() << std::endl;
        return false;
    }
}

template <FairTopK::Optimization opt>
bool solveGruobi(int threadCount, const std::vector<Eigen::VectorXd> &points, const std::vector<FairTopK::Groups>& groups,
    int k, FairTopK::GroupsMask pGroups, const std::vector<std::pair<int , int> >& pGroupsBounds, double margin,
    Eigen::VectorXd& weights) {
    GRBEnv env = GRBEnv(true);
    env.set("OutputFlag", "0");
    if (threadCount > 0)
        env.set(GRB_IntParam_Threads, threadCount);
    env.start();
    
    GRBModel model = GRBModel(env);

    constexpr double epsilon = 1e-8;
    model.set(GRB_DoubleParam_IntFeasTol, epsilon);
    model.set(GRB_DoubleParam_FeasibilityTol, epsilon);

    if constexpr (opt == FairTopK::Optimization::None) {
        model.set("MIPFocus", "1");
        model.set("SolutionLimit", "1");
    }

    if constexpr (opt != FairTopK::Optimization::None) {
        model.set("MIPGap", "1e-6");
    }

    int count = points.size();
    int dimension = points[0].rows();
    int pGroupCount = pGroupsBounds.size();

    std::vector<double> ones(count, 1.0);

    std::vector<GRBVar> scoreVars;
    std::vector<GRBVar> indicatorVars;
    std::vector<std::vector<GRBVar> > pGroupsIndVars;

    scoreVars.reserve(dimension + 1);
    indicatorVars.reserve(count);

    for (int i = 0; i < pGroupCount; i++) {
        std::vector<GRBVar> vars;
        vars.reserve(count);
        pGroupsIndVars.push_back(std::move(vars));
    }

    auto pGroupsVec = FairTopK::getPGroupsVec(pGroups);

    for (int i = 0; i < dimension; i++) {
        double lb = std::max(0.0, weights(i) - margin);
        double ub = std::min(1.0, weights(i) + margin);
        GRBVar var = model.addVar(lb, ub, 0.0, GRB_CONTINUOUS);
        scoreVars.push_back(var);
    }

    {
        GRBVar var = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
        scoreVars.push_back(var);
    }

    for (int i = 0; i < count; i++) {
        GRBVar var = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);
        indicatorVars.push_back(var);

        FairTopK::Groups candidateGroups = groups[i];

        if ((candidateGroups & pGroups) == 0)  continue;
        
        for (int j = 0; j < pGroupCount; j++) {
            auto mask = FairTopK::getGroupsMask(pGroupsVec[j]);
            if ((candidateGroups & mask) != 0)
                pGroupsIndVars[j].push_back(var);
        }
    }

    //sum_{i=1}^{d} w_i = 1
    {
        GRBLinExpr expr = 0;
        expr.addTerms(ones.data(), scoreVars.data(), dimension);
        model.addConstr(expr == 1.0);
    }


    for (int i = 0; i < count; i++) {
        GRBLinExpr expr = 0;  

        expr.addTerms(points[i].data(), scoreVars.data(), dimension);
        expr -= scoreVars[dimension];
        expr -= indicatorVars[i];

        model.addRange(expr, -1.0, 0.0);
    }

    for (int i = 0; i < pGroupCount; i++) {
        auto [lowerBound, upperBound] = pGroupsBounds[i];
        const auto &pGroupIndVars = pGroupsIndVars[i];

        GRBLinExpr expr = 0;
        expr.addTerms(ones.data(), pGroupIndVars.data(), pGroupIndVars.size());
        model.addRange(expr, lowerBound, upperBound);
    }

    {
        GRBLinExpr expr = 0;
        expr.addTerms(ones.data(), indicatorVars.data(), count);
        model.addConstr(expr == k);
    }

    if constexpr (opt == FairTopK::Optimization::Utility || opt == FairTopK::Optimization::StableUtility) {
        std::vector<double> scores(count, 0.0);
        std::transform(points.cbegin(), points.cend(), scores.begin(),
            [&weights](const auto& point) -> double { return point.dot(weights); } );
   
        GRBLinExpr objExpr = 0;
        objExpr.addTerms(scores.data(), indicatorVars.data(), count);

        model.setObjective(objExpr, GRB_MAXIMIZE);
    }
    else if constexpr (opt == FairTopK::Optimization::WeightsDifference) {
        std::vector<GRBVar> boundVars;
        boundVars.reserve(dimension);
        for (int i = 0; i < dimension; i++) {
            GRBVar var = model.addVar(0.0, GRB_INFINITY, 1.0, GRB_CONTINUOUS);
            boundVars.push_back(var);
        }

        constexpr int boundConstrVarsCount = 2;

        std::array<GRBVar, boundConstrVarsCount> boundConstrVars;
        constexpr std::array<double, boundConstrVarsCount> coeffsPos = { 1.0, -1.0 };
        constexpr std::array<double, boundConstrVarsCount> coeffsNeg = { 1.0, 1.0 };

        GRBLinExpr expr = 0;

        for (int i = 0; i < dimension; i++) {
            boundConstrVars[0] = boundVars[i];
            boundConstrVars[1] = scoreVars[i];

            double weight = weights(i);

            expr.clear();
            expr.addTerms(coeffsPos.data(), boundConstrVars.data(), boundConstrVarsCount);
            model.addConstr(expr >= -weight);

            expr.clear();
            expr.addTerms(coeffsNeg.data(), boundConstrVars.data(), boundConstrVarsCount);
            model.addConstr(expr >= weight);
        }

        model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
    }

    model.optimize();

    int status = model.get(GRB_IntAttr_Status);
    if (status == GRB_INFEASIBLE || status == GRB_INF_OR_UNBD) {
        return false;
    }
    else {
        Eigen::VectorXd refWeights = weights;
        for (int i = 0; i < dimension; i++) {
            weights(i) = scoreVars[i].get(GRB_DoubleAttr_X);
        }
        
        if constexpr (opt == FairTopK::Optimization::StableUtility) {
            FairTopK::stabilizeFairWeightVector(points, groups, k, pGroups, pGroupsBounds, 
                refWeights, margin, epsilon, weights);
        }

        return true;
    }
}

template <FairTopK::Optimization opt>
bool solveSCIP([[maybe_unused]] int threadCount, const std::vector<Eigen::VectorXd> &points, 
    const std::vector<FairTopK::Groups>& groups, int k, FairTopK::GroupsMask pGroups, 
    const std::vector<std::pair<int , int> >& pGroupsBounds, double margin, Eigen::VectorXd& weights) {
    SCIP *scip = nullptr;

    SCIP_CALL(SCIPcreate(&scip));
    SCIP_CALL(SCIPincludeDefaultPlugins(scip));
    SCIPmessagehdlrSetQuiet(SCIPgetMessagehdlr(scip), true);

    SCIP_CALL(SCIPcreateProbBasic(scip, ""));

    constexpr double epsilon = 1e-8;
    SCIP_CALL(SCIPsetRealParam(scip, "numerics/feastol", epsilon));

    if constexpr (opt == FairTopK::Optimization::None) {
        SCIP_CALL(SCIPsetIntParam(scip, "limits/solutions", 1));
        SCIP_CALL(SCIPsetIntParam(scip, "limits/maxsol", 1));
    }

    if constexpr (opt != FairTopK::Optimization::None) {
        SCIP_CALL(SCIPsetRealParam(scip, "limits/gap", 1e-6));
    }

    int count = points.size();
    int dimension = points[0].rows();
    int pGroupCount = pGroupsBounds.size();

    Eigen::VectorXd refWeights = weights;

    std::vector<double> ones(count, 1.0);

    std::vector<SCIP_VAR *> scoreVars;
    std::vector<SCIP_VAR *> indicatorVars;
    std::vector<std::vector<SCIP_VAR *> > pGroupsIndVars;
    std::vector<SCIP_CONS *> constraints;
    
    std::vector<SCIP_VAR *> boundVars;

    const int addlConsCount = (opt == FairTopK::Optimization::WeightsDifference) ? 2 * dimension : 0;

    scoreVars.reserve(dimension + 1);
    indicatorVars.reserve(count);
    constraints.reserve(count + 2 + pGroupCount + addlConsCount);

    for (int i = 0; i < pGroupCount; i++) {
        std::vector<SCIP_VAR *> vars;
        vars.reserve(count);
        pGroupsIndVars.push_back(std::move(vars));
    }

    auto pGroupsVec = FairTopK::getPGroupsVec(pGroups);

    for (int i = 0; i < dimension; i++) {
        double lb = std::max(0.0, weights(i) - margin);
        double ub = std::min(1.0, weights(i) + margin);

        SCIP_VAR *var = nullptr;
        SCIP_CALL(SCIPcreateVarBasic(scip, &var, nullptr, lb, ub, 0.0, SCIP_VARTYPE_CONTINUOUS));
        SCIP_CALL(SCIPaddVar(scip, var));

        scoreVars.push_back(var);
    }

    {
        SCIP_VAR *var = nullptr;
        SCIP_CALL(SCIPcreateVarBasic(scip, &var, nullptr, 0.0, 1.0, 0.0, SCIP_VARTYPE_CONTINUOUS));
        SCIP_CALL(SCIPaddVar(scip, var));

        scoreVars.push_back(var);
    }

    for (int i = 0; i < count; i++) {
        SCIP_VAR *var = nullptr;
        SCIP_CALL(SCIPcreateVarBasic(scip, &var, nullptr, 0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
        SCIP_CALL(SCIPaddVar(scip, var));

        indicatorVars.push_back(var);

        FairTopK::Groups candidateGroups = groups[i];

        if ((candidateGroups & pGroups) == 0)  continue;
        
        for (int j = 0; j < pGroupCount; j++) {
            auto mask = FairTopK::getGroupsMask(pGroupsVec[j]);
            if ((candidateGroups & mask) != 0)
                pGroupsIndVars[j].push_back(var);
        }
    }

    {
        SCIP_CONS *cons = nullptr;
        SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", dimension, scoreVars.data(), ones.data(), 1.0, 1.0));
        SCIP_CALL(SCIPaddCons(scip, cons));

        constraints.push_back(cons);
    }

    {
        std::vector<SCIP_VAR *> consVars = scoreVars;
        Eigen::VectorXd consVals = -Eigen::VectorXd::Ones(dimension + 2);

        consVars.push_back(nullptr);

        for (int i = 0; i < count; i++) {
            consVars[dimension + 1] = indicatorVars[i];
            consVals.head(dimension) = points[i];

            SCIP_CONS *cons = nullptr;
            SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", dimension + 2, consVars.data(), consVals.data(), 
                -1.0, 0.0));
            SCIP_CALL(SCIPaddCons(scip, cons));

            constraints.push_back(cons);
        }
    }

    for (int i = 0; i < pGroupCount; i++) {
        auto [lowerBound, upperBound] = pGroupsBounds[i];
        auto &pGroupIndVars = pGroupsIndVars[i];

        SCIP_CONS *cons = nullptr;
        SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", pGroupIndVars.size(), pGroupIndVars.data(), ones.data(), 
            (SCIP_Real)lowerBound, (SCIP_Real)upperBound));
        SCIP_CALL(SCIPaddCons(scip, cons));

        constraints.push_back(cons);
    }

    {
        SCIP_CONS *cons = nullptr;
        SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", count, indicatorVars.data(), ones.data(), 
            (SCIP_Real)k, (SCIP_Real)k));
        SCIP_CALL(SCIPaddCons(scip, cons));

        constraints.push_back(cons);
    }

    if constexpr (opt == FairTopK::Optimization::Utility || opt == FairTopK::Optimization::StableUtility) {
        for (int i = 0; i < count; i++) {
            SCIP_CALL(SCIPchgVarObj(scip, indicatorVars[i], weights.dot(points[i])));
        }

        SCIP_CALL(SCIPsetObjsense(scip, SCIP_OBJSENSE_MAXIMIZE));
    }
    else if constexpr (opt == FairTopK::Optimization::WeightsDifference) {
        boundVars.reserve(dimension);
        for (int i = 0; i < dimension; i++) {
            SCIP_VAR *var = nullptr;
            SCIP_CALL(SCIPcreateVarBasic(scip, &var, nullptr, 0.0, SCIPinfinity(scip), 1.0, SCIP_VARTYPE_CONTINUOUS));
            SCIP_CALL(SCIPaddVar(scip, var));

            boundVars.push_back(var);
        }

        constexpr int boundConstrVarsCount = 2;

        std::array<SCIP_VAR *, boundConstrVarsCount> boundConstrVars;
        constexpr std::array<double, boundConstrVarsCount> coeffsPos = { 1.0, -1.0 };
        constexpr std::array<double, boundConstrVarsCount> coeffsNeg = { 1.0, 1.0 };

        for (int i = 0; i < dimension; i++) {
            boundConstrVars[0] = boundVars[i];
            boundConstrVars[1] = scoreVars[i];

            double weight = weights(i);

            SCIP_CONS *cons = nullptr;
            SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", 
                boundConstrVarsCount, boundConstrVars.data(), const_cast<double *>(coeffsPos.data()),
                -weight, SCIPinfinity(scip)));
            SCIP_CALL(SCIPaddCons(scip, cons));

            constraints.push_back(cons);

            cons = nullptr;
            SCIP_CALL(SCIPcreateConsBasicLinear(scip, &cons, "", 
                boundConstrVarsCount, boundConstrVars.data(), const_cast<double *>(coeffsNeg.data()),
                weight, SCIPinfinity(scip)));
            SCIP_CALL(SCIPaddCons(scip, cons));

            constraints.push_back(cons);
        }

        SCIP_CALL(SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE));
    }

    SCIP_CALL(SCIPsolve(scip));

    bool found = (SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL);
    if (found)  {
        SCIP_SOL *sol = SCIPgetBestSol(scip);

        Eigen::VectorXd refWeights = weights;
        for (int i = 0; i < dimension; i++) {
            weights(i) = SCIPgetSolVal(scip, sol, scoreVars[i]);
        }

        if constexpr (opt == FairTopK::Optimization::StableUtility) {
            FairTopK::stabilizeFairWeightVector(points, groups, k, pGroups, pGroupsBounds, 
                refWeights, margin, epsilon, weights);
        }
    }

    for (auto var : scoreVars) {
        SCIP_CALL(SCIPreleaseVar(scip, &var));
    }
    for (auto var: indicatorVars) {
        SCIP_CALL(SCIPreleaseVar(scip, &var));
    }
    for (auto cons: constraints) {
        SCIP_CALL(SCIPreleaseCons(scip, &cons));
    }

    if constexpr (opt == FairTopK::Optimization::WeightsDifference) {
        for (auto var : boundVars) { 
            SCIP_CALL(SCIPreleaseVar(scip, &var));
        }
    }

    SCIP_CALL(SCIPfree(&scip));

    return found;
}

int main(int argc, char* argv[]) {
    std::vector<Eigen::VectorXd> points;
    std::vector<FairTopK::Groups> groups;
    FairTopK::GroupsMask protectedGroups = 0;

    auto [fileName, params] = FairTopK::parseCommandLine(argc, argv);

    if (params.solver != "gurobi" && params.solver != "scip") {
        std::cerr << "Error: Unknown solver: " << params.solver << std::endl;
        return -1;
    }

    bool isGurobi = (params.solver == "gurobi");
    if (isGurobi && !checkGurobiLicense()) {
        return -1;
    }

    bool success = FairTopK::DataLoader::readPreprocessedDataset(fileName, points, groups, protectedGroups);
    if (!success) return -1;

    auto solveFunc = boost::mp11::mp_with_index<(std::size_t)FairTopK::Optimization::NumOptions>((std::size_t)params.opt,
        [isGurobi](auto opt) { return isGurobi ? solveGruobi<FairTopK::Optimization(opt())> :
                                                 solveSCIP<FairTopK::Optimization(opt())>;
        });

    FairTopK::fairTopkExperiments(points, groups, protectedGroups, params, 
        [threadCount = params.threadCount, solveFunc]<class... Args>(Args&&... params) {
            return solveFunc(threadCount, std::forward<Args>(params)...);
    });

    return 0;
}