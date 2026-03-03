/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#include "data_loader.h"
#include <string>
#include <fstream>
#include <unordered_map>
#include <ctime>
#include <filesystem>
#include <algorithm>

#include <boost/tokenizer.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>

namespace FairTopK {

namespace DataLoader {

namespace {
void readPreprocessedCompasData(const std::string& file, std::vector<Eigen::VectorXd>& points,  
    std::vector<int>& genders, std::vector<int>& races);
void readPreprocessedJEEData(const std::string& file, std::vector<Eigen::VectorXd>& points,
    std::vector<int>& genders, std::vector<int>& categroies);
}

bool readPreprocessedDataset(const std::string& file, std::vector<Eigen::VectorXd>& points, 
    std::vector<Groups>& groups, GroupsMask& protectedGroups) {
    if (!std::filesystem::exists(file)) {
        std::cerr << "Error: Fail to find the dataset file \'" << file 
                  << "\'. Verify the path and filename are correct." << std::endl;
        return false;
    }

    if (file.find("compas") != std::string::npos) {
        std::vector<int> races;
        std::vector<int> genders;
        readPreprocessedCompasData(file, points, genders, races);
        groups.reserve(points.size());
        constexpr int numRaces = 3, numGenders = 2;
        constexpr int afRace = 0, maleGender = 1;
        for (int i = 0; i < points.size(); i++) {
            int race = races[i];
            int gender = genders[i];
            Groups candidateGroups = (1 << race) | (1 << (gender + numRaces));
            if (race == afRace && gender == maleGender)
                candidateGroups |= (1 << (numRaces + numGenders));

            groups.push_back(candidateGroups);
        }

        protectedGroups = (1 << afRace) | (1 << (maleGender + numRaces)) | (1 << (numRaces + numGenders));
    }
    else if (file.find("jee") != std::string::npos) {
        std::vector<int> genders;
        std::vector<int> categroies;
        readPreprocessedJEEData(file, points, genders, categroies);
        groups.reserve(points.size());
        constexpr int numGenders = 2;
        constexpr int femaleGender = 1, genCategory = 0;
        for (int i = 0; i < points.size(); i++) {
            Groups candidateGroups = (1 << genders[i]) | (1 << ((categroies[i] != genCategory) + numGenders));
            groups.push_back(candidateGroups);
        }

        protectedGroups = (1 << femaleGender) | (1 << (1 + numGenders));
    }
    else {
        std::cerr << "Error: Unsupported dataset: \'" <<  file << "\'." << std::endl;
        return false;
    }

    return true;
}

namespace {

void readPreprocessedCompasData(const std::string& file, std::vector<Eigen::VectorXd>& points, 
    std::vector<int>& genders, std::vector<int>& races) {
    std::ifstream inf(file, std::ifstream::in);
    
    auto data = xt::load_csv<double>(inf);
    auto shape = data.shape();

    int dimension = shape[1] - 2;

    int count = shape[0];

    Eigen::VectorXd normalizer(dimension);
    {
        Eigen::VectorXd cwiseMinPt(dimension), cwiseMaxPt(dimension);
        auto cwiseMinRow = xt::row(data, 0);
        for (int i = 0; i < dimension; i++) {
            cwiseMinPt(i) = cwiseMinRow(i);
        }
        auto cwiseMaxRow = xt::row(data, 1);
        for (int i = 0; i < dimension; i++) {
            cwiseMaxPt(i) = cwiseMaxRow(i);
        }
        normalizer = (cwiseMaxPt - cwiseMinPt).cwiseInverse();
    }

    for (int i = 2; i < count; i++) {
        auto row = xt::row(data, i);

        Eigen::VectorXd point(dimension);
        for (int j = 0; j < dimension; j++) {
            point(j) = row(j);
        }
        point = point.cwiseProduct(normalizer);

        points.push_back(std::move(point));
        genders.push_back((int)row(dimension));
        races.push_back((int)row(dimension + 1));
    }
}

void readPreprocessedJEEData(const std::string& file, std::vector<Eigen::VectorXd>& points, 
    std::vector<int>& genders, std::vector<int>& categroies) {
    std::ifstream inf(file, std::ifstream::in);
    
    auto data = xt::load_csv<double>(inf);
    auto shape = data.shape();

    int dimension = shape[1] - 2;

    int count = shape[0];

    Eigen::VectorXd normalizer(dimension);
    {
        Eigen::VectorXd cwiseMinPt(dimension), cwiseMaxPt(dimension);
        auto cwiseMinRow = xt::row(data, 0);
        for (int i = 0; i < dimension; i++) {
            cwiseMinPt(i) = cwiseMinRow(i);
        }
        auto cwiseMaxRow = xt::row(data, 1);
        for (int i = 0; i < dimension; i++) {
            cwiseMaxPt(i) = cwiseMaxRow(i);
        }
        normalizer = (cwiseMaxPt - cwiseMinPt).cwiseInverse();
    }

    for (int i = 2; i < count; i++) {
        auto row = xt::row(data, i);

        Eigen::VectorXd point(dimension);
        for (int j = 0; j < dimension; j++) {
            point(j) = row(j);
        }
        point = point.cwiseProduct(normalizer);

        points.push_back(std::move(point));
        genders.push_back((int)row(dimension));
        categroies.push_back((int)row(dimension + 1));
    }
}

}

}

}