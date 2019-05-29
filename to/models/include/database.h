//
// Created by jab on 29/05/19.
//
#ifndef TO_DATABASE_H
#define TO_DATABASE_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <sstream>

#include "road_user.h"
#include <logger.h>

using namespace std;

class Database {

private:
    unordered_map<string, shared_ptr<RoadUser>> database;

public:

    Database() = default;

    /* General access and display functions. */
    void displayDatabase();
    void upsert(shared_ptr<RoadUser> roadUser);
    void deleteRoadUser(string uuid);
    void deleteAll();
    std::shared_ptr<RoadUser> findRoadUser(string uuid);
    int getSize();
    vector<shared_ptr<RoadUser>> findAll();
};

#endif //TO_DATABASE_H
