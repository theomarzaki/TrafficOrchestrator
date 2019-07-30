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
#include <list>

#include "road_user.h"

class Database {

private:
    std::unordered_map<std::string, std::shared_ptr<RoadUser>> database;

public:

    Database() = default;

    /* General access and display functions. */
    void displayDatabase();
    void upsert(const std::shared_ptr<RoadUser>& roadUser);
    void deleteRoadUser(const std::string& uuid);
    void deleteAll();
    std::shared_ptr<RoadUser> findRoadUser(const std::string& uuid);
    int getSize();
    std::vector<std::shared_ptr<RoadUser>> findAll();
    std::unique_ptr<std::list<std::shared_ptr<RoadUser>>> dump();
};

#endif //TO_DATABASE_H
