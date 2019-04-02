// this file stores the road user information presented by the v2x gateway

// Created by : KCL

#include "road_user.cpp"

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>

using namespace std;

class Database {

private:
    unordered_map<string, shared_ptr<RoadUser>> database;

public:

  Database() {
  }

  /* General access and display functions. */
  void displayDatabase();
  void upsert(shared_ptr<RoadUser> roadUser);
  void deleteRoadUser(string uuid);
  void deleteAll();
  bool findRoadUser(string uuid);
  int getSize();
  vector<shared_ptr<RoadUser>> findAll();
};


/**
*   @description Displays all RoadUsers in the database.
*/
void Database::displayDatabase() {
  for_each(database.begin(), database.end() , [](pair<string, shared_ptr<RoadUser>> element){
      cout << element.first << " :: " << element.second << endl;
  });
}

/**
*   @description Update or Insert a RoadUser pointer into the database.
*   @param roadUser is a RoadUser pointer.
*/

void Database::upsert(shared_ptr<RoadUser> roadUser) {
  database[roadUser->getUuid()] = roadUser;
}

/**
*   @description Erases a RoadUser pointer from the database.
*   @param roadUser is a RoadUser pointer.
*/
void Database::deleteRoadUser(string uuid) {
  const auto &iterator{database.find(uuid)};
  if (iterator != database.end()) {
    //FIXME: huge memory leak. Need to find how to delete a pointer in a map
//    delete iterator->second; // NOT WORKING
    database.erase(uuid);
  }
}

/**
*   @description Removes all RoadUser pointers in the database.
*/
void Database::deleteAll() {
/*
  //FIXME: huge memory leak. Need to find how to delete a pointer in a map
  for_each(database.begin(), database.end() , [](pair<string, RoadUser*> element){
      delete element.second;
  });
*/
  database.clear();
}

int Database::getSize() {
  return database.size();
}

vector<shared_ptr<RoadUser>> Database::findAll() {
  auto values = vector<shared_ptr<RoadUser>>();
  for(auto elem : database) {
    values.push_back(elem.second);
  }
  return values;
}
