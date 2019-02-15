// this file stores the road user information presented by the v2x gateway

// Created by : KCL

#include "road_user.cpp"

#include <iostream>
#include <string>
#include <vector>

using std::string;
using std::cout;
using std::vector;

using namespace std;

class Database {

private:

  vector<RoadUser*> * database; // Pointer to vector of RoadUser pointers.

public:

Database() {

  database = new vector<RoadUser*>();

}

/* General access and display functions. */
void displayDatabase();
void insertRoadUser(RoadUser *);
void deleteRoadUser(string uuid);
void deleteAll();
RoadUser * getRoadUser(string uuid);
bool findRoadUser(string uuid);
int getSize();
vector<RoadUser*> * getDatabase();

~Database();

};

Database::~Database() {}

/**
*   @description Displays all RoadUsers in the database.
*/
void Database::displayDatabase() {
    for(vector<RoadUser*>::iterator it = database->begin(); it != database->end(); ++it) {
        cout << *it;
    }
}

/**
*   @description Inserts a RoadUser pointer into the database.
*   @param roadUser is a RoadUser pointer.
*/
void Database::insertRoadUser(RoadUser * roadUser) {
  if(findRoadUser(roadUser->getUuid())) {
    deleteRoadUser(roadUser->getUuid());
    database->push_back(roadUser);
  }
  else {
    database->push_back(roadUser);
  }

}

/**
*   @description Erases a RoadUser pointer from the database.
*   @param roadUser is a RoadUser pointer.
*/
void Database::deleteRoadUser(string uuid) {
  for(vector<RoadUser*>::iterator it = database->begin(); it != database->end(); ++it) {
    if((*it)->getUuid() == uuid) {
      database->erase(it);
    }
  }
}

/**
*   @description Removes all RoadUser pointers in the database.
*/
void Database::deleteAll() {
  database->clear();
}

/**
*   @description Searches the vector of pointers for a certain RoadUser.
*   @param uuid is a unique [id] field associated with each RoadUser (pointer).
*   @return a pointer to a RoadUser associated with the parameter.
*/
RoadUser * Database::getRoadUser(string uuid) {
  for(vector<RoadUser*>::iterator it = database->begin(); it != database->end(); ++it) {
    if((*it)->getUuid() == uuid) {
      return *it;
    }
  }
}

bool Database::findRoadUser(string uuid) {
  for(vector<RoadUser*>::iterator it = database->begin(); it != database->end(); ++it) {
    if((*it)->getUuid() == uuid) {
      return true;
    }
    else {
      return false;
    }
  }
}

int Database::getSize() {
  return database->size();
}

vector<RoadUser*> * Database::getDatabase() {
  return database;
}
