#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include <arpa/inet.h>
#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <tuple>

#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>

using namespace rapidjson;
using namespace std;
using std::string;
using std::cout;
using std::tuple;
using std::vector;

typedef uint32_t uint4;

#define NOTIFY_ADD "notify_add"
#define SUBSCRIPTION_RESPONSE "subscription_response"
#define UNSUBSCRIPTION_RESPONSE "unsubscription_response"
#define TRAJECTORY_FEEDBACK "maneuver_feedback"
string incomplete_message = "";
#define MAXIMUM_TRANSFER 100000

/* struct to represent fields found in JSON string. */
struct Detected_Road_User {

string type;
string context;
string origin;
string version;
uint64_t timestamp;
string uuid;
string its_station_type;
bool connected;
int32_t latitude;
int32_t longitude;
string position_type;
uint16_t heading;
uint16_t speed;
uint16_t acceleration;
uint16_t yaw_rate;
float length;
float width;
float height;
string color;
uint4 lane_position;
uint8_t existence_probability;
uint16_t position_semi_major_confidence;
uint16_t position_semi_minor_confidence;
uint16_t position_semi_major_orientation;
uint16_t heading_c;
uint16_t speed_c;
uint16_t acceleration_c;
uint16_t yaw_rate_c;
float length_c;
float width_c;
float height_c;
string signature;

};

struct Detected_To_Notification {
  string type;
  string context;
  string origin;
  string version;
  uint64_t timestamp;
  int subscriptionId;
  vector<Detected_Road_User> ru_description_list;
  string signature;
};

struct Detected_Trajectory_Feedback {
  string type;
  string context;
  string origin;
  string version;
  uint64_t timestamp;
  string uuid_vehicle;
  string uuid_to;
  string uuid_maneuver;
  uint64_t timestamp_message;
  string feedback;
  string reason;
  string signature;

};

struct Detected_Subscription_Response {
  string type;
  string context;
  string origin;
  string version;
  uint64_t timestamp;
  string result;
  int request_id;
  int subscriptionId;
  string signature;

};

struct Detected_Unsubscription_Response {
  string type;
  string context;
  string origin;
  string version;
  uint64_t timestamp;
  string result;
  int request_id;
  string signature;
};

Document parse(string readFromServer) {
  Document document;
  ParseResult result = document.Parse(readFromServer.c_str());
  if(!result) return NULL;
  return document;
}

Detected_Road_User assignRoadUserVals(Document document, Detected_To_Notification detectedToNotification) {
	Detected_Road_User values;

  values.type = document["type"].GetString();
  values.context = document["context"].GetString();
  values.origin = document["origin"].GetString();
  values.version = document["version"].GetString();
  values.timestamp = document["timestamp"].GetUint64();
  values.uuid = document["message"]["uuid"].GetString();
  values.its_station_type = document["message"]["its_station_type"].GetString();
  values.connected = document["message"]["connected"].GetBool();
  values.latitude = document["message"]["position"]["latitude"].GetDouble();
  values.longitude = document["message"]["position"]["longitude"].GetDouble();
  values.position_type = document["message"]["position_type"].GetString();
  values.heading = document["message"]["heading"].GetDouble();
  values.speed = document["message"]["speed"].GetDouble();
  values.acceleration = document["message"]["acceleration"].GetDouble();
  values.yaw_rate = document["message"]["yaw_rate"].GetDouble();
  values.length = document["message"]["size"]["length"].GetDouble();
  values.width = document["message"]["size"]["width"].GetDouble();
  values.height = document["message"]["size"]["height"].GetDouble();
  values.color = document["message"]["color"].GetString();
  values.lane_position = document["message"]["lane_position"].GetUint();
  values.existence_probability = document["message"]["existence_probability"].GetUint();
  values.position_semi_major_confidence = document["message"]["confidence"]["position_semi_major_confidence"].GetUint();
  values.position_semi_minor_confidence = document["message"]["confidence"]["position_semi_minor_confidence"].GetUint();
  values.position_semi_major_orientation = document["message"]["confidence"]["position_semi_major_orientation"].GetUint();
  values.heading_c = document["message"]["confidence"]["heading"].GetUint();
  values.speed_c = document["message"]["confidence"]["speed"].GetUint();
  values.acceleration_c = document["message"]["confidence"]["acceleration"].GetUint();
  values.yaw_rate_c = document["message"]["confidence"]["yaw_rate"].GetUint();
  values.length_c = document["message"]["confidence"]["size"]["length"].GetDouble();
  values.width_c = document["message"]["confidence"]["size"]["width"].GetDouble();
  values.height_c = document["message"]["confidence"]["size"]["height"].GetDouble();
  values.signature = document["signature"].GetString();


  return values;

}

Detected_To_Notification assignNotificationVals(Document document) {
  Detected_To_Notification values;
  values.type = document["type"].GetString();
  values.context = document["context"].GetString();
  values.origin = document["origin"].GetString();
  values.version = document["version"].GetString();
  values.timestamp = document["timestamp"].GetUint64();
  values.subscriptionId = document["message"]["subscription_id"].GetInt();
  values.signature = document["signature"].GetString();

  for(auto& v : document["message"]["ru_description_list"].GetArray()) {
    StringBuffer sb;
    Writer<StringBuffer> writer(sb);
    v.Accept(writer);
    values.ru_description_list.push_back(assignRoadUserVals(parse(sb.GetString()),values));
    sb.Clear();
    writer.Reset(sb);
  }

  return values;
}

Detected_Trajectory_Feedback assignTrajectoryFeedbackVals(Document document) {
  Detected_Trajectory_Feedback values;

  values.type = document["type"].GetString();
  values.context = document["context"].GetString();
  values.origin = document["origin"].GetString();
  values.version = document["version"].GetString();
  values.uuid_vehicle = document["source_uuid"].GetString();
  values.uuid_to = document["destination_uuid"].GetString();
  values.timestamp = document["timestamp"].GetUint64();
  values.uuid_maneuver = document["message"]["uuid_maneuver"].GetString();
  values.timestamp_message = document["message"]["timestamp"].GetUint64();
  values.feedback = document["message"]["feedback"].GetString();
  values.reason = document["message"]["reason"].GetString();
  values.signature = document["signature"].GetString();

  return values;

}


Detected_Subscription_Response assignSubResponseVals(Document document) {

  Detected_Subscription_Response values;

  values.type = document["type"].GetString();
  values.context = document["context"].GetString();
  values.origin = document["origin"].GetString();
  values.version = document["version"].GetString();
  values.timestamp = document["timestamp"].GetUint64();
  values.result = document["message"]["result"].GetString();
  values.request_id = document["message"]["request_id"].GetInt();
  values.subscriptionId = document["message"]["subscription_id"].GetInt();
  values.signature = document["signature"].GetString();

  return values;

}

Detected_Unsubscription_Response assignUnsubResponseVals(Document document) {

  Detected_Unsubscription_Response values;

  values.type = document["type"].GetString();
  values.context = document["context"].GetString();
  values.origin = document["origin"].GetString();
  values.version = document["version"].GetString();
  values.timestamp = document["timestamp"].GetUint64();
  values.result = document["message"]["result"].GetInt();
  values.request_id = document["message"]["request_id"].GetInt();
  values.signature = document["signature"].GetString();

  return values;

}

int filterInput(Document document) {

  if(!(document.IsObject()) || !(document.HasMember("type"))){
    return -1;
  }

  if(document["type"] == NOTIFY_ADD) {
    return 0;
  }

  else if(document["type"] == SUBSCRIPTION_RESPONSE) {
    return 1;
  }

  else if(document["type"] == UNSUBSCRIPTION_RESPONSE) {
    return 2;
  }

  else if(document["type"] == TRAJECTORY_FEEDBACK) {
    cout << "\n\n\n\n\n\n\n *********************************** Received Trajectory Feedback *********************************** \n\n\n\n\n\n\n";
    return 3;
  }

  else {
    return -1;
  }

}

string listenDataTCP(int socket_c) {

  char dataReceived[MAXIMUM_TRANSFER];
  memset(dataReceived,0,sizeof(dataReceived));

  while(1) {
    int i = read(socket_c,dataReceived,sizeof(dataReceived));

    if(i < 0) {
      perror("Error: Failed to receive transmitted data.\n");
      break;
    }
    else if(i == 0) {
      printf("Socket closed from the remote server.\n");
      break;
    }
    else if(i > 0) {
      if(i>1) printf("Received %d bytes of data. Data received: %s\n",i,dataReceived);
      auto found = string(dataReceived).find("\n");
      auto ending = string(dataReceived).find_last_of("}");
      auto start = string(dataReceived).find("{");
      if((found!=std::string::npos)){
        if(found != 0){
            string copy_of_return = incomplete_message;
            incomplete_message = string(dataReceived).substr(found,i);
            string toReturn = copy_of_return + string(dataReceived).substr(0,found);
            auto startingB = string(toReturn).find("{");
            if(startingB != std::string::npos && startingB == 0) {
              cout << toReturn << endl;
              return toReturn;
            }else incomplete_message = "";
            // return toReturn;
          }else{
            auto startingBracket = string(incomplete_message).find("{");
            if((startingBracket!=std::string::npos)){
              if(startingBracket == 0){
                string terminating_return = incomplete_message;
                incomplete_message = "";
                cout << terminating_return << endl;
                return terminating_return;
              } else incomplete_message = "";
            }
            }
        }else incomplete_message += string(dataReceived).substr(0,i);
    }
  }
}
