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
#define NOTIFY_DELETE "notify_delete"
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
string message_id;
string source_uuid;

};

struct Detected_To_Notification {
  string type;
  string context;
  string origin;
  string version;
  string uuid;
  string source_uuid;
  uint64_t timestamp;
  int subscriptionId;
  vector<Detected_Road_User> ru_description_list;
  string signature;
  string message_id;
};

struct Detected_Trajectory_Feedback {
  string type;
  string context;
  string origin;
  string version;
  uint64_t timestamp;
  string source_uuid;
  string uuid_vehicle;
  string uuid_to;
  string uuid_maneuver;
  uint64_t timestamp_message;
  string feedback;
  string reason;
  string signature;
  string message_id;

};

struct Detected_Subscription_Response {
  string type;
  string context;
  string origin;
  string version;
  uint64_t timestamp;
  string result;
  int request_id;
  string source_uuid;
  string destination_uuid;
  int subscriptionId;
  string signature;
  string message_id;

};

struct Detected_Unsubscription_Response {
  string type;
  string context;
  string origin;
  string version;
  uint64_t timestamp;
  string result;
  string source_uuid;
  string destination_uuid;
  int request_id;
  string signature;
  string message_id;
};

void write_to_log(const string & text){
	std::ofstream log_file("../logs/log_file.txt",std::ios_base::out | std::ios_base::app);
	log_file << text << endl;
}

Document parse(string readFromServer) {
  Document document;
  ParseResult result = document.Parse(readFromServer.c_str());
  if(!result) return NULL;
  return document;
}

Detected_Road_User assignRoadUserVals(Document document, Detected_To_Notification detectedToNotification) {
	Detected_Road_User values;

  values.type = document["type"].GetString();
  document.HasMember("context") ? values.context = document["context"].GetString() : values.context = "placeholder";
  document.HasMember("origin") ? values.origin = document["origin"].GetString() : values.origin = "placeholder";
  document.HasMember("version") ? values.version = document["version"].GetString() : values.version = "placeholder";
  document.HasMember("timestamp") ? values.timestamp = document["timestamp"].GetUint64() : values.timestamp = -1;
  document["message"].HasMember("uuid") ? values.uuid = document["message"]["uuid"].GetString() : values.uuid = "placeholder";
  document["message"].HasMember("its_station_type") ? values.its_station_type = document["message"]["its_station_type"].GetString() : values.its_station_type = "placeholder";
  document["message"].HasMember("connected") ? values.connected = document["message"]["connected"].GetBool() : values.connected = false;
  document["message"]["position"].HasMember("latitude") ? values.latitude = document["message"]["position"]["latitude"].GetDouble() : values.latitude = 0.0;
  document["message"]["position"].HasMember("longitude") ? values.longitude = document["message"]["position"]["longitude"].GetDouble() : values.longitude = 0.0;
  document["message"].HasMember("position_type") ? values.position_type = document["message"]["position_type"].GetString() : values.position_type = "placeholder";
  document["message"].HasMember("heading") ? values.heading = document["message"]["heading"].GetDouble() : values.heading = 0.0;
  document["message"].HasMember("speed") ? values.speed = document["message"]["speed"].GetDouble() : values.speed = 0.0;
  document["message"].HasMember("acceleration") ? values.acceleration = document["message"]["acceleration"].GetDouble() : values.acceleration = 0.0;
  document["message"].HasMember("yaw_rate") ? values.yaw_rate = document["message"]["yaw_rate"].GetDouble() : values.yaw_rate = 0.0;
  document["message"]["size"].HasMember("length") ? values.length = document["message"]["size"]["length"].GetDouble() : values.length = 0.0;
  document["message"]["size"].HasMember("width") ? values.width = document["message"]["size"]["width"].GetDouble() : values.width = 0.0;
  document["message"]["size"].HasMember("height") ? values.height = document["message"]["size"]["height"].GetDouble() : values.height = 0.0;
  document["message"].HasMember("existence_probability") ? values.existence_probability = document["message"]["existence_probability"].GetUint() : values.existence_probability = -1;
  document["message"]["confidence"].HasMember("position_semi_major_confidence") ? values.position_semi_major_confidence = document["message"]["confidence"]["position_semi_major_confidence"].GetUint() : values.position_semi_major_confidence = -1;
  document["message"]["confidence"].HasMember("position_semi_minor_confidence") ? values.position_semi_minor_confidence = document["message"]["confidence"]["position_semi_minor_confidence"].GetUint() : values.position_semi_minor_confidence = -1;
  document["message"]["confidence"].HasMember("position_semi_major_orientation") ? values.position_semi_major_orientation = document["message"]["confidence"]["position_semi_major_orientation"].GetUint() : values.position_semi_major_orientation = -1;
  document["message"]["confidence"].HasMember("heading") ? values.heading_c = document["message"]["confidence"]["heading"].GetUint() : values.heading_c = -1;
  document["message"]["confidence"].HasMember("speed") ? values.speed_c = document["message"]["confidence"]["speed"].GetUint() : values.speed_c = -1;
  document["message"]["confidence"].HasMember("acceleration") ? values.acceleration_c = document["message"]["confidence"]["acceleration"].GetUint() : values.acceleration_c = 0;
  document["message"]["confidence"].HasMember("yaw_rate") ? values.yaw_rate_c = document["message"]["confidence"]["yaw_rate"].GetUint() : values.yaw_rate_c = 0;
  document["message"]["confidence"]["size"].HasMember("length") ? values.length_c = document["message"]["confidence"]["size"]["length"].GetDouble() : values.length_c = 0.0;
  document["message"]["confidence"]["size"].HasMember("width") ? values.width_c = document["message"]["confidence"]["size"]["width"].GetDouble() : values.width_c = 0.0;
  document["message"]["confidence"]["size"].HasMember("height") ? values.height_c = document["message"]["confidence"]["size"]["height"].GetDouble() : values.height_c = 0.0;
  document["message"].HasMember("color") ? values.color = document["message"]["color"].GetString() : values.color = "0x0000";
  document["message"].HasMember("lane_position") ? values.lane_position = document["message"]["lane_position"].GetUint() : values.lane_position = 15;
  document.HasMember("signature") ? values.signature = document["signature"].GetString() : values.signature = "placeholder";
  document.HasMember("source_uuid") ? values.source_uuid = document["source_uuid"].GetString() : values.source_uuid = "placeholder";
  document.HasMember("message_id") ? values.message_id = document["message_id"].GetString() : values.message_id = "placeholder";

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
  (document.HasMember("signature")) ? (values.signature = document["signature"].GetString()) : (values.signature = "placeholder");
  (document.HasMember("source_uuid")) ? (values.source_uuid = document["source_uuid"].GetString()) : (values.source_uuid = "placeholder");
  (document.HasMember("message_id")) ? (values.message_id = document["message_id"].GetString()) : (values.message_id = "placeholder");


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
  (document.HasMember("signature")) ? (values.signature = document["signature"].GetString()) : (values.signature = "placeholder");
  (document.HasMember("source_uuid")) ? (values.source_uuid = document["source_uuid"].GetString()) : (values.source_uuid = "placeholder");
  (document.HasMember("message_id")) ? (values.message_id = document["message_id"].GetString()) : (values.message_id = "placeholder");

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
  (document.HasMember("signature")) ? (values.signature = document["signature"].GetString()) : (values.signature = "placeholder");
  (document.HasMember("source_uuid")) ? (values.source_uuid = document["source_uuid"].GetString()) : (values.source_uuid = "placeholder");
  (document.HasMember("message_id")) ? (values.message_id = document["message_id"].GetString()) : (values.message_id = "placeholder");

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
  (document.HasMember("signature")) ? (values.signature = document["signature"].GetString()) : (values.signature = "placeholder");
  (document.HasMember("source_uuid")) ? (values.source_uuid = document["source_uuid"].GetString()) : (values.source_uuid = "placeholder");
  (document.HasMember("message_id")) ? (values.message_id = document["message_id"].GetString()) : (values.message_id = "placeholder");

  return values;

}

int filterInput(Document document) {

  if(!(document.IsObject()) || !(document.HasMember("type")) || document.IsArray()){
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

  else if(document["type"] == NOTIFY_DELETE) {
      return 4;
  }

  else {
    return -1;
  }

}

string listenDataTCP(int socket_c) {



  char dataReceived[MAXIMUM_TRANSFER];
  memset(dataReceived,0,sizeof(dataReceived));
  // std::ofstream out("logs.txt");
  // auto *coutbuf = std::cout.rdbuf();
  // std::cout.rdbuf(out.rdbuf());

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
      // if(i>1) printf("Received %d bytes of data. Data received: %s\n",i,dataReceived);
      auto found = string(dataReceived).find("\n");
      if((found!=std::string::npos)){
            if (found + 1 != i){
              string copy_of_return = incomplete_message;
              // cout << "RETURNING" << copy_of_return + string(dataReceived).substr(i,found) << endl;
              incomplete_message = string(dataReceived).substr(found+1,i);
              write_to_log(copy_of_return + string(dataReceived).substr(0,found));
              return copy_of_return + string(dataReceived).substr(0,found);
            }else{
            // cout << "space at: " << found << "message length at: " << i << "message: " << dataReceived << endl;
            string copy_of_return = incomplete_message;
            incomplete_message = string();
            write_to_log(copy_of_return + string(dataReceived).substr(0,found+1));
            return copy_of_return + string(dataReceived).substr(0,found);
          }
        }else incomplete_message += string(dataReceived); // concatinating incomplete messages
        }
    }
    // std::cout.rdbuf(coutbuf);
  }
