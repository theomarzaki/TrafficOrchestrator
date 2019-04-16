// This file detects the messages sent by the v2X Gatway, Parses the JSON
// Message to obtain the relevant message type for futher processing

// Created by: KCL

// Modified by: Omar Nassef (KCL)

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
#include <chrono>
#include <thread>

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

enum class message_type {
    notify_add, notify_delete, subscription_response, unsubscription_response, trajectory_feedback, unknown, heart_beat, reconnect
};

#define NOTIFY_ADD "notify_add"
#define RECONNECT "RECONNECT"
#define HEART_BEAT "heart_beat"
#define NOTIFY_DELETE "notify_delete"
#define SUBSCRIPTION_RESPONSE "subscription_response"
#define UNSUBSCRIPTION_RESPONSE "unsubscription_response"
#define TRAJECTORY_FEEDBACK "maneuver_feedback"
string incomplete_message = string();
#define MAXIMUM_TRANSFER 100000
std::list<string> incomplete_messages;

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
	std::ofstream log_file("logs/to_logs.log",std::ios_base::out | std::ios_base::app);
	log_file << text << endl;
}

Document parse(string readFromServer) {
  Document document;
  Document::AllocatorType& allocator = document.GetAllocator();
  if(readFromServer == "RECONNECT"){
    document.SetObject();
    document.AddMember("type",Value().SetString(string("RECONNECT").c_str(),allocator),allocator);
  }
  else if(readFromServer == "\n"){
    document.SetObject();
    document.AddMember("type",Value().SetString(string("heart_beat").c_str(),allocator),allocator);
  }
  else{
    document.Parse(readFromServer.c_str());
  }
  return document;
}

Detected_Road_User assignRoadUserVals(Document &document) {
	Detected_Road_User values;
  if(!(document.IsObject())){
    return values;
  }
  //TODO: use int instead of double when possible
  values.type = document["type"].GetString();
  document.HasMember("context") ? values.context = document["context"].GetString() : values.context = "placeholder";
  document.HasMember("origin") ? values.origin = document["origin"].GetString() : values.origin = "placeholder";
  document.HasMember("version") ? values.version = document["version"].GetString() : values.version = "placeholder";
  document.HasMember("timestamp") ? values.timestamp = document["timestamp"].GetUint64() : values.timestamp = 0;
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
  if (document["message"].HasMember("size")) {
    document["message"]["size"].HasMember("length") ? values.length = document["message"]["size"]["length"].GetDouble() : values.length = 0.0;
    document["message"]["size"].HasMember("width") ? values.width = document["message"]["size"]["width"].GetDouble() : values.width = 0.0;
    document["message"]["size"].HasMember("height") ? values.height = document["message"]["size"]["height"].GetDouble() : values.height = 0.0;
  }
  document["message"].HasMember("existence_probability") ? values.existence_probability = document["message"]["existence_probability"].GetUint() : values.existence_probability = -1;
  if (document["message"].HasMember("accuracy")) {
    document["message"]["accuracy"].HasMember("position_semi_major_confidence") ? values.position_semi_major_confidence = document["message"]["accuracy"]["position_semi_major_confidence"].GetUint() : values.position_semi_major_confidence = -1;
    document["message"]["accuracy"].HasMember("position_semi_minor_confidence") ? values.position_semi_minor_confidence = document["message"]["accuracy"]["position_semi_minor_confidence"].GetUint() : values.position_semi_minor_confidence = -1;
    document["message"]["accuracy"].HasMember("position_semi_major_orientation") ? values.position_semi_major_orientation = document["message"]["accuracy"]["position_semi_major_orientation"].GetUint() : values.position_semi_major_orientation = -1;
    document["message"]["accuracy"].HasMember("heading") ? values.heading_c = document["message"]["accuracy"]["heading"].GetUint() : values.heading_c = -1;
    document["message"]["accuracy"].HasMember("speed") ? values.speed_c = document["message"]["accuracy"]["speed"].GetUint() : values.speed_c = -1;
    document["message"]["accuracy"].HasMember("acceleration") ? values.acceleration_c = document["message"]["accuracy"]["acceleration"].GetUint() : values.acceleration_c = 0;
    document["message"]["accuracy"].HasMember("yaw_rate") ? values.yaw_rate_c = document["message"]["accuracy"]["yaw_rate"].GetUint() : values.yaw_rate_c = 0;
    if (document["message"]["accuracy"].HasMember("size")) {
      document["message"]["accuracy"]["size"].HasMember("length") ? values.length_c = document["message"]["accuracy"]["size"]["length"].GetDouble() : values.length_c = 0.0;
      document["message"]["accuracy"]["size"].HasMember("width") ? values.width_c = document["message"]["accuracy"]["size"]["width"].GetDouble() : values.width_c = 0.0;
      document["message"]["accuracy"]["size"].HasMember("height") ? values.height_c = document["message"]["accuracy"]["size"]["height"].GetDouble() : values.height_c = 0.0;
    }
  }
  document["message"].HasMember("color") ? values.color = document["message"]["color"].GetString() : values.color = "0x0000";
  document["message"].HasMember("lane_position") ? values.lane_position = document["message"]["lane_position"].GetUint() : values.lane_position = 15;
  document.HasMember("signature") ? values.signature = document["signature"].GetString() : values.signature = "placeholder";
  document.HasMember("source_uuid") ? values.source_uuid = document["source_uuid"].GetString() : values.source_uuid = "placeholder";
  document.HasMember("message_id") ? values.message_id = document["message_id"].GetString() : values.message_id = "placeholder";

  return values;

}

std::vector<string> assignNotificationDeleteVals(Document &document) {
  std::vector<string> values;
  if (document["message"].HasMember("ru_uuid_list") && document["message"]["ru_uuid_list"].IsArray()) {
    const rapidjson::Value &array = document["message"]["ru_uuid_list"].GetArray();
    for (rapidjson::SizeType i = 0; i < array.Size(); i++) {
      values.emplace_back(array[i].GetString());
    }
  } else {
    write_to_log("error: notify_delete do not contain member ru_uuid_list.");
  }
  return values;
}

Detected_To_Notification assignNotificationVals(Document &document) {
  Detected_To_Notification values;
  values.type = document["type"].GetString();
  document.HasMember("context") ? values.context = document["context"].GetString() : values.context = "placeholder";
  document.HasMember("origin") ? values.origin = document["origin"].GetString() : values.origin = "placeholder";
  document.HasMember("version") ? values.version = document["version"].GetString() : values.version = "placeholder";
  document.HasMember("timestamp") ? values.timestamp = document["timestamp"].GetUint64() : values.timestamp = -1;
  document["message"].HasMember("subscription_id") ? values.subscriptionId = document["message"]["subscription_id"].GetInt() : values.subscriptionId = 0;
  (document.HasMember("signature")) ? (values.signature = document["signature"].GetString()) : (values.signature = "placeholder");
  (document.HasMember("source_uuid")) ? (values.source_uuid = document["source_uuid"].GetString()) : (values.source_uuid = "placeholder");
  (document.HasMember("message_id")) ? (values.message_id = document["message_id"].GetString()) : (values.message_id = "placeholder");
  write_to_log("received notification with timestamp=" + to_string(values.timestamp));
  for(auto& v : document["message"]["ru_description_list"].GetArray()) {
    StringBuffer sb;
    Writer<StringBuffer> writer(sb);
    v.Accept(writer);
    Document ruDocument = parse(sb.GetString());
    const Detected_Road_User &ru = assignRoadUserVals(ruDocument);
    if (ru.timestamp == 0){
      write_to_log("received invalid road user description in notify_add");
    } else {
      values.ru_description_list.push_back(ru);
    }
    sb.Clear();
    writer.Reset(sb);
  }


  return values;
}

Detected_Trajectory_Feedback assignTrajectoryFeedbackVals(Document &document) {
  Detected_Trajectory_Feedback values;

  document.HasMember("type") ? values.type = document["type"].GetString() : values.type = "placeholder";
  document.HasMember("context") ? values.context = document["context"].GetString() : values.context = "placeholder";
  document.HasMember("origin") ? values.origin = document["origin"].GetString() : values.origin = "placeholder";
  document.HasMember("version") ? values.version = document["version"].GetString() : values.version = "placeholder";
  document.HasMember("timestamp") ? values.timestamp = document["timestamp"].GetUint64() : values.timestamp = -1;
  document.HasMember("source_uuid") ? values.uuid_vehicle = document["source_uuid"].GetString() : values.uuid_vehicle = "placeholder";
  document.HasMember("destination_uuid") ? values.uuid_to = document["destination_uuid"].GetString() : values.uuid_to = "placeholder";
  document["message"].HasMember("uuid_maneuver") ? values.uuid_maneuver = document["message"]["uuid_maneuver"].GetString() : values.uuid_maneuver = "placeholder";
  document["message"].HasMember("timestamp") ? values.timestamp_message = document["message"]["timestamp"].GetUint64() : values.timestamp_message = -1;
  document["message"].HasMember("feedback") ? values.feedback = document["message"]["feedback"].GetString() : values.feedback = "placeholder";
  document["message"].HasMember("reason") ? values.reason = document["message"]["reason"].GetString() : values.reason = "placeholder";
  (document.HasMember("signature")) ? (values.signature = document["signature"].GetString()) : (values.signature = "placeholder");
  (document.HasMember("message_id")) ? (values.message_id = document["message_id"].GetString()) : (values.message_id = "placeholder");

  return values;
}


Detected_Subscription_Response assignSubResponseVals(Document &document) {

  Detected_Subscription_Response values;

  document.HasMember("type") ? values.type = document["type"].GetString() : values.type = "placeholder";
  document.HasMember("context") ? values.context = document["context"].GetString() : values.context = "placeholder";
  document.HasMember("origin") ? values.origin = document["origin"].GetString() : values.origin = "placeholder";
  document.HasMember("version") ? values.version = document["version"].GetString() : values.version = "placeholder";
  document.HasMember("timestamp") ? values.timestamp = document["timestamp"].GetUint64() : values.timestamp = -1;
  document.HasMember("result") ? values.result = document["result"].GetString() : values.result = "placeholder";
  document["message"].HasMember("request_id") ? values.request_id = document["message"]["request_id"].GetInt() : values.request_id = -1;
  document["message"].HasMember("subscription_id") ? values.subscriptionId = document["message"]["subscription_id"].GetInt() : values.subscriptionId = -1;
  (document.HasMember("signature")) ? (values.signature = document["signature"].GetString()) : (values.signature = "placeholder");
  (document.HasMember("source_uuid")) ? (values.source_uuid = document["source_uuid"].GetString()) : (values.source_uuid = "placeholder");
  (document.HasMember("message_id")) ? (values.message_id = document["message_id"].GetString()) : (values.message_id = "placeholder");

  return values;

}

Detected_Unsubscription_Response assignUnsubResponseVals(Document &document) {

  Detected_Unsubscription_Response values;

  document.HasMember("type") ? values.type = document["type"].GetString() : values.type = "placeholder";
  document.HasMember("context") ? values.context = document["context"].GetString() : values.context = "placeholder";
  document.HasMember("origin") ? values.origin = document["origin"].GetString() : values.origin = "placeholder";
  document.HasMember("version") ? values.version = document["version"].GetString() : values.version = "placeholder";
  document.HasMember("timestamp") ? values.timestamp = document["timestamp"].GetUint64() : values.timestamp = -1;
  document["message"].HasMember("result") ? values.result = document["message"]["result"].GetInt() : values.result = -1;
  document["message"].HasMember("request_id") ? values.request_id = document["message"]["request_id"].GetInt() : values.request_id = -1;
  (document.HasMember("signature")) ? (values.signature = document["signature"].GetString()) : (values.signature = "placeholder");
  (document.HasMember("source_uuid")) ? (values.source_uuid = document["source_uuid"].GetString()) : (values.source_uuid = "placeholder");
  (document.HasMember("message_id")) ? (values.message_id = document["message_id"].GetString()) : (values.message_id = "placeholder");

  return values;

}

message_type filterInput(Document &document) {
  if(!(document.IsObject())){
    return message_type::unknown;
  }
  if(document["type"] == NOTIFY_ADD) {
    return message_type::notify_add;
  }

  else if(document["type"] == RECONNECT) {
    return message_type::reconnect;
  }

  else if(document["type"] == HEART_BEAT) {
    return message_type::heart_beat;
  }

  else if(document["type"] == SUBSCRIPTION_RESPONSE) {
    return message_type::subscription_response;
  }

  else if(document["type"] == UNSUBSCRIPTION_RESPONSE) {
    return message_type::unsubscription_response;
  }

  else if(document["type"] == TRAJECTORY_FEEDBACK) {
    return message_type::trajectory_feedback;
  }

  else if(document["type"] == NOTIFY_DELETE) {
      return message_type::notify_delete;
  }
  else {
    write_to_log("error: received message with unknown type: " + string(document["type"].GetString()));
    return message_type::unknown;
  }

}

string listenDataTCP(int socket_c) {

  char dataReceived[MAXIMUM_TRANSFER];
  memset(dataReceived, 0, sizeof(dataReceived));
  string to_return,returning;

  while (returning.empty()) {
    int i = read(socket_c, dataReceived, sizeof(dataReceived));

    if (i < 0) {
      write_to_log("Error: Failed to receive transmitted data.\n");
      break;
    } else if (i == 0) {
      printf("Socket closed from the remote server.\n");
      returning = "RECONNECT";
    } else if (i > 0) {
      for (int index = 0; index < i; index++) {
        char chrc = dataReceived[index];
        if (chrc == '\n') {
          to_return = incomplete_message;
          incomplete_messages.push_back(to_return);
          incomplete_message = string();
        } else {
          incomplete_message += chrc;
        }
      }
    }

    if (!incomplete_messages.empty()) {
      returning = incomplete_messages.front();
      write_to_log(incomplete_messages.front() + "\n");
      incomplete_messages.pop_front();
    }
  }
  return returning;
}
