// This file detects the messages sent by the v2X Gatway, Parses the JSON
// Message to obtain the relevant message type for futher processing

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_DETECT_IFACE_H
#define TO_DETECT_IFACE_H

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
#include <list>

#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define NOTIFY_ADD "notify_add"
#define RECONNECT "RECONNECT"
#define HEART_BEAT "heart_beat"
#define NOTIFY_DELETE "notify_delete"
#define SUBSCRIPTION_RESPONSE "subscription_response"
#define UNSUBSCRIPTION_RESPONSE "unsubscription_response"
#define TRAJECTORY_FEEDBACK "maneuver_feedback"
#define MAXIMUM_TRANSFER 100000

typedef uint32_t uint4;

enum class message_type {
    notify_add, notify_delete, subscription_response, unsubscription_response, trajectory_feedback, unknown, heart_beat, reconnect
};

/* struct to represent fields found in JSON string. */
struct Detected_Road_User {

  std::string type;
  std::string context;
  std::string origin;
  std::string version;
  uint64_t timestamp;
  std::string uuid;
  std::string its_station_type;
  bool connected;
  int32_t latitude;
  int32_t longitude;
  std::string position_type;
  uint16_t heading;
  uint16_t speed;
  int16_t acceleration;
  int16_t yaw_rate;
  float length;
  float width;
  float height;
  std::string color;
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
  std::string signature;
  std::string message_id;
  std::string source_uuid;

};

struct Detected_To_Notification {
  std::string type;
  std::string context;
  std::string origin;
  std::string version;
  std::string uuid;
  std::string source_uuid;
  uint64_t timestamp;
  int subscriptionId;
  std::vector<Detected_Road_User> ru_description_list;
  std::string signature;
  std::string message_id;
};

struct Detected_Trajectory_Feedback {
  std::string type;
  std::string context;
  std::string origin;
  std::string version;
  uint64_t timestamp;
  std::string source_uuid;
  std::string uuid_vehicle;
  std::string uuid_to;
  std::string uuid_maneuver;
  uint64_t timestamp_message;
  std::string feedback;
  std::string reason;
  std::string signature;
  std::string message_id;

};

struct Detected_Subscription_Response {
  std::string type;
  std::string context;
  std::string origin;
  std::string version;
  uint64_t timestamp;
  std::string result;
  int request_id;
  std::string source_uuid;
  std::string destination_uuid;
  int subscriptionId;
  std::string signature;
  std::string message_id;

};

struct Detected_Unsubscription_Response {
  std::string type;
  std::string context;
  std::string origin;
  std::string version;
  uint64_t timestamp;
  std::string result;
  std::string source_uuid;
  std::string destination_uuid;
  int request_id;
  std::string signature;
  std::string message_id;
};

rapidjson::Document parse(std::string readFromServer);

Detected_Road_User assignRoadUserVals(rapidjson::Document &document);
std::vector<std::string> assignNotificationDeleteVals(rapidjson::Document &document);
Detected_To_Notification assignNotificationVals(rapidjson::Document &document);
Detected_Trajectory_Feedback assignTrajectoryFeedbackVals(rapidjson::Document &document);
Detected_Subscription_Response assignSubResponseVals(rapidjson::Document &document);
Detected_Unsubscription_Response assignUnsubResponseVals(rapidjson::Document &document);
message_type filterInput(rapidjson::Document &document);
std::vector<std::string> listenDataTCP(int socket_c);

#endif