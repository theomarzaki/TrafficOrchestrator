// This file detects the messages sent by the v2X Gatway, Parses the JSON
// Message to obtain the relevant message type for futher processing

// Created by: KCL

// Modified by: Omar Nassef (KCL)

#include "protocol.h"
#include "logger.h"

std::string Protocol::createSubscriptionRequestJSON(std::shared_ptr<SubscriptionRequest> subscriptionReq) {
    rapidjson::Document document;

    document.SetObject();

    rapidjson::Document::AllocatorType &allocator = document.GetAllocator();

    rapidjson::Value timestamp(subscriptionReq->getTimestamp());

    if (!subscriptionReq->getFilter()) {
        document.AddMember("type", rapidjson::Value().SetString(subscriptionReq->getType().c_str(), allocator), allocator)
                .AddMember("context", rapidjson::Value().SetString(subscriptionReq->getContext().c_str(), allocator), allocator)
                .AddMember("origin", rapidjson::Value().SetString(subscriptionReq->getOrigin().c_str(), allocator), allocator)
                .AddMember("version", rapidjson::Value().SetString(subscriptionReq->getVersion().c_str(), allocator), allocator)
                .AddMember("timestamp", timestamp, allocator);

        rapidjson::Value message(rapidjson::kObjectType);
        rapidjson::Value filter(rapidjson::kObjectType);

        message.AddMember("filter", filter, allocator);
        message.AddMember("request_id", rapidjson::Value().SetInt(subscriptionReq->getRequestId()), allocator);

        document.AddMember("message", message, allocator)
                .AddMember("signature", rapidjson::Value().SetString(subscriptionReq->getSignature().c_str(), allocator), allocator);

    } else {
        document.AddMember("type", rapidjson::Value().SetString(subscriptionReq->getType().c_str(), allocator), allocator)
                .AddMember("context", rapidjson::Value().SetString(subscriptionReq->getContext().c_str(), allocator), allocator)
                .AddMember("origin", rapidjson::Value().SetString(subscriptionReq->getOrigin().c_str(), allocator), allocator)
                .AddMember("version", rapidjson::Value().SetString(subscriptionReq->getVersion().c_str(), allocator), allocator)
                .AddMember("timestamp", timestamp, allocator)
                .AddMember("source_uuid", rapidjson::Value().SetString(subscriptionReq->getSourceUUID().c_str(), allocator), allocator)
                .AddMember("destination_uuid", rapidjson::Value().SetString(subscriptionReq->getDestinationUUID().c_str(), allocator), allocator);

        rapidjson::Value object(rapidjson::kObjectType);
        rapidjson::Value objectTwo(rapidjson::kObjectType);
        rapidjson::Value objectThree(rapidjson::kObjectType);
        rapidjson::Value objectFour(rapidjson::kObjectType);
        rapidjson::Value objectFive(rapidjson::kObjectType);

        objectTwo.AddMember("latitude", rapidjson::Value().SetInt(subscriptionReq->getNorthEast().second), allocator)
                .AddMember("longitude", rapidjson::Value().SetInt(subscriptionReq->getNorthEast().first), allocator);
        objectFive.AddMember("latitude", rapidjson::Value().SetInt(subscriptionReq->getSouthWest().second), allocator).
                AddMember("longitude", rapidjson::Value().SetInt(subscriptionReq->getSouthWest().first), allocator);

        object.AddMember("shape", rapidjson::Value().SetString(subscriptionReq->getShape().c_str(), allocator), allocator);
        object.AddMember("northeast", objectTwo, allocator);
        object.AddMember("southwest", objectFive, allocator);
        objectFour.AddMember("area", object, allocator);
        objectThree.AddMember("filter", objectFour, allocator);
        objectThree.AddMember("request_id", rapidjson::Value().SetInt(subscriptionReq->getRequestId()), allocator);
        document.AddMember("message", objectThree, allocator);
        document.AddMember("signature", rapidjson::Value().SetString(subscriptionReq->getSignature().c_str(), allocator), allocator);

    }

    rapidjson::StringBuffer strbuf;
    /* Allocates memory buffer for writing the JSON string. */
    rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
    document.Accept(writer);

    return strbuf.GetString();

}

std::string Protocol::createUnsubscriptionRequestJSON(std::shared_ptr<UnsubscriptionRequest> unsubscriptionReq) {
    rapidjson::Document document;

    document.SetObject();

    rapidjson::Document::AllocatorType &allocator = document.GetAllocator();

    rapidjson::Value timestamp(unsubscriptionReq->getTimestamp());
    rapidjson::Value subscription_id(unsubscriptionReq->getSubscriptionId());

    document.AddMember("type", rapidjson::Value().SetString(unsubscriptionReq->getType().c_str(), allocator), allocator)
            .AddMember("context", rapidjson::Value().SetString(unsubscriptionReq->getContext().c_str(), allocator), allocator)
            .AddMember("origin", rapidjson::Value().SetString(unsubscriptionReq->getOrigin().c_str(), allocator), allocator)
            .AddMember("version", rapidjson::Value().SetString(unsubscriptionReq->getVersion().c_str(), allocator), allocator)
            .AddMember("timestamp", timestamp, allocator)
            .AddMember("source_uuid", rapidjson::Value().SetString(unsubscriptionReq->getSourceUUID().c_str(), allocator), allocator)
            .AddMember("destination_uuid", rapidjson::Value().SetString(unsubscriptionReq->getDestinationUUID().c_str(), allocator), allocator);

    rapidjson::Value object(rapidjson::kObjectType);

    object.AddMember("subscription_id", subscription_id, allocator);

    document.AddMember("message", object, allocator);

    document.AddMember("signature", rapidjson::Value().SetString(unsubscriptionReq->getSignature().c_str(), allocator), allocator);

    rapidjson::StringBuffer strbuf;
    /* Allocates memory buffer for writing the JSON string. */
    rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
    document.Accept(writer);

    return strbuf.GetString();

}

std::string Protocol::createManeuverJSON(std::shared_ptr<ManeuverRecommendation> maneuverRec) {

    rapidjson::Document document; // RapidJSON rapidjson::Document to build JSON message.
    document.SetObject();
    rapidjson::Document::AllocatorType &allocator = document.GetAllocator();

    /* Adds trajectory recommendation fields to the JSON document. */
    document.AddMember("type", rapidjson::Value().SetString(maneuverRec->getType().c_str(), allocator), allocator)
            .AddMember("context", rapidjson::Value().SetString(maneuverRec->getContext().c_str(), allocator), allocator)
            .AddMember("origin", rapidjson::Value().SetString(maneuverRec->getOrigin().c_str(), allocator), allocator)
            .AddMember("version", rapidjson::Value().SetString(maneuverRec->getVersion().c_str(), allocator), allocator)
            .AddMember("source_uuid", rapidjson::Value().SetString(maneuverRec->getSourceUUID().c_str(), allocator), allocator)
            .AddMember("destination_uuid", rapidjson::Value().SetString(maneuverRec->getUuidTo().c_str(), allocator), allocator)
            .AddMember("timestamp", rapidjson::Value().SetUint64(maneuverRec->getTimestamp()), allocator)
            .AddMember("message_id", rapidjson::Value().SetString(maneuverRec->getMessageID().c_str(), allocator), allocator);


    rapidjson::Value message(rapidjson::kObjectType);
    message.AddMember("uuid_maneuver", rapidjson::Value().SetString(maneuverRec->getUuidManeuver().c_str(), allocator), allocator);

    rapidjson::Value waypoints(rapidjson::kArrayType);

    for (auto waypoint : maneuverRec->getWaypoints()) {
        rapidjson::Value point(rapidjson::kObjectType);

        point.AddMember("timestamp", rapidjson::Value().SetUint64(waypoint->getTimestamp()), allocator);

        rapidjson::Value position(rapidjson::kObjectType);
        position.AddMember("latitude", rapidjson::Value().SetInt(waypoint->getLatitude()), allocator)
                .AddMember("longitude", rapidjson::Value().SetInt(waypoint->getLongitude()), allocator);

        point.AddMember("position", position, allocator)
                .AddMember("speed", rapidjson::Value().SetUint(waypoint->getSpeed()), allocator)
                .AddMember("lane_position", rapidjson::Value().SetUint(waypoint->getLanePosition()), allocator);


        waypoints.PushBack(point, allocator);
    }

    message.AddMember("waypoints", waypoints, allocator);

    rapidjson::Value action(rapidjson::kObjectType);

    // action.AddMember("timestamp", rapidjson::Value().SetUint64(maneuverRec->getTimestampAction()),allocator);
    //
    // rapidjson::Value action_position(rapidjson::kObjectType);
    //
    // action_position.AddMember("latitude", rapidjson::Value().SetInt(maneuverRec->getLatitudeAction()),allocator)
    // .AddMember("longitude",rapidjson::Value().SetInt(maneuverRec->getLongitudeAction()),allocator);
    //
    // action.AddMember("position", action_position, allocator)
    // .AddMember("speed", rapidjson::Value().SetUint(maneuverRec->getSpeedAction()), allocator)
    // .AddMember("lane_position", rapidjson::Value().SetUint(maneuverRec->getLanePositionAction()), allocator);

    // message.AddMember("action", action, allocator);

    document.AddMember("message", message, allocator);
    document.AddMember("signature", rapidjson::Value().SetString(maneuverRec->getSignature().c_str(), allocator), allocator);

    rapidjson::StringBuffer strbuf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
    document.Accept(writer);

    return strbuf.GetString();
}

// STUBY BUGZY YOU WILL CHANGE THE MESSAGE TO HELP ME
std::string Protocol::createRUDDescription(std::shared_ptr<ManeuverRecommendation> maneuverRec) {

    rapidjson::Document document; // RapidJSON rapidjson::Document to build JSON message.
    document.SetObject();
    rapidjson::Document::AllocatorType &allocator = document.GetAllocator();

    /* Adds trajectory recommendation fields to the JSON document. */
    document.AddMember("type", rapidjson::Value().SetString("ru_description", allocator), allocator)
            .AddMember("context", rapidjson::Value().SetString("general", allocator), allocator)
            .AddMember("origin", rapidjson::Value().SetString("self", allocator), allocator)
            .AddMember("version", rapidjson::Value().SetString(maneuverRec->getVersion().c_str(), allocator), allocator)
            .AddMember("source_uuid", rapidjson::Value().SetString(std::string(maneuverRec->getUuidTo()+"_rec").c_str(), allocator), allocator)
            .AddMember("timestamp", rapidjson::Value().SetUint64(maneuverRec->getTimestamp()), allocator)
            .AddMember("message_id", rapidjson::Value().SetString(maneuverRec->getMessageID().c_str(), allocator), allocator);

    rapidjson::Value message(rapidjson::kObjectType);
    message.AddMember("uuid", rapidjson::Value().SetString(std::string(maneuverRec->getUuidTo()+"_rec").c_str(), allocator), allocator)
            .AddMember("its_station_type", rapidjson::Value().SetString("passengerCar", allocator), allocator)
            .AddMember("connected", rapidjson::Value().SetBool(true), allocator)
            .AddMember("position_type", rapidjson::Value().SetString("gnss_raw_rtk", allocator), allocator)
            .AddMember("heading", rapidjson::Value().SetInt(maneuverRec->getWaypoints().at(0)->getHeading()), allocator)
            .AddMember("speed", rapidjson::Value().SetInt(maneuverRec->getWaypoints().at(0)->getSpeed()), allocator)
            .AddMember("lane_position", rapidjson::Value().SetUint(maneuverRec->getWaypoints().at(0)->getLanePosition()), allocator)
            .AddMember("acceleration", rapidjson::Value().SetUint(0), allocator)
            .AddMember("yaw_rate", rapidjson::Value().SetUint(0), allocator)
            .AddMember("raw_data", rapidjson::Value().SetBool(true), allocator)
            .AddMember("color", rapidjson::Value().SetString("0xFFFFFF", allocator), allocator)
            .AddMember("existence_probability", rapidjson::Value().SetUint(100), allocator);

    rapidjson::Value position(rapidjson::kObjectType);
    position.AddMember("latitude", rapidjson::Value().SetInt64(maneuverRec->getWaypoints().at(0)->getLatitude()), allocator)
            .AddMember("longitude", rapidjson::Value().SetInt64(maneuverRec->getWaypoints().at(0)->getLongitude()), allocator);
    message.AddMember("position", position, allocator);

    rapidjson::Value size(rapidjson::kObjectType);
    size.AddMember("length", rapidjson::Value().SetFloat(4), allocator)
            .AddMember("width", rapidjson::Value().SetFloat(2), allocator)
            .AddMember("height", rapidjson::Value().SetFloat(2), allocator);
    message.AddMember("size", size, allocator);

    rapidjson::Value accuracy(rapidjson::kObjectType);
    accuracy.AddMember("position_semi_major_confidence", rapidjson::Value().SetInt(2), allocator)
            .AddMember("position_semi_minor_confidence", rapidjson::Value().SetInt(2), allocator)
            .AddMember("position_semi_major_orientation", rapidjson::Value().SetInt(2), allocator)
            .AddMember("heading", rapidjson::Value().SetInt(2), allocator)
            .AddMember("speed", rapidjson::Value().SetInt(10), allocator)
            .AddMember("acceleration", rapidjson::Value().SetInt(2), allocator)
            .AddMember("yaw_rate", rapidjson::Value().SetInt(2), allocator);

    rapidjson::Value size2(rapidjson::kObjectType);
    size2.AddMember("length", rapidjson::Value().SetInt(1), allocator)
            .AddMember("width", rapidjson::Value().SetInt(1), allocator)
            .AddMember("height", rapidjson::Value().SetInt(1), allocator);
    accuracy.AddMember("size", size2, allocator);

    message.AddMember("accuracy", accuracy, allocator);

    document.AddMember("message", message, allocator);

    rapidjson::Value extra(rapidjson::kArrayType);
    document.AddMember("extra", extra, allocator);

    document.AddMember("signature", rapidjson::Value().SetString(maneuverRec->getSignature().c_str(), allocator), allocator);

    rapidjson::StringBuffer strbuf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
    document.Accept(writer);

    return strbuf.GetString();
}

rapidjson::Document    Protocol::parse(std::string readFromServer) {
  rapidjson::Document document;
  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  if(readFromServer == "RECONNECT"){
    document.SetObject();
    document.AddMember("type",rapidjson::Value() .SetString(std::string("RECONNECT").c_str(),allocator),allocator);
  }
  else if(readFromServer == "\n"){
    document.SetObject();
    document.AddMember("type",rapidjson::Value() .SetString(std::string("heart_beat").c_str(),allocator),allocator);
  }
  else{
    document.Parse(readFromServer.c_str());
  }
  return document;
}

Protocol::Detected_Road_User Protocol::assignRoadUserVals(rapidjson::Document &document) {
	Detected_Road_User values;
  // if(!(document.IsObject())){
  //   return values;
  // }
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

std::vector<std::string> Protocol::assignNotificationDeleteVals(rapidjson::Document    &document) {
  std::vector<std::string> values;
  if (document["message"].HasMember("ru_uuid_list") && document["message"]["ru_uuid_list"].IsArray()) {
    const rapidjson::Value &array = document["message"]["ru_uuid_list"].GetArray();
    for (rapidjson::SizeType i = 0; i < array.Size(); i++) {
      values.emplace_back(array[i].GetString());
    }
  } else {
    logger::write("error: notify_delete do not contain member ru_uuid_list.");
  }
  return values;
}

Protocol::Detected_To_Notification Protocol::assignNotificationVals(rapidjson::Document    &document) {
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
  logger::write("received notification with timestamp=" + std::to_string(values.timestamp));
  for(auto& v : document["message"]["ru_description_list"].GetArray()) {
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
    v.Accept(writer);
    rapidjson::Document ruDocument = parse(sb.GetString());
    const Detected_Road_User &ru = assignRoadUserVals(ruDocument);
    if (ru.timestamp == 0){
      logger::write("received invalid road user description in notify_add");
    } else {
      values.ru_description_list.push_back(ru);
    }
    sb.Clear();
    writer.Reset(sb);
  }


  return values;
}

Protocol::Detected_Trajectory_Feedback Protocol::assignTrajectoryFeedbackVals(rapidjson::Document    &document) {
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


Protocol::Detected_Subscription_Response Protocol::assignSubResponseVals(rapidjson::Document    &document) {

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

Protocol::Detected_Unsubscription_Response Protocol::assignUnsubResponseVals(rapidjson::Document    &document) {

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

Protocol::message_type Protocol::filterInput(rapidjson::Document    &document) {
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
    logger::write("error: received message with unknown type: " + std::string(document["type"].GetString()));
    return message_type::unknown;
  }
}
