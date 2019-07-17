// This script is responsible for sending the respective messages from classes to the v2x gatway

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#include <logger.h>
#include "include/network_interface.h"

int SendInterface::m_socket = -999;
std::string SendInterface::connectionAddress;
int SendInterface::port;
std::string SendInterface::receiveAddress;
int SendInterface::receivePort;

std::string SendInterface::createSubscriptionRequestJSON(std::shared_ptr<SubscriptionRequest> subscriptionReq) {
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

std::string SendInterface::createUnsubscriptionRequestJSON(std::shared_ptr<UnsubscriptionRequest> unsubscriptionReq) {
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

std::string SendInterface::createManeuverJSON(std::shared_ptr<ManeuverRecommendation> maneuverRec) {

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
std::string SendInterface::createRUDDescription(std::shared_ptr<ManeuverRecommendation> maneuverRec) {

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

int SendInterface::sendTCP(std::string jsonString, bool newSocket) {
    struct sockaddr_in address, client_addr;

    jsonString += "\n";

    if (m_socket == -999 or newSocket) {
        m_socket = socket(AF_INET, SOCK_STREAM, 0);
        address.sin_addr.s_addr = inet_addr(SendInterface::connectionAddress.c_str());
        address.sin_family = AF_INET;
        address.sin_port = htons(SendInterface::port);

        memset(&client_addr, 0, sizeof(client_addr));
        client_addr.sin_family = AF_INET;
        client_addr.sin_port = htons(SendInterface::receivePort);
        client_addr.sin_addr.s_addr = inet_addr(SendInterface::receiveAddress.c_str());

        ::bind(m_socket, (struct sockaddr *) &client_addr, sizeof(client_addr));

        /* Connect to the remote server. */
        if (connect(m_socket, (struct sockaddr *) &address, sizeof(address)) < 0) {
            logger::write("[ERROR] Send: Connection Error.\n");
        }
    }
    if (m_socket == -1) {
        logger::write("[ERROR] Send: Socket was not created.");
    } else {
        // Send: Connected!
        // handle properly the SIGPIPE with MSG_NOSIGNAL
        if (send(m_socket, jsonString.c_str(), jsonString.size(), MSG_NOSIGNAL) < 0) {
            logger::write("[ERROR] Send failed.\n");
        }
    }
    return m_socket;
}
