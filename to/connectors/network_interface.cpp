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
    Document document;

    document.SetObject();

    Document::AllocatorType &allocator = document.GetAllocator();

    Value timestamp(subscriptionReq->getTimestamp());

    if (!subscriptionReq->getFilter()) {
        document.AddMember("type", Value().SetString(subscriptionReq->getType().c_str(), allocator), allocator)
                .AddMember("context", Value().SetString(subscriptionReq->getContext().c_str(), allocator), allocator)
                .AddMember("origin", Value().SetString(subscriptionReq->getOrigin().c_str(), allocator), allocator)
                .AddMember("version", Value().SetString(subscriptionReq->getVersion().c_str(), allocator), allocator)
                .AddMember("timestamp", timestamp, allocator);

        Value message(kObjectType);
        Value filter(kObjectType);

        message.AddMember("filter", filter, allocator);
        message.AddMember("request_id", Value().SetInt(subscriptionReq->getRequestId()), allocator);

        document.AddMember("message", message, allocator)
                .AddMember("signature", Value().SetString(subscriptionReq->getSignature().c_str(), allocator), allocator);

    } else {
        document.AddMember("type", Value().SetString(subscriptionReq->getType().c_str(), allocator), allocator)
                .AddMember("context", Value().SetString(subscriptionReq->getContext().c_str(), allocator), allocator)
                .AddMember("origin", Value().SetString(subscriptionReq->getOrigin().c_str(), allocator), allocator)
                .AddMember("version", Value().SetString(subscriptionReq->getVersion().c_str(), allocator), allocator)
                .AddMember("timestamp", timestamp, allocator)
                .AddMember("source_uuid", Value().SetString(subscriptionReq->getSourceUUID().c_str(), allocator), allocator)
                .AddMember("destination_uuid", Value().SetString(subscriptionReq->getDestinationUUID().c_str(), allocator), allocator);

        Value object(kObjectType);
        Value objectTwo(kObjectType);
        Value objectThree(kObjectType);
        Value objectFour(kObjectType);
        Value objectFive(kObjectType);

        objectTwo.AddMember("latitude", Value().SetInt(subscriptionReq->getNorthEast().second), allocator)
                .AddMember("longitude", Value().SetInt(subscriptionReq->getNorthEast().first), allocator);
        objectFive.AddMember("latitude", Value().SetInt(subscriptionReq->getSouthWest().second), allocator).
                AddMember("longitude", Value().SetInt(subscriptionReq->getSouthWest().first), allocator);

        object.AddMember("shape", Value().SetString(subscriptionReq->getShape().c_str(), allocator), allocator);
        object.AddMember("northeast", objectTwo, allocator);
        object.AddMember("southwest", objectFive, allocator);
        objectFour.AddMember("area", object, allocator);
        objectThree.AddMember("filter", objectFour, allocator);
        objectThree.AddMember("request_id", Value().SetInt(subscriptionReq->getRequestId()), allocator);
        document.AddMember("message", objectThree, allocator);
        document.AddMember("signature", Value().SetString(subscriptionReq->getSignature().c_str(), allocator), allocator);

    }

    StringBuffer strbuf;
    /* Allocates memory buffer for writing the JSON string. */
    Writer<StringBuffer> writer(strbuf);
    document.Accept(writer);

    return strbuf.GetString();

}

std::string SendInterface::createUnsubscriptionRequestJSON(std::shared_ptr<UnsubscriptionRequest> unsubscriptionReq) {
    Document document;

    document.SetObject();

    Document::AllocatorType &allocator = document.GetAllocator();

    Value timestamp(unsubscriptionReq->getTimestamp());
    Value subscription_id(unsubscriptionReq->getSubscriptionId());

    document.AddMember("type", Value().SetString(unsubscriptionReq->getType().c_str(), allocator), allocator)
            .AddMember("context", Value().SetString(unsubscriptionReq->getContext().c_str(), allocator), allocator)
            .AddMember("origin", Value().SetString(unsubscriptionReq->getOrigin().c_str(), allocator), allocator)
            .AddMember("version", Value().SetString(unsubscriptionReq->getVersion().c_str(), allocator), allocator)
            .AddMember("timestamp", timestamp, allocator)
            .AddMember("source_uuid", Value().SetString(unsubscriptionReq->getSourceUUID().c_str(), allocator), allocator)
            .AddMember("destination_uuid", Value().SetString(unsubscriptionReq->getDestinationUUID().c_str(), allocator), allocator);

    Value object(kObjectType);

    object.AddMember("subscription_id", subscription_id, allocator);

    document.AddMember("message", object, allocator);

    document.AddMember("signature", Value().SetString(unsubscriptionReq->getSignature().c_str(), allocator), allocator);

    StringBuffer strbuf;
    /* Allocates memory buffer for writing the JSON string. */
    Writer<StringBuffer> writer(strbuf);
    document.Accept(writer);

    return strbuf.GetString();

}

std::string SendInterface::createManeuverJSON(std::shared_ptr<ManeuverRecommendation> maneuverRec) {

    Document document; // RapidJSON Document to build JSON message.
    document.SetObject();
    Document::AllocatorType &allocator = document.GetAllocator();

    /* Adds trajectory recommendation fields to the JSON document. */
    document.AddMember("type", Value().SetString(maneuverRec->getType().c_str(), allocator), allocator)
            .AddMember("context", Value().SetString(maneuverRec->getContext().c_str(), allocator), allocator)
            .AddMember("origin", Value().SetString(maneuverRec->getOrigin().c_str(), allocator), allocator)
            .AddMember("version", Value().SetString(maneuverRec->getVersion().c_str(), allocator), allocator)
            .AddMember("source_uuid", Value().SetString(maneuverRec->getSourceUUID().c_str(), allocator), allocator)
            .AddMember("destination_uuid", Value().SetString(maneuverRec->getUuidTo().c_str(), allocator), allocator)
            .AddMember("timestamp", Value().SetUint64(maneuverRec->getTimestamp()), allocator)
            .AddMember("message_id", Value().SetString(maneuverRec->getMessageID().c_str(), allocator), allocator);


    Value message(kObjectType);
    message.AddMember("uuid_maneuver", Value().SetString(maneuverRec->getUuidManeuver().c_str(), allocator), allocator);

    Value waypoints(kArrayType);

    for (auto waypoint : maneuverRec->getWaypoints()) {
        Value point(kObjectType);

        point.AddMember("timestamp", Value().SetUint64(waypoint->getTimestamp()), allocator);

        Value position(kObjectType);
        position.AddMember("latitude", Value().SetInt(waypoint->getLatitude()), allocator)
                .AddMember("longitude", Value().SetInt(waypoint->getLongitude()), allocator);

        point.AddMember("position", position, allocator)
                .AddMember("speed", Value().SetUint(waypoint->getSpeed()), allocator)
                .AddMember("lane_position", Value().SetUint(waypoint->getLanePosition()), allocator);


        waypoints.PushBack(point, allocator);
    }

    message.AddMember("waypoints", waypoints, allocator);

    Value action(kObjectType);

    // action.AddMember("timestamp", Value().SetUint64(maneuverRec->getTimestampAction()),allocator);
    //
    // Value action_position(kObjectType);
    //
    // action_position.AddMember("latitude", Value().SetInt(maneuverRec->getLatitudeAction()),allocator)
    // .AddMember("longitude",Value().SetInt(maneuverRec->getLongitudeAction()),allocator);
    //
    // action.AddMember("position", action_position, allocator)
    // .AddMember("speed", Value().SetUint(maneuverRec->getSpeedAction()), allocator)
    // .AddMember("lane_position", Value().SetUint(maneuverRec->getLanePositionAction()), allocator);

    // message.AddMember("action", action, allocator);

    document.AddMember("message", message, allocator);
    document.AddMember("signature", Value().SetString(maneuverRec->getSignature().c_str(), allocator), allocator);

    StringBuffer strbuf;
    Writer<StringBuffer> writer(strbuf);
    document.Accept(writer);

    return strbuf.GetString();

}

int SendInterface::sendTCP(std::string jsonString) {
    struct sockaddr_in address, client_addr;

    jsonString += "\n";

    if (m_socket == -999) {
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
