// This script is responsible for sending the respective messages from classes to the v2x gatway

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_NET_IFACE_H
#define TO_NET_IFACE_H

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>
#include <arpa/inet.h>

#include <logger.h>

#include "maneuver_recommendation.h"
#include "subscription_request.h"
#include "unsubscription_request.h"
#include "waypoint.h"

#define MAXIMUM_TRANSFER 100000

using namespace rapidjson;

namespace SendInterface {

    extern int m_socket;
    extern std::string connectionAddress;
    extern int port;
    extern std::string receiveAddress;
    extern int receivePort;

    std::string createSubscriptionRequestJSON(std::shared_ptr<SubscriptionRequest> subscriptionReq);
    std::string createUnsubscriptionRequestJSON(std::shared_ptr<UnsubscriptionRequest> unsubscriptionReq);
    std::string createManeuverJSON(std::shared_ptr<ManeuverRecommendation> maneuverRec);

    std::string createRUDDescription(std::shared_ptr<ManeuverRecommendation> maneuverRec);

    int sendTCP(std::string jsonString, bool newSocket=false);
}

#endif
