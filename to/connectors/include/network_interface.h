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
#include <libnet.h>

#include <logger.h>

#include "maneuver_recommendation.h"
#include "subscription_request.h"
#include "unsubscription_request.h"
#include "waypoint.h"

#define MAXIMUM_TRANSFER 100000

using namespace rapidjson;

namespace SendInterface {

    static int pre_socket;
    static std::string connectionAddress;
    static int port;
    static std::string receiveAddress;
    static int receivePort;

    static std::string createSubscriptionRequestJSON(std::shared_ptr<SubscriptionRequest> subscriptionReq);
    static std::string createUnsubscriptionRequestJSON(std::shared_ptr<UnsubscriptionRequest> unsubscriptionReq);
    static std::string createManeuverJSON(std::shared_ptr<ManeuverRecommendation> maneuverRec);
    static int sendTCP(string jsonString);
};

#endif
