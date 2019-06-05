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

std::string createSubscriptionRequestJSON(std::shared_ptr<SubscriptionRequest> subscriptionReq);

std::string createUnsubscriptionRequestJSON(std::shared_ptr<UnsubscriptionRequest> unsubscriptionReq);

/**
*
*	@description Uses a trajectory recommendation to write a JSON string
*	containing all the fields relating to that recommendation.
*
*	@param trajectoryRec is a pointer to a TrajectoryRecommendation.
*	@return strbuf.GetString() is the trajectory recommendation in JSON string format.
*/
std::string createManeuverJSON(std::shared_ptr<ManeuverRecommendation> maneuverRec);

int sendDataTCP(int pre_socket, std::string connectionAdress, int port, std::string receiveAddress, int receivePort, std::string jsonString);

#endif
