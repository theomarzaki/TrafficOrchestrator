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
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <unistd.h>
#include <netdb.h>

#include <logger.h>

#include "maneuver_recommendation.h"
#include "subscription_request.h"
#include "unsubscription_request.h"
#include "waypoint.h"

#define MAXIMUM_TRANSFER 100000

namespace NetworkInterface {

    extern bool connected;
    extern int m_socket;

    bool connectTCP(std::string targetAddress, int targetPort, std::string receiveAddress, int receivePort);
    bool sendTCP(std::string jsonString);
    std::vector<std::string> listenDataTCP();
}

#endif
