// This script is responsible for sending the respective messages from classes to the v2x gatway

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#include <logger.h>
#include "network_interface.h"
#include "protocol.h"

#include <list>

bool NetworkInterface::connected = false;
int NetworkInterface::m_socket = -999;

std::string incomplete_message = std::string();
std::list<std::string> incomplete_messages;


bool NetworkInterface::connectTCP(std::string targetAddress, int targetPort, std::string receiveAddress, int receivePort) {
    struct sockaddr_in address, client_addr;

    if (m_socket < 0) {
        m_socket = socket(AF_INET, SOCK_STREAM, 0);
    }

    address.sin_addr.s_addr = inet_addr(targetAddress.c_str());
    address.sin_family = AF_INET;
    address.sin_port = htons(targetPort);

    memset(&client_addr, 0, sizeof(client_addr));
    client_addr.sin_family = AF_INET;
    client_addr.sin_port = htons(receivePort);
    client_addr.sin_addr.s_addr = inet_addr(receiveAddress.c_str());

    ::bind(m_socket, (struct sockaddr *) &client_addr, sizeof(client_addr));

    /* Connect to the remote server. */
    if (connect(m_socket, (struct sockaddr *) &address, sizeof(address)) < 0) {
        logger::write("[ERROR] Connection Attempt Failed.\n");
        m_socket = -999;
        connected = false;
    } else {
        logger::write("[INFO] Connection Acquire.\n");
        connected = true;
    }
    return connected;
}

bool NetworkInterface::sendTCP(std::string jsonString) {

    jsonString += "\n";

    if (m_socket < 0 or !connected) {
        logger::write("[ERROR] Network not connected");
        return false;
    } else {
        // Send: Connected!
        // handle properly the SIGPIPE with MSG_NOSIGNAL
        if (send(m_socket, jsonString.c_str(), jsonString.size(), MSG_NOSIGNAL) < 0) {
            logger::write("[ERROR] Send failed.\n");
            connected = false;
            return false;
        }
        return true;
    }
}

std::vector<std::string> NetworkInterface::listenDataTCP() {

    char dataReceived[MAXIMUM_TRANSFER];
    memset(dataReceived, 0, sizeof(dataReceived));
    std::string to_return;
    std::vector<std::string> returning;

    while (returning.empty()) {
        int i = read(NetworkInterface::m_socket, dataReceived, sizeof(dataReceived));

        if(i < 0) {
            logger::write("Error: Failed to receive transmitted data.\n");
            logger::write("trying to reconnect.\n");
            connected = false;
            return {};
        }
        else if(i == 0) {
            logger::write("Socket closed from the remote server.\n");
            connected = false;
            return {};
        }
        else if(i > 0) {
            for(int index = 0; index < i; index++){
                char chrc = dataReceived[index];
                if (chrc == '\n') {
                    to_return = incomplete_message;
                    incomplete_messages.push_back(to_return);
                    incomplete_message = std::string();
                } else {
                    incomplete_message += chrc;
                }
            }
        }

        while(!incomplete_messages.empty()) {
            auto message_packet = incomplete_messages.front();
            if(message_packet != "\n" and !message_packet.empty())
                logger::write(message_packet + "\n");
            returning.emplace_back(incomplete_messages.front());
            incomplete_messages.pop_front();
        }
    }
    return returning;
}