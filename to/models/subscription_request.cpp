// This script is a class that represents sending a subscription request to the v2x Gateway
// in order to start receiving road users

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#include "include/subscription_request.h"

SubscriptionRequest::~SubscriptionRequest() {}

std::string SubscriptionRequest::getType(){return type;}
std::string SubscriptionRequest::getContext(){return context;}
std::string SubscriptionRequest::getOrigin(){return origin;}
std::string SubscriptionRequest::getVersion(){return version;}
uint64_t SubscriptionRequest::getTimestamp(){return timestamp;}
bool SubscriptionRequest::getFilter(){return filter;}
std::string SubscriptionRequest::getShape(){return shape;};
std::string SubscriptionRequest::getSignature(){return signature;}
int SubscriptionRequest::getRequestId(){return request_id;}
std::string SubscriptionRequest::getSourceUUID(){return source_uuid;}
std::string SubscriptionRequest::getDestinationUUID(){return destination_uuid;}
std::string SubscriptionRequest::getMessageID(){return message_id;}
std::pair<int,int> SubscriptionRequest::getNorthEast(){return northeast;}
std::pair<int,int> SubscriptionRequest::getSouthWest(){return southwest;}

void SubscriptionRequest::setType(std::string parameter){type = parameter;}
void SubscriptionRequest::setContext(std::string parameter){context = parameter;}
void SubscriptionRequest::setOrigin(std::string parameter){origin = parameter;}
void SubscriptionRequest::setVersion(std::string parameter){version = parameter;}
void SubscriptionRequest::setTimestamp(uint64_t parameter){timestamp = parameter;}
void SubscriptionRequest::setFilter(bool parameter){filter = parameter;}
void SubscriptionRequest::setShape(std::string parameter){shape = parameter;}
void SubscriptionRequest::setSignature(std::string parameter){signature = parameter;}
void SubscriptionRequest::setRequestId(int parameter){request_id = parameter;}
void SubscriptionRequest::setSourceUUID(std::string parameter){source_uuid = parameter;}
void SubscriptionRequest::setDestinationUUID(std::string parameter){destination_uuid = parameter;}
void SubscriptionRequest::setMessageID(std::string parameter){message_id = parameter;}
void SubscriptionRequest::setNorthEast(std::pair<int,int> parameter){northeast = std::make_pair(parameter.first,parameter.second);}
void SubscriptionRequest::setSouthWest(std::pair<int,int> parameter){southwest = std::make_pair(parameter.first,parameter.second);}

std::ostream& operator<<(std::ostream& os, SubscriptionRequest * subscriptionReq) {

  os
  << "["
  << subscriptionReq->getType()
  << ","
  << subscriptionReq->getContext()
  << ","
  << subscriptionReq->getOrigin()
  << ","
  << subscriptionReq->getVersion()
  << ","
  << subscriptionReq->getTimestamp()
	<< ","
	<< subscriptionReq->getSourceUUID()
  << ","
	<< subscriptionReq->getDestinationUUID()
	<< ","
  << subscriptionReq->getFilter()
  << ","
	<<subscriptionReq->getRequestId()
	<< ","
  << subscriptionReq->getShape()
  << ","
  << subscriptionReq->getSignature()
  << "]\n";
  return os;

}
