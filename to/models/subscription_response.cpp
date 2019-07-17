 // this script models the subscription response sent by the v2x gateway presented in class structure

 // Created by: KCL

 // Modified by: Omar Nassef (KCL)
#include "include/subscription_response.h"

std::string SubscriptionResponse::getType(){return type;}
std::string SubscriptionResponse::getContext(){return context;}
std::string SubscriptionResponse::getOrigin(){return origin;}
std::string SubscriptionResponse::getVersion(){return version;}
uint64_t SubscriptionResponse::getTimestamp(){return timestamp;}
std::string SubscriptionResponse::getResult(){return result;}
int SubscriptionResponse::getSubscriptionId(){return subscriptionId;}
std::string SubscriptionResponse::getSignature(){return signature;}
int SubscriptionResponse::getRequestId(){return request_id;}
std::string SubscriptionResponse::getSourceUUID(){return source_uuid;}
std::string SubscriptionResponse::getDestinationUUID(){return destination_uuid;}

void SubscriptionResponse::setType(std::string parameter){type = parameter;}
void SubscriptionResponse::setContext(std::string parameter){context = parameter;}
void SubscriptionResponse::setOrigin(std::string parameter){origin = parameter;}
void SubscriptionResponse::setVersion(std::string parameter){version = parameter;}
void SubscriptionResponse::setTimestamp(uint64_t parameter){timestamp = parameter;}
void SubscriptionResponse::setResult(std::string parameter){result = parameter;}
void SubscriptionResponse::setSubscriptionId(int parameter){subscriptionId = parameter;}
void SubscriptionResponse::setSignature(std::string parameter){signature = parameter;}
void SubscriptionResponse::setRequestId(int parameter){request_id = parameter;}
void SubscriptionResponse::setSourceUUID(std::string parameter){source_uuid = parameter;}
void SubscriptionResponse::setDestinationUUID(std::string parameter){destination_uuid = parameter;}


std::ostream& operator<<(std::ostream& os, SubscriptionResponse * subscriptionResp) {

  os
  << "["
  << subscriptionResp->getType()
  << ","
  << subscriptionResp->getContext()
  << ","
  << subscriptionResp->getOrigin()
  << ","
  << subscriptionResp->getVersion()
  << ","
	<< subscriptionResp->getSourceUUID()
	<< ","
	<< subscriptionResp->getDestinationUUID()
	<< ","
  << subscriptionResp->getTimestamp()
  << ","
  << subscriptionResp->getResult()
  << ","
	<< subscriptionResp->getRequestId()
	<< ","
  << subscriptionResp->getSubscriptionId()
  << ","
  << subscriptionResp->getSignature()
  << "]\n";
  return os;

}
