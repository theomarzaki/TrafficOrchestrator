 // this script models the subscription response sent by the v2x gateway presented in class structure

 // Created by: KCL

 // Modified by: Omar Nassef (KCL)
#include "include/subscription_response.h"

string SubscriptionResponse::getType(){return type;}
string SubscriptionResponse::getContext(){return context;}
string SubscriptionResponse::getOrigin(){return origin;}
string SubscriptionResponse::getVersion(){return version;}
uint64_t SubscriptionResponse::getTimestamp(){return timestamp;}
string SubscriptionResponse::getResult(){return result;}
int SubscriptionResponse::getSubscriptionId(){return subscriptionId;}
string SubscriptionResponse::getSignature(){return signature;}
int SubscriptionResponse::getRequestId(){return request_id;}
string SubscriptionResponse::getSourceUUID(){return source_uuid;}
string SubscriptionResponse::getDestinationUUID(){return destination_uuid;}

void SubscriptionResponse::setType(string parameter){type = parameter;}
void SubscriptionResponse::setContext(string parameter){context = parameter;}
void SubscriptionResponse::setOrigin(string parameter){origin = parameter;}
void SubscriptionResponse::setVersion(string parameter){version = parameter;}
void SubscriptionResponse::setTimestamp(uint64_t parameter){timestamp = parameter;}
void SubscriptionResponse::setResult(string parameter){result = parameter;}
void SubscriptionResponse::setSubscriptionId(int parameter){subscriptionId = parameter;}
void SubscriptionResponse::setSignature(string parameter){signature = parameter;}
void SubscriptionResponse::setRequestId(int parameter){request_id = parameter;}
void SubscriptionResponse::setSourceUUID(string parameter){source_uuid = parameter;}
void SubscriptionResponse::setDestinationUUID(string parameter){destination_uuid = parameter;}


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
