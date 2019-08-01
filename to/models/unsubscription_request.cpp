// this script models the unsubscription response sent by the TO presented in class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#include "unsubscription_request.h"

std::string UnsubscriptionRequest::getType(){return type;}
std::string UnsubscriptionRequest::getContext(){return context;}
std::string UnsubscriptionRequest::getOrigin(){return origin;}
std::string UnsubscriptionRequest::getVersion(){return version;}
uint64_t UnsubscriptionRequest::getTimestamp(){return timestamp;}
int UnsubscriptionRequest::getSubscriptionId(){return subscriptionId;}
std::string UnsubscriptionRequest::getSignature(){return signature;}
int UnsubscriptionRequest::getRequestId(){return request_id;}
std::string UnsubscriptionRequest::getSourceUUID(){return source_uuid;}
std::string UnsubscriptionRequest::getDestinationUUID(){return destination_uuid;}

void UnsubscriptionRequest::setType(std::string parameter){type = std::move(parameter);}
void UnsubscriptionRequest::setContext(std::string parameter){context = std::move(parameter);}
void UnsubscriptionRequest::setOrigin(std::string parameter){origin = std::move(parameter);}
void UnsubscriptionRequest::setVersion(std::string parameter){version = std::move(parameter);}
void UnsubscriptionRequest::setTimestamp(uint64_t parameter){timestamp = parameter;}
void UnsubscriptionRequest::setSubscriptionId(int parameter){subscriptionId = parameter;}
void UnsubscriptionRequest::setSignature(std::string parameter){signature = std::move(parameter);}
void UnsubscriptionRequest::setRequestId(int parameter){request_id=parameter;}
void UnsubscriptionRequest::setSourceUUID(std::string parameter){source_uuid = std::move(parameter);}
void UnsubscriptionRequest::setDestinationUUID(std::string parameter){destination_uuid = std::move(parameter);}

std::ostream& operator<<(std::ostream& os, UnsubscriptionRequest * unsubscriptionReq) {

  os
  << "["
  << unsubscriptionReq->getType()
  << ","
  << unsubscriptionReq->getContext()
  << ","
  << unsubscriptionReq->getOrigin()
  << ","
  << unsubscriptionReq->getVersion()
  << ","
  << unsubscriptionReq->getTimestamp()
	<< ","
	<< unsubscriptionReq->getSourceUUID()
	<< ","
	<< unsubscriptionReq->getDestinationUUID()
	<< ","
	<< unsubscriptionReq->getRequestId()
  << ","
  << unsubscriptionReq->getSubscriptionId()
  << ","
  << unsubscriptionReq->getSignature()
  << "]\n";
  return os;

}
