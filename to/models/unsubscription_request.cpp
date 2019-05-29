// this script models the unsubscription response sent by the TO presented in class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#include "include/unsubscription_request.h"

string UnsubscriptionRequest::getType(){return type;}
string UnsubscriptionRequest::getContext(){return context;}
string UnsubscriptionRequest::getOrigin(){return origin;}
string UnsubscriptionRequest::getVersion(){return version;}
uint64_t UnsubscriptionRequest::getTimestamp(){return timestamp;}
int UnsubscriptionRequest::getSubscriptionId(){return subscriptionId;}
string UnsubscriptionRequest::getSignature(){return signature;}
int UnsubscriptionRequest::getRequestId(){return request_id;}
string UnsubscriptionRequest::getSourceUUID(){return source_uuid;}
string UnsubscriptionRequest::getDestinationUUID(){return destination_uuid;}

void UnsubscriptionRequest::setType(string parameter){type = parameter;}
void UnsubscriptionRequest::setContext(string parameter){context = parameter;}
void UnsubscriptionRequest::setOrigin(string parameter){origin = parameter;}
void UnsubscriptionRequest::setVersion(string parameter){version = parameter;}
void UnsubscriptionRequest::setTimestamp(uint64_t parameter){timestamp = parameter;}
void UnsubscriptionRequest::setSubscriptionId(int parameter){subscriptionId = parameter;}
void UnsubscriptionRequest::setSignature(string parameter){signature = parameter;}
void UnsubscriptionRequest::setRequestId(int parameter){request_id=parameter;}
void UnsubscriptionRequest::setSourceUUID(string parameter){source_uuid = parameter;}
void UnsubscriptionRequest::setDestinationUUID(string parameter){destination_uuid = parameter;}

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
