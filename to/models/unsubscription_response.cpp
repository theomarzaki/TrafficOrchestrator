// this script models the unsubscription response sent by the v2x gateway presented in class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#include "unsubscription_response.h"
#include <utility>

std::string UnsubscriptionResponse::getType(){return type;}
std::string UnsubscriptionResponse::getContext(){return context;}
std::string UnsubscriptionResponse::getOrigin(){return origin;}
std::string UnsubscriptionResponse::getVersion(){return version;}
uint64_t UnsubscriptionResponse::getTimestamp(){return timestamp;}
std::string UnsubscriptionResponse::getResult(){return result;}
std::string UnsubscriptionResponse::getSignature(){return signature;}
int UnsubscriptionResponse::getRequestId(){return request_id;}
std::string UnsubscriptionResponse::getSouceUUID(){return source_uuid;}
std::string UnsubscriptionResponse::getDestinationUUID(){return destination_uuid;}


void UnsubscriptionResponse::setType(std::string parameter){type = std::move(parameter);}
void UnsubscriptionResponse::setContext(std::string parameter){context = std::move(parameter);}
void UnsubscriptionResponse::setOrigin(std::string parameter){origin = std::move(parameter);}
void UnsubscriptionResponse::setVersion(std::string parameter){version = std::move(parameter);}
void UnsubscriptionResponse::setTimestamp(uint64_t parameter){timestamp = parameter;}
void UnsubscriptionResponse::setResult(std::string parameter){result = std::move(parameter);}
void UnsubscriptionResponse::setSignature(std::string parameter){signature = std::move(parameter);}
void UnsubscriptionResponse::setRequestId(int parameter){request_id = parameter;}
void UnsubscriptionResponse::setSourceUUID(std::string parameter){source_uuid = std::move(parameter);}
void UnsubscriptionResponse::setDestinationUUID(std::string parameter){destination_uuid = std::move(parameter);}

std::ostream& operator<<(std::ostream& os, UnsubscriptionResponse * unsubscriptionResp) {

  os
  << "["
  << unsubscriptionResp->getType()
  << ","
  << unsubscriptionResp->getContext()
  << ","
  << unsubscriptionResp->getOrigin()
  << ","
  << unsubscriptionResp->getVersion()
  << ","
	<< unsubscriptionResp->getSouceUUID()
	<< ","
	<< unsubscriptionResp->getDestinationUUID()
	<< ","
  << unsubscriptionResp->getTimestamp()
  << ","
  << unsubscriptionResp->getResult()
	<< ","
	<< unsubscriptionResp->getRequestId()
  << ","
  << unsubscriptionResp->getSignature()
  << "]\n";
  return os;

}
