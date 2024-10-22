// this script models the unsubscription response sent by the v2x gateway presented in class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)

#include <iostream>
#include <ostream>
#include <string>

using namespace std;
using std::string;

class UnsubscriptionResponse {
private:
	string type;
	string context;
	string origin;
	string version;
	uint64_t timestamp;
	string result;
	string signature;
	int request_id;
	string source_uuid;
	string destination_uuid;

public:

	UnsubscriptionResponse(string type, string context, string origin, string version,
	uint64_t timestamp, string result,int request_id,string signature) :
	type(type),
	context(context),
	origin(origin),
	version(version),
	timestamp(timestamp),
	result(result),
	signature(signature),
	request_id(request_id)
	{
		type = "unsubscription_response";
		context = "unsubscription_management";
		origin = "v2x gateway";
	}

	UnsubscriptionResponse() {
		type = "unsubscription_response";
		context = "unsubscription_management";
		origin = "v2x gateway";
	}

	friend std::ostream& operator<< (ostream& os, UnsubscriptionResponse * unsubscriptionRes);

	string getType();
	string getContext();
	string getOrigin();
	string getVersion();
	uint64_t getTimestamp();
	string getResult();
	string getSignature();
	int getRequestId();
	string getSouceUUID();
	string getDestinationUUID();

	void setType(string);
	void setContext(string);
	void setOrigin(string);
	void setVersion(string);
	void setTimestamp(uint64_t);
	void setResult(string);
	void setSignature(string);
	void setRequestId(int);
	void setSourceUUID(string);
	void setDestinationUUID(string);

};

string UnsubscriptionResponse::getType(){return type;}
string UnsubscriptionResponse::getContext(){return context;}
string UnsubscriptionResponse::getOrigin(){return origin;}
string UnsubscriptionResponse::getVersion(){return version;}
uint64_t UnsubscriptionResponse::getTimestamp(){return timestamp;}
string UnsubscriptionResponse::getResult(){return result;}
string UnsubscriptionResponse::getSignature(){return signature;}
int UnsubscriptionResponse::getRequestId(){return request_id;}
string UnsubscriptionResponse::getSouceUUID(){return source_uuid;}
string UnsubscriptionResponse::getDestinationUUID(){return destination_uuid;}


void UnsubscriptionResponse::setType(string parameter){type = parameter;}
void UnsubscriptionResponse::setContext(string parameter){context = parameter;}
void UnsubscriptionResponse::setOrigin(string parameter){origin = parameter;}
void UnsubscriptionResponse::setVersion(string parameter){version = parameter;}
void UnsubscriptionResponse::setTimestamp(uint64_t parameter){timestamp = parameter;}
void UnsubscriptionResponse::setResult(string parameter){result = parameter;}
void UnsubscriptionResponse::setSignature(string parameter){signature = parameter;}
void UnsubscriptionResponse::setRequestId(int parameter){request_id = parameter;}
void UnsubscriptionResponse::setSourceUUID(string parameter){source_uuid = parameter;}
void UnsubscriptionResponse::setDestinationUUID(string parameter){destination_uuid = parameter;}

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
