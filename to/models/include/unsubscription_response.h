// this script models the unsubscription response sent by the v2x gateway presented in class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_UNSUB_RESPONSE_H
#define TO_UNSUB_RESPONSE_H

#include <iostream>
#include <ostream>
#include <string>

class UnsubscriptionResponse {
private:
	std::string type;
	std::string context;
	std::string origin;
	std::string version;
	uint64_t timestamp;
	std::string result;
	std::string signature;
	int request_id;
	std::string source_uuid;
	std::string destination_uuid;

public:

	UnsubscriptionResponse(std::string type, std::string context, std::string origin, std::string version,
	uint64_t timestamp, std::string result,int request_id,std::string signature) :
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

	friend std::ostream& operator<< (std::ostream& os, UnsubscriptionResponse * unsubscriptionRes);

	std::string getType();
	std::string getContext();
	std::string getOrigin();
	std::string getVersion();
	uint64_t getTimestamp();
	std::string getResult();
	std::string getSignature();
	int getRequestId();
	std::string getSouceUUID();
	std::string getDestinationUUID();

	void setType(std::string);
	void setContext(std::string);
	void setOrigin(std::string);
	void setVersion(std::string);
	void setTimestamp(uint64_t);
	void setResult(std::string);
	void setSignature(std::string);
	void setRequestId(int);
	void setSourceUUID(std::string);
	void setDestinationUUID(std::string);

};

#endif
