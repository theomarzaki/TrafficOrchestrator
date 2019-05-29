// this script models the unsubscription response sent by the v2x gateway presented in class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_UNSUB_RESPONSE_H
#define TO_UNSUB_RESPONSE_H

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

#endif
