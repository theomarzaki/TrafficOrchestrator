 // this script models the subscription response sent by the v2x gateway presented in class structure

 // Created by: KCL

 // Modified by: Omar Nassef (KCL)
#ifndef TO_SUB_RESPONSE_H
#define TO_SUB_RESPONSE_H

#include <iostream>
#include <string>
#include <ostream>

using namespace std;
using std::string;

class SubscriptionResponse {
	private:
		string type;
		string context;
		string origin;
		string version;
		uint64_t timestamp;
		string result;
		int subscriptionId;
		string signature;
		int request_id;
		string source_uuid;
		string destination_uuid;

	public:

		SubscriptionResponse(string type, string context, string origin, string version,
		uint64_t timestamp, string result, int subscriptionId, string signature,int request_id) :
		type(type),
		context(context),
		origin(origin),
		version(version),
		timestamp(timestamp),
		result(result),
		subscriptionId(subscriptionId),
		signature(signature),
		request_id(request_id)
		{
			type = "subscription_response";
			context = "subscriptions";
		}

		SubscriptionResponse() {
			type = "subscriptions";
			context = "subscriptions";
		}

	friend std::ostream& operator<< (ostream& os, SubscriptionResponse * subscriptionResp);

		string getType();
		string getContext();
		string getOrigin();
		string getVersion();
		uint64_t getTimestamp();
		string getResult();
		int getSubscriptionId();
		string getSignature();
		int getRequestId();
		string getSourceUUID();
		string getDestinationUUID();

		void setType(string);
		void setContext(string);
		void setOrigin(string);
		void setVersion(string);
		void setTimestamp(uint64_t);
		void setResult(string);
		void setSubscriptionId(int);
		void setSignature(string);
		void setRequestId(int);
		void setSourceUUID(string);
		void setDestinationUUID(string);

};

#endif
