 // this script models the subscription response sent by the v2x gateway presented in class structure

 // Created by: KCL

 // Modified by: Omar Nassef (KCL)
#ifndef TO_SUB_RESPONSE_H
#define TO_SUB_RESPONSE_H

#include <iostream>
#include <string>
#include <ostream>

class SubscriptionResponse {
	private:
		std::string type;
		std::string context;
		std::string origin;
		std::string version;
		uint64_t timestamp;
		std::string result;
		int subscriptionId;
		std::string signature;
		int request_id;
		std::string source_uuid;
		std::string destination_uuid;

	public:

		SubscriptionResponse(std::string type, std::string context, std::string origin, std::string version,
		uint64_t timestamp, std::string result, int subscriptionId, std::string signature,int request_id) :
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

	friend std::ostream& operator<< (std::ostream& os, SubscriptionResponse * subscriptionResp);

		std::string getType();
		std::string getContext();
		std::string getOrigin();
		std::string getVersion();
		uint64_t getTimestamp();
		std::string getResult();
		int getSubscriptionId();
		std::string getSignature();
		int getRequestId();
		std::string getSourceUUID();
		std::string getDestinationUUID();

		void setType(std::string);
		void setContext(std::string);
		void setOrigin(std::string);
		void setVersion(std::string);
		void setTimestamp(uint64_t);
		void setResult(std::string);
		void setSubscriptionId(int);
		void setSignature(std::string);
		void setRequestId(int);
		void setSourceUUID(std::string);
		void setDestinationUUID(std::string);

};

#endif
