#include <iostream>
#include <string>
#include <ostream>

using namespace std;
using std::cout;
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
			origin = "injector";
			version = "0.1.0";
		}

		SubscriptionResponse() {
			type = "subscriptions";
			context = "subscriptions";
			origin = "injector";
			version = "0.1.0";
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

		void setType(string);
		void setContext(string);
		void setOrigin(string);
		void setVersion(string);
		void setTimestamp(uint64_t);
		void setResult(string);
		void setSubscriptionId(int);
		void setSignature(string);
		void setRequestId(int);
};


string SubscriptionResponse::getType(){return type;}
string SubscriptionResponse::getContext(){return context;}
string SubscriptionResponse::getOrigin(){return origin;}
string SubscriptionResponse::getVersion(){return version;}
uint64_t SubscriptionResponse::getTimestamp(){return timestamp;}
string SubscriptionResponse::getResult(){return result;}
int SubscriptionResponse::getSubscriptionId(){return subscriptionId;}
string SubscriptionResponse::getSignature(){return signature;}
int SubscriptionResponse::getRequestId(){return request_id;}

void SubscriptionResponse::setType(string parameter){type = parameter;}
void SubscriptionResponse::setContext(string parameter){context = parameter;}
void SubscriptionResponse::setOrigin(string parameter){origin = parameter;}
void SubscriptionResponse::setVersion(string parameter){version = parameter;}
void SubscriptionResponse::setTimestamp(uint64_t parameter){timestamp = parameter;}
void SubscriptionResponse::setResult(string parameter){result = parameter;}
void SubscriptionResponse::setSubscriptionId(int parameter){subscriptionId = parameter;}
void SubscriptionResponse::setSignature(string parameter){signature = parameter;}
void SubscriptionResponse::setRequestId(int parameter){request_id = parameter;}

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
