#include <iostream>
#include <string>
#include <ostream>

using namespace std;

using std::cout;

class UnsubscriptionRequest {
private:
	string type;
	string context;
	string origin;
	string version = "1.0.0";
	uint64_t timestamp;
	int request_id;
	int subscriptionId;
	string signature;
	string source_uuid;
	string destination_uuid;

public:

	UnsubscriptionRequest(string type, string context, string origin, string version,
	uint64_t timestamp, int subscriptionId,int request_id, string signature, string source_uuid):
	type(type),
	context(context),
	origin(origin),
	version(version),
	timestamp(timestamp),
	subscriptionId(subscriptionId),
	signature(signature),
	request_id(request_id),
	source_uuid(source_uuid)
	{
		type = "unsubscription_request";
		context = "subscriptions";
		origin = "traffic_orchestrator";
		source_uuid = "OB19D";
		destination_uuid = "placeholder";
	}


	UnsubscriptionRequest() {
		type = "unsubscription_request";
		context = "subscriptions";
		origin = "traffic_orchestrator";
		source_uuid = "OB19D";
		destination_uuid = "placeholder";
	}

	friend std::ostream& operator<< (ostream& os, UnsubscriptionRequest * unsubscriptionReq);

	string getType();
	string getContext();
	string getOrigin();
	string getVersion();
	uint64_t getTimestamp();
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
	void setSubscriptionId(int);
	void setSignature(string);
	void setRequestId(int);
	void setSourceUUID(string);
	void setDestinationUUID(string);

};

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
