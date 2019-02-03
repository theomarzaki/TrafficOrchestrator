#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>

#include "maneuver_recommendation.cpp"
#include "subscription_request.cpp"
#include "unsubscription_request.cpp"

using namespace rapidjson;
using std::string;
using std::cout;

string createSubscriptionRequestJSON(SubscriptionRequest * subscriptionReq) {
	Document document;

	document.SetObject();

	Document::AllocatorType& allocator = document.GetAllocator();

	Value timestamp(subscriptionReq->getTimestamp());
	Value longitude(subscriptionReq->getLongitude());
	Value latitude(subscriptionReq->getLatitude());
	Value radius(subscriptionReq->getRadius());

	if(subscriptionReq->getFilter() == false) {
		document.AddMember("type",Value().SetString(subscriptionReq->getType().c_str(),allocator),allocator)
		.AddMember("context",Value().SetString(subscriptionReq->getContext().c_str(),allocator),allocator)
		.AddMember("origin",Value().SetString(subscriptionReq->getOrigin().c_str(),allocator),allocator)
		.AddMember("version",Value().SetString(subscriptionReq->getVersion().c_str(),allocator),allocator)
		.AddMember("timestamp",timestamp,allocator);

		Value message(kObjectType);
		Value filter(kObjectType);

		message.AddMember("filter", filter, allocator);
		message.AddMember("request_id",Value().SetInt(subscriptionReq->getRequestId()),allocator); // TODO CHANGED

		document.AddMember("message", message, allocator)
		.AddMember("signature",Value().SetString(subscriptionReq->getSignature().c_str(),allocator),allocator);

	} else if(subscriptionReq->getFilter() == true) {
		document.AddMember("type",Value().SetString(subscriptionReq->getType().c_str(),allocator),allocator)
		.AddMember("context",Value().SetString(subscriptionReq->getContext().c_str(),allocator),allocator)
		.AddMember("origin",Value().SetString(subscriptionReq->getOrigin().c_str(),allocator),allocator)
		.AddMember("version",Value().SetString(subscriptionReq->getVersion().c_str(),allocator),allocator)
		.AddMember("timestamp",timestamp,allocator)
		.AddMember("source_uuid",Value().SetString(subscriptionReq->getSourceUUID().c_str(),allocator),allocator)
		.AddMember("destination_uuid",Value().SetString(subscriptionReq->getDestinationUUID().c_str(),allocator),allocator);

		Value object(kObjectType);
		Value objectTwo(kObjectType);
		Value objectThree(kObjectType);
		Value objectFour(kObjectType);

		objectTwo.AddMember("latitude",latitude,allocator).AddMember("longitude",longitude,allocator);

		object.AddMember("shape", Value().SetString(subscriptionReq->getShape().c_str(), allocator),allocator);
		object.AddMember("center",objectTwo,allocator);
		object.AddMember("radius",radius,allocator);
		objectFour.AddMember("area", object, allocator);
		objectThree.AddMember("filter", objectFour, allocator);
		objectThree.AddMember("request_id",Value().SetInt(subscriptionReq->getRequestId()),allocator);
		document.AddMember("message", objectThree, allocator);
		document.AddMember("signature",Value().SetString(subscriptionReq->getSignature().c_str(),allocator),allocator);

	}

	StringBuffer strbuf;
	/* Allocates memory buffer for writing the JSON string. */
	Writer<StringBuffer> writer(strbuf);
	document.Accept(writer);

	return strbuf.GetString();

}

string createUnsubscriptionRequestJSON(UnsubscriptionRequest * unsubscriptionReq) {
	Document document;

	document.SetObject();

	Document::AllocatorType& allocator = document.GetAllocator();

	Value timestamp(unsubscriptionReq->getTimestamp());
	Value subscription_id(unsubscriptionReq->getSubscriptionId());

	document.AddMember("type",Value().SetString(unsubscriptionReq->getType().c_str(),allocator),allocator)
	.AddMember("context",Value().SetString(unsubscriptionReq->getContext().c_str(),allocator),allocator)
	.AddMember("origin",Value().SetString(unsubscriptionReq->getOrigin().c_str(),allocator),allocator)
	.AddMember("version",Value().SetString(unsubscriptionReq->getVersion().c_str(),allocator),allocator)
	.AddMember("timestamp",timestamp,allocator)
	.AddMember("source_uuid",Value().SetString(unsubscriptionReq->getSourceUUID().c_str(),allocator),allocator)
	.AddMember("destination_uuid",Value().SetString(unsubscriptionReq->getSourceUUID().c_str(),allocator),allocator);;

	Value object(kObjectType);

	object.AddMember("subscription_id", subscription_id, allocator);

	document.AddMember("message", object, allocator);

	document.AddMember("signature",Value().SetString(unsubscriptionReq->getSignature().c_str(),allocator),allocator);

	StringBuffer strbuf;
	/* Allocates memory buffer for writing the JSON string. */
	Writer<StringBuffer> writer(strbuf);
	document.Accept(writer);

	return strbuf.GetString();

}

/**
*
*	@description Uses a trajectory recommendation to write a JSON string
*	containing all the fields relating to that recommendation.
*
*	@param trajectoryRec is a pointer to a TrajectoryRecommendation.
*	@return strbuf.GetString() is the trajectory recommendation in JSON string format.
*/
string createManeuverJSON(ManeuverRecommendation * maneuverRec) {

	Document document; // RapidJSON Document to build JSON message.
	document.SetObject();
	Document::AllocatorType& allocator = document.GetAllocator();

	/* Adds trajectory recommendation fields to the JSON document. */
	document.AddMember("type", Value().SetString(maneuverRec->getType().c_str(), allocator),allocator)
	.AddMember("context", Value().SetString(maneuverRec->getContext().c_str(), allocator), allocator)
	.AddMember("origin", Value().SetString(maneuverRec->getOrigin().c_str(),allocator), allocator)
	.AddMember("version",Value().SetString(maneuverRec->getVersion().c_str(),allocator),allocator)
	.AddMember("source_uuid", Value().SetString(maneuverRec->getSourceUUID().c_str(),allocator), allocator)
	.AddMember("destination_uuid", Value().SetString(maneuverRec->getUuidTo().c_str(),allocator), allocator)
	.AddMember("timestamp", Value().SetUint64(maneuverRec->getTimestamp()), allocator);
	// .AddMember("message_id", Value().SetString(maneuverRec->getMessageID().c_str(),allocator), allocator);



	Value message(kObjectType);
	message.AddMember("uuid_maneuver", Value().SetString(maneuverRec->getUuidManeuver().c_str(),allocator), allocator);

	Value waypoints(kArrayType);

	for(Waypoint * waypoint : maneuverRec->getWaypoints()) {
		Value point(kObjectType);

		point.AddMember("timestamp", Value().SetUint64(waypoint->getTimestamp()), allocator);

		Value position(kObjectType);
		position.AddMember("latitude", Value().SetInt(waypoint->getLatitude()),allocator)
		.AddMember("longitude",Value().SetInt(waypoint->getLongitude()),allocator);

		point.AddMember("position", position, allocator)
		.AddMember("speed", Value().SetUint(waypoint->getSpeed()), allocator)
		.AddMember("lane_position", Value().SetUint(waypoint->getLanePosition()), allocator);

		waypoints.PushBack(point, allocator);
	}

	message.AddMember("waypoints", waypoints, allocator);

	Value action(kObjectType);

	// action.AddMember("timestamp", Value().SetUint64(maneuverRec->getTimestampAction()),allocator);
	//
	// Value action_position(kObjectType);
	//
	// action_position.AddMember("latitude", Value().SetInt(maneuverRec->getLatitudeAction()),allocator)
	// .AddMember("longitude",Value().SetInt(maneuverRec->getLongitudeAction()),allocator);
	//
	// action.AddMember("position", action_position, allocator)
	// .AddMember("speed", Value().SetUint(maneuverRec->getSpeedAction()), allocator)
	// .AddMember("lane_position", Value().SetUint(maneuverRec->getLanePositionAction()), allocator);

	// message.AddMember("action", action, allocator);

	document.AddMember("message", message, allocator);
	document.AddMember("signature", Value().SetString(maneuverRec->getSignature().c_str(),allocator), allocator);

	StringBuffer strbuf;
	Writer<StringBuffer> writer(strbuf);
	document.Accept(writer);

	return strbuf.GetString();

}

int sendDataTCP(int pre_socket,string connectionAdress, int port,string receiveAddress,int receivePort, string jsonString) {
	int socket_connect;
	int validator;
	struct sockaddr_in address,client_addr;

	jsonString = jsonString + "\n";

	/* Create a socket. */
	if(pre_socket == -999) socket_connect = socket(AF_INET, SOCK_STREAM,0);
	else socket_connect = pre_socket;


	if(socket_connect == -1)
		printf("Send: Socket was not created.");

	address.sin_addr.s_addr = inet_addr(connectionAdress.c_str());
	address.sin_family = AF_INET;
	address.sin_port = htons(port);

	memset(&client_addr, 0, sizeof(client_addr));
	client_addr.sin_family = AF_INET;
	client_addr.sin_port = htons(receivePort);
	client_addr.sin_addr.s_addr = inet_addr(receiveAddress.c_str());

	// SEND ERROR HERE FOR TR (NEW VERSION) --> remove int pre_socket

	::bind(socket_connect, (struct sockaddr *) &client_addr, sizeof(client_addr));

	/* Connect to the remote server. */
	validator = connect(socket_connect, (struct sockaddr *)&address, sizeof(address));

	// if(validator < 0)
	// 	printf("Send: Connection Error.\n");
	// else
	// printf("Send: Connected!\n");

	validator = send(socket_connect,jsonString.c_str(),jsonString.size(),0);

	if(validator < 0)
		printf("Send failed.\n");


		char dataReceived[MAXIMUM_TRANSFER];
	  memset(dataReceived,0,sizeof(dataReceived));

		return socket_connect;

}
