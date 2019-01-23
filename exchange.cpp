#include <torch/torch.h>
#include <torch/script.h>
#include "detection_interface.cpp"
#include "database.cpp"
#include "maneuver_feedback.cpp"
#include "network_interface.cpp"
#include "subscription_response.cpp"
#include "Utils.cpp"
#include "unsubscription_response.cpp"
#include "trajectory_calculator.cpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "rapidjson/document.h"
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

using namespace rapidjson;

using namespace std::chrono;
using std::cout;

Database * database;
SubscriptionResponse * subscriptionResp;
UnsubscriptionResponse * unsubscriptionResp;
ManeuverFeedback * maneuverFeed;

string sendAddress;
int sendPort;
string receiveAddress;
int receivePort;

double distanceRadius;
uint32_t mergingLongitude;
uint32_t mergingLatitude;
string uuidTo;
bool filter = true;

RoadUser * detectedToRoadUserList(vector<Detected_Road_User> v) {

	cout << "Detected number of RoadUsers: " << v.size() <<".\n";

	RoadUser * road_users = new RoadUser[v.size()];

	for(int i = 0; i < v.size(); i++) {

		Detected_Road_User d = v[i];

		RoadUser * roadUser = new RoadUser(); // Declares and initalises a RoadUser pointer.
		roadUser->setType(d.type);
		roadUser->setContext(d.context);
		roadUser->setOrigin(d.origin);
		roadUser->setVersion(d.version);
		roadUser->setTimestamp(d.timestamp);
		roadUser->setUuid(d.uuid);
		roadUser->setItsStationType(d.its_station_type);
		roadUser->setConnected(d.connected);
		roadUser->setLatitude(d.latitude);
		roadUser->setLongitude(d.longitude);
		roadUser->setPositionType(d.position_type);
		roadUser->setHeading(d.heading);
		roadUser->setSpeed(d.speed);
		roadUser->setAcceleration(d.acceleration);
		roadUser->setYawRate(d.yaw_rate);
		roadUser->setLength(d.length);
		roadUser->setWidth(d.width);
		roadUser->setHeight(d.height);
		roadUser->setColor(d.color);
		roadUser->setLanePosition(d.lane_position);
		roadUser->setExistenceProbability(d.existence_probability);
		roadUser->setPositionSemiMajorConfidence(d.position_semi_major_confidence);
		roadUser->setPositionSemiMinorConfidence(d.position_semi_minor_confidence);
		roadUser->setPositionSemiMajorOrientation(d.position_semi_major_orientation);
		roadUser->setHeadingConfidence(d.heading_c);
		roadUser->setSpeedConfidence(d.speed_c);
		roadUser->setAccelerationConfidence(d.acceleration_c);
		roadUser->setYawRateConfidence(d.yaw_rate_c);
		roadUser->setLengthConfidence(d.length_c);
		roadUser->setWidthConfidence(d.width_c);
		roadUser->setHeightConfidence(d.height_c);
		roadUser->setSignature(d.signature);

		road_users[i] = *roadUser;

	}

	return road_users;

}

ManeuverFeedback * detectedToFeedback(Detected_Trajectory_Feedback d) {

	ManeuverFeedback * maneuverFeed = new ManeuverFeedback();
	maneuverFeed->setType(d.type);
	maneuverFeed->setContext(d.context);
	maneuverFeed->setOrigin(d.origin);
	maneuverFeed->setVersion(d.version);
	maneuverFeed->setTimestamp(d.timestamp);
	maneuverFeed->setUuidVehicle(d.uuid_vehicle);
	maneuverFeed->setUuidTo(d.uuid_to);
	maneuverFeed->setUuidManeuver(d.uuid_maneuver);
	maneuverFeed->setTimestampMessage(d.timestamp_message);
	maneuverFeed->setFeedback(d.feedback);
	maneuverFeed->setReason(d.reason);
	maneuverFeed->setSignature(d.signature);

	return maneuverFeed;

}

SubscriptionResponse * detectedToSubscription(Detected_Subscription_Response d) {
	SubscriptionResponse * subscriptionResp = new SubscriptionResponse();
	subscriptionResp->setType(d.type);
	subscriptionResp->setContext(d.context);
	subscriptionResp->setOrigin(d.origin);
	subscriptionResp->setVersion(d.version);
	subscriptionResp->setTimestamp(d.timestamp);
	subscriptionResp->setResult(d.result);
	subscriptionResp->setRequestId(d.request_id);
	subscriptionResp->setSubscriptionId(d.subscriptionId);
	subscriptionResp->setSignature(d.signature);
	subscriptionResp->setSourceUUID(d.source_uuid);
	subscriptionResp->setDestinationUUID(d.destination_uuid);

	return subscriptionResp;
}


UnsubscriptionResponse * detectedToUnsubscription(Detected_Unsubscription_Response d) {
	UnsubscriptionResponse * unsubscriptionResp = new UnsubscriptionResponse();
	unsubscriptionResp->setType(d.type);
	unsubscriptionResp->setContext(d.context);
	unsubscriptionResp->setOrigin(d.origin);
	unsubscriptionResp->setVersion(d.version);
	unsubscriptionResp->setTimestamp(d.timestamp);
	unsubscriptionResp->setRequestId(d.request_id);
	unsubscriptionResp->setResult(d.result);
	unsubscriptionResp->setSignature(d.signature);
	unsubscriptionResp->setSourceUUID(d.source_uuid);
	unsubscriptionResp->setDestinationUUID(d.destination_uuid);

	return unsubscriptionResp;
}

void generateUuidTo() {
	uuidTo = to_string(10000000 + ( std::rand() % ( 99999999 - 10000000 + 1 )));
}

int generateReqID(){
	uuidTo = to_string(10000000 + ( std::rand() % ( 99999999 - 10000000 + 1 )));
	stringstream geek(uuidTo);
	int x = 0;
	geek >> x;
	return x;
}


int initiateSubscription(string sendAddress, int sendPort,string receiveAddress,int receivePort, bool filter,int radius,uint32_t longitude, uint32_t latitude) {
	milliseconds timeSub = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	SubscriptionRequest * subscriptionReq = new SubscriptionRequest();
	subscriptionReq->setFilter(filter);
	subscriptionReq->setRadius(radius);
	subscriptionReq->setLongitude(longitude);
	subscriptionReq->setLatitude(latitude);
	subscriptionReq->setShape("circle");
	subscriptionReq->setSignature("TEMPLATE");
	subscriptionReq->setRequestId(generateReqID());
	subscriptionReq->setTimestamp(timeSub.count());
	auto socket = sendDataTCP(-999,sendAddress,sendPort,receiveAddress,receivePort,createSubscriptionRequestJSON(subscriptionReq));
	return socket;
}

void initiateUnsubscription(string sendAddress, int sendPort, SubscriptionResponse * subscriptionResp) {

	milliseconds timeUnsub = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	UnsubscriptionRequest * unsubscriptionReq = new UnsubscriptionRequest();
	unsubscriptionReq->setSubscriptionId(subscriptionResp->getSubscriptionId());
	unsubscriptionReq->setTimestamp(timeUnsub.count());
	sendDataTCP(-999,sendAddress,sendPort,receiveAddress,receivePort,createUnsubscriptionRequestJSON(unsubscriptionReq));
}

int filterExecution(string data) {
	int filterNum = filterInput(parse(data));

	if(filterNum == -1) {
		cout << "Error: Incomplete Message.";
		return -1;
	}
	else if(filterNum == 0) {
		int size = assignNotificationVals(parse(data)).ru_description_list.size();
		RoadUser * road_users = detectedToRoadUserList(assignNotificationVals(parse(data)).ru_description_list);
		for(int j = 0; j < size; j++) {
			database->insertRoadUser(&road_users[j]);
		}
		return 0;
	}
	else if(filterNum == 4) {
		int size = assignNotificationVals(parse(data)).ru_description_list.size();
		RoadUser * road_users = detectedToRoadUserList(assignNotificationVals(parse(data)).ru_description_list);
		for(int j = 0; j < size; j++) {
			database->deleteRoadUser(road_users[j].getUuid());
		}
		return 0;
	}
	else if(filterNum == 1) {
		subscriptionResp = detectedToSubscription(assignSubResponseVals(parse(data)));
		return 1;
	}
	else if (filterNum == 2) {
		unsubscriptionResp = detectedToUnsubscription(assignUnsubResponseVals(parse(data)));
		return 2;
	}
	else if (filterNum == 3) {
		maneuverFeed = detectedToFeedback(assignTrajectoryFeedbackVals(parse(data)));
		if(maneuverFeed->getFeedback() == "refuse" || maneuverFeed->getFeedback() == "abort") {
			return 3;
		}
		return -2;
	}

	else {
		return -100;
	}

}

void inputSendAddress(string address) {
	sendAddress = address;
}

void inputSendPort(int port) {
	sendPort = port;
}

void inputReceivePort(int port) {
	receivePort = port;
}

void inputReceiveAddress(string address) {
	receiveAddress = address;
}

void inputMergeLocation(uint32_t longt, uint32_t lat){
	mergingLongitude = longt;
	mergingLatitude = lat;
}

void inputDistanceRadius(int radius) {
	distanceRadius = radius;
}

void sendTrajectoryRecommendations(vector<ManeuverRecommendation*> v,int socket) {
	for(ManeuverRecommendation * m : v) {
		cout << createManeuverJSON(m) << endl;
		sendDataTCP(socket,sendAddress, sendPort,receiveAddress,receivePort, createManeuverJSON(m));
	}
}

void initaliseDatabase() {
	database = new Database();
}

int main() {

	FILE* file = fopen("../include/TO_config.json", "r");
	if(file == 0) {
    std::cout << "Config File failed to load." << std::endl;
	}

	std::shared_ptr<torch::jit::script::Module> lstm_model = torch::jit::load("../include/lstm_model.pt");

  if(lstm_model != nullptr) std::cout << "import of lstm model successful\n";

	std::shared_ptr<torch::jit::script::Module> rl_model = torch::jit::load("../include/rl_model.pt");

  if(rl_model != nullptr) std::cout << "import of rl model successful\n";


	char readBuffer[65536];
	FileReadStream is(file, readBuffer, sizeof(readBuffer));
	Document document;
	document.ParseStream(is);
	fclose(file);

	inputSendAddress(document["sendAddress"].GetString());
	inputSendPort(document["sendPort"].GetInt());
	inputDistanceRadius(document["distanceRadius"].GetInt());
	inputMergeLocation(document["longitude"].GetUint(),document["latitude"].GetUint());
	inputReceivePort(document["receivePort"].GetInt());
	inputReceiveAddress(document["receiveAddress"].GetString());

	auto socket = initiateSubscription(sendAddress,sendPort,receiveAddress,receivePort,filter,document["distanceRadius"].GetInt(),document["longitude"].GetUint(),document["latitude"].GetUint());
	printf("Subscription Service: Sending subscription request to address %s using port %d.\n", sendAddress.c_str(), sendPort);
	initaliseDatabase();
	printf("Initialising Database: Preparing to receive subscription response.\n");
	bool listening = false;

	auto end_time = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(20000);


	do {
		if(listening == true) {
			printf("Waiting for notify packets.\n");
		}
		int filterValue = filterExecution(listenDataTCP(socket));
		if(filterValue == 1) {
			printf("%s\n","Subscription Response Received.");
			listening = true;
		}
		if(filterValue == 0 || filterValue == 3) {
			vector<ManeuverRecommendation*> recommendations = ManeuverParser(database,distanceRadius,lstm_model,rl_model);
			database->deleteAll();
			if(recommendations.size() > 0) {
				cout << "<<<<<<<<<<<<<<<<<< Predicting Vehicle States/RL TR >>>>>>>>>>>>>>>>>>>" << endl;
				cout << "\n\n\n\n\n\n\n" << "***********************************"<< "Sending " << recommendations.size() << " trajectory recommendations." << "***********************************" << "\n\n\n\n\n\n\n" << endl;
				sendTrajectoryRecommendations(recommendations,socket);
			}else	printf("No Trajectories Calculated.\n");
		}
		if(filterValue == 4) {
			printf("Road User Deleted.\n");
		}
	} while(std::chrono::high_resolution_clock::now() <  end_time);

	cout << "want to unsubscribe" << endl;

	initiateUnsubscription(sendAddress,sendPort,subscriptionResp);

	return 0;
}


// std::vector<torch::jit::IValue> rl_inputs;
// std::vector<torch::jit::IValue> lstm_inputs;
// lstm_inputs.push_back(torch::rand({1, 2, 19}));
// auto lstm_out = lstm_model->forward(lstm_inputs).toTensor();
// rl_inputs.push_back(lstm_out);
// auto rl_out = rl_model->forward(rl_inputs).toTensor();

// std::cout << lstm_inputs << std::endl;
// cout << "LSTM OUTPUTS" << endl;
// std::cout << lstm_out << std::endl;
// cout << "RL OUTPUTS" << endl;
// cout << rl_out << endl;
