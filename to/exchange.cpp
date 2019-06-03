#include <utility>

// This is the Main script that brings all the components together

// Obtains TO connection data from configuration file and starts a connection and listens

// Created by : KCL

// Modified by : Omar Nassef(KCL)

#include <fstream>
#include <string>
#include <sstream>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <experimental/filesystem>
#include <experimental/random>
#include <csignal>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <sys/time.h>

#include "rapidjson/document.h"
#include <torch/torch.h>
#include <torch/script.h>

#include "detection_interface.cpp"
#include "database.cpp"
#include "maneuver_feedback.cpp"
#include "network_interface.cpp"
#include "subscription_response.cpp"
#include "CreateTrajectory.cpp"
#include "unsubscription_response.cpp"
#include "logger.h"
#include "road_safety.cpp"

using namespace rapidjson;
using namespace experimental;

Database * database;
std::shared_ptr<SubscriptionResponse> subscriptionResp;
std::shared_ptr<UnsubscriptionResponse> unsubscriptionResp;
std::shared_ptr<ManeuverFeedback> maneuverFeed;

string sendAddress;
int sendPort;
string receiveAddress;
int receivePort;


pair<int,int> northeast;
pair<int,int> southwest;
string uuidTo;
int request_id;
int socket_c;
bool filter = true;
std::shared_ptr<torch::jit::script::Module> lstm_model;
std::shared_ptr<torch::jit::script::Module> rl_model;


vector<shared_ptr<RoadUser>> detectedToRoadUserList(vector<Detected_Road_User> v) {

	logger::write("Detected number of RoadUsers: " + string(to_string(v.size())));

	vector<shared_ptr<RoadUser>> road_users;

	for(int i = 0; i < v.size(); i++) {

		Detected_Road_User d = v[i];

		auto roadUser{std::make_shared<RoadUser>()}; // Declares and initalises a RoadUser pointer.
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
    roadUser->setSourceUUID(d.source_uuid);
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

		road_users.push_back(roadUser);

	}

	return road_users;

}

auto detectedToFeedback(Detected_Trajectory_Feedback d) {

	auto maneuverFeed{std::make_shared<ManeuverFeedback>()};
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

auto detectedToSubscription(Detected_Subscription_Response d) {
	auto subscriptionResp{std::make_shared<SubscriptionResponse>()};
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


auto detectedToUnsubscription(Detected_Unsubscription_Response d) {
	auto unsubscriptionResp{std::make_shared<UnsubscriptionResponse>()};
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
	// FIXME limited generation : use std random library instead
	uuidTo = std::to_string(std::experimental::randint(10000000,99999999));
}

void generateReqID(){
	srand(time(NULL));
	request_id = std::rand();
}

void sendTrajectoryRecommendations(vector<std::shared_ptr<ManeuverRecommendation>> v,int socket) {
	for(const auto &m : v) {
		m->setSourceUUID("traffic_orchestrator_" + to_string(request_id));
		logger::write(createManeuverJSON(m));
		sendDataTCP(socket, sendAddress, sendPort, receiveAddress, receivePort, createManeuverJSON(m));
	}
}

void initiateSubscription(const string &sendAddress, int sendPort,string receiveAddress,int receivePort, bool filter) {
	milliseconds timeSub = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	auto subscriptionReq{std::make_shared<SubscriptionRequest>()};
	generateReqID();
	subscriptionReq->setSourceUUID("traffic_orchestrator_" + to_string(request_id));
	subscriptionReq->setFilter(filter);

	subscriptionReq->setShape("rectangle");
	subscriptionReq->setSignature("TEMPLATE");
	subscriptionReq->setRequestId(request_id);
	subscriptionReq->setNorthEast(northeast);
	subscriptionReq->setSouthWest(southwest);
	// FIXME do not cast an unsigned int 64 from a long
	subscriptionReq->setTimestamp(static_cast<uint64_t>(timeSub.count()));
	subscriptionReq->setMessageID(std::string(subscriptionReq->getOrigin()) + "/" + std::string(to_string(subscriptionReq->getRequestId())) + "/" + std::string(to_string(subscriptionReq->getTimestamp())));
	socket_c = sendDataTCP(-999,sendAddress,sendPort,std::move(receiveAddress),receivePort,createSubscriptionRequestJSON(subscriptionReq));
	logger::write("Sent subscription request to " + sendAddress + ":"+ to_string(sendPort));
}

void initiateUnsubscription(const string &sendAddress, int sendPort, std::shared_ptr<SubscriptionResponse> subscriptionResp) {

	milliseconds timeUnsub = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	auto unsubscriptionReq{std::make_shared<UnsubscriptionRequest>()};
	unsubscriptionReq->setSourceUUID("traffic_orchestrator_" + to_string(request_id));
	unsubscriptionReq->setSubscriptionId(request_id);
	// FIXME do not cast an unsigned int 64 from a long
	unsubscriptionReq->setTimestamp(static_cast<uint64_t>(timeUnsub.count()));
	sendDataTCP(-999,sendAddress,sendPort,receiveAddress,receivePort,createUnsubscriptionRequestJSON(unsubscriptionReq));
}

auto handleSubscriptionResponse(Document &document) {
	logger::write("Subscription Response Received.");
	return detectedToSubscription(assignSubResponseVals(document));
}

auto handleUnSubscriptionResponse(Document &document) {
	logger::write("unsubscription response Received.");
	return detectedToUnsubscription(assignUnsubResponseVals(document));
}

void handleNotifyAdd(Document &document) {
	logger::write("Notify Add Received.");
	const vector<Detected_Road_User> &roadUsers = assignNotificationVals(document).ru_description_list;
	auto road_users{detectedToRoadUserList(roadUsers)};
	for(const auto &road_user : road_users) {
		database->upsert(road_user);
	}
}

bool handleTrajectoryFeedback(Document &document) {
	maneuverFeed = detectedToFeedback(assignTrajectoryFeedbackVals(document));
	auto roadUser = database->findRoadUser(maneuverFeed->getUuidVehicle());
	logger::write("Maneuver Feedback: " + maneuverFeed->getFeedback());
	if(maneuverFeed->getFeedback() == "refuse" || maneuverFeed->getFeedback() == "abort") {
		logger::write("calculating new Trajectory for Vehicle");
		if(roadUser != nullptr){
			roadUser->setProcessingWaypoint(false);
			database->upsert(roadUser);
		}
		return false;
	}
	if(maneuverFeed->getFeedback() == "checkpoint"){
		if(roadUser != nullptr){
			roadUser->setProcessingWaypoint(false);
			database->upsert(roadUser);
		}
	}
	return true;
}

void handleNotifyDelete(Document &document) {
	logger::write("Notify delete Received.");
	auto uuidsVector{assignNotificationDeleteVals(document)};
	for_each(uuidsVector.begin(), uuidsVector.end(),
					 [](string uuid)
					 {
							 database->deleteRoadUser(uuid);
							 logger::write("Deleted road user " + uuid);
					 });

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

void inputNorthEast(int longt, int lat){
	northeast = make_pair(longt,lat);
}

void inputSouthWest(int longt, int lat){
	southwest = make_pair(longt,lat);
}

void initaliseDatabase() {
	database = new Database();
}

void computeManeuvers(const shared_ptr<torch::jit::script::Module> &lstm_model,
                      const shared_ptr<torch::jit::script::Module> &rl_model, int socket) {
  auto recommendations = ManeuverParser(database,rl_model);
  if(!recommendations.empty()) {
					logger::write("Sending recommendations.\n");
					sendTrajectoryRecommendations(recommendations,socket);
				} else {
					logger::write("No Trajectories Calculated.\n");
				}
}

void computeSafetyActions(){
	auto recommendations = stabiliseRoad(database);
	if(!recommendations.empty()) {
			logger::write("Sending Safety Action.\n");
			sendTrajectoryRecommendations(recommendations,socket_c);
		}
}

// Function Handling the exit of TO
void terminate_to(int signum ){
	logger::write("Sending unsubscription request.\n");
	initiateUnsubscription(sendAddress,sendPort,subscriptionResp);
	std::this_thread::sleep_for(std::chrono::milliseconds(15000));
	close(socket_c);
	lstm_model.reset();
	rl_model.reset();
	exit(signum);
}


int main() {

    auto returnCode{0};

    FILE *file = fopen("include/TO_config.json", "r");
    if (file == 0) {
        logger::write("Config File failed to load.");
        returnCode = 1;
    } else if (!filesystem::create_directory("logs") && !filesystem::exists("logs")) {
        logger::write("Unable to create the logs directory, we stop");
        returnCode = 2;
    } else {
        lstm_model = torch::jit::load("include/lstm_model.pt");

        if (lstm_model != nullptr) logger::write("import of lstm model successful\n");

        rl_model = torch::jit::load("include/rl_model_deuling.pt");

        if (rl_model != nullptr) logger::write("import of rl model successful\n");


        char readBuffer[65536];
        FileReadStream is(file, readBuffer, sizeof(readBuffer));
        Document document;
        document.ParseStream(is);
        fclose(file);

        inputSendAddress(document["sendAddress"].GetString());
        inputSendPort(document["sendPort"].GetInt());
        inputNorthEast(document["northeast"]["longitude"].GetInt(), document["northeast"]["latitude"].GetInt());
				inputSouthWest(document["southwest"]["longitude"].GetInt(), document["southwest"]["latitude"].GetInt());
        inputReceivePort(document["receivePort"].GetInt());
        inputReceiveAddress(document["receiveAddress"].GetString());

        initiateSubscription(sendAddress, sendPort, receiveAddress, receivePort, filter);
        initaliseDatabase();
        bool listening = false;
        string reconnect_flag;

				// terminate TO on abortion/interruption
				signal(SIGINT,terminate_to);

        do {
            auto captured_data = listenDataTCP(socket_c);
            Document document = parse(captured_data);
            message_type messageType = filterInput(document);
            if (captured_data == "\n" || captured_data == string()) {
                messageType = message_type::heart_beat;
            }

            switch (messageType) {
                case message_type::notify_add:
                    handleNotifyAdd(document);
                    computeManeuvers(lstm_model, rl_model, socket_c);
										computeSafetyActions();
                    break;
                case message_type::notify_delete:
                    handleNotifyDelete(document);
                    break;
                case message_type::subscription_response:
                    subscriptionResp = handleSubscriptionResponse(document);
                    break;
                case message_type::unsubscription_response:
                    unsubscriptionResp = handleUnSubscriptionResponse(document);
                    break;
                case message_type::trajectory_feedback:
                    if (!handleTrajectoryFeedback(document)) {
                        computeManeuvers(lstm_model, rl_model, socket_c);
                    }
                    break;
                case message_type::heart_beat:
                    break;
                case message_type::reconnect:
                    logger::write("Reconnecting");
                    break;
                default:
                    logger::write("error: couldn't handle message " + captured_data);
                    break;
            }
            reconnect_flag = captured_data;
        } while (reconnect_flag != "RECONNECT");
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
						close(socket_c);
						lstm_model.reset();
						rl_model.reset();
            main();
        }
    }
    return returnCode;
}
