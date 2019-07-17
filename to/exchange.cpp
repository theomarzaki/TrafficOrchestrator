// This is the Main script that brings all the components together

// Obtains TO connection data from configuration file and starts a connection and listens

// Created by : KCL

// Modified by : Omar Nassef(KCL)

#include <utility>
#include <cstdio>
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

#include <road_safety.h>
#include <logger.h>
#include <database.h>
#include <maneuver_feedback.h>
#include <unsubscription_response.h>
#include <subscription_response.h>
#include <optimizer_engine.h>
#include <detection_interface.h>
#include <create_trajectory.h>
#include <network_interface.h>

auto database{std::make_shared<Database>()};
std::shared_ptr<SubscriptionResponse> subscriptionResponse;

pair<int,int> northeast;
pair<int,int> southwest;
int request_id;
int socket_c;
bool filter = true;
std::shared_ptr<torch::jit::script::Module> lstm_model;
std::shared_ptr<torch::jit::script::Module> rl_model;


vector<std::shared_ptr<RoadUser>> detectedToRoadUserList(const vector<Detected_Road_User> &v) {

	logger::write("Detected number of RoadUsers: " + std::string(std::to_string(v.size())));

	vector<std::shared_ptr<RoadUser>> road_users;

	for(const auto& d : v) {

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

auto detectedToFeedback(const Detected_Trajectory_Feedback& d) {

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

auto detectedToSubscription(const Detected_Subscription_Response& d) {
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


auto detectedToUnsubscription(const Detected_Unsubscription_Response& d) {
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

void generateReqID(){
	srand(time(nullptr));
    // FIXME limited generation : use std random library instead
    request_id = std::rand();
}

void sendTrajectoryRecommendations(const vector<std::shared_ptr<ManeuverRecommendation>> &v) {
	for(const auto &m : v) {
		m->setSourceUUID("traffic_orchestrator_" + std::to_string(request_id));
        auto maneuverJson{SendInterface::createManeuverJSON(m)};
        logger::write(maneuverJson);
        // we trace the emission as far as possible
        std::stringstream log;
        // we may have v2x_gateway into the source_uuid, bu we receive the original one
        log << "traffic_orchestrator maneuver sent_to v2x_gateway "
            << m->getMessageID()
            << " at "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()
            << std::endl;
        std::cout << log.str();
        std::cout.flush();
        // FIXME when we use socket_c, we've go an error
        SendInterface::sendTCP(maneuverJson);
	}
}

void initiateSubscription() {
	std::chrono::milliseconds timeSub = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
	auto subscriptionReq{std::make_shared<SubscriptionRequest>()};
	generateReqID();
	subscriptionReq->setSourceUUID("traffic_orchestrator_" + std::to_string(request_id));
	subscriptionReq->setFilter(filter);

	subscriptionReq->setShape("rectangle");
	subscriptionReq->setSignature("TEMPLATE");
	subscriptionReq->setRequestId(request_id);
	subscriptionReq->setNorthEast(northeast);
	subscriptionReq->setSouthWest(southwest);
	// FIXME do not cast an unsigned int 64 from a long
	subscriptionReq->setTimestamp(static_cast<uint64_t>(timeSub.count()));
	subscriptionReq->setMessageID(std::string(subscriptionReq->getOrigin()) + "/" + std::string(std::to_string(subscriptionReq->getRequestId())) + "/" + std::string(std::to_string(subscriptionReq->getTimestamp())));
    socket_c = SendInterface::sendTCP(SendInterface::createSubscriptionRequestJSON(subscriptionReq),true);
	logger::write("Sent subscription request to " + SendInterface::connectionAddress + ":"+ std::to_string(SendInterface::port));
}

void initiateUnsubscription() {

    std::chrono::milliseconds timeUnsub = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
	auto unsubscriptionReq{std::make_shared<UnsubscriptionRequest>()};
	unsubscriptionReq->setSourceUUID("traffic_orchestrator_" + std::to_string(request_id));
	unsubscriptionReq->setSubscriptionId(request_id);
	// FIXME do not cast an unsigned int 64 from a long
	unsubscriptionReq->setTimestamp(static_cast<uint64_t>(timeUnsub.count()));
    SendInterface::sendTCP(SendInterface::createUnsubscriptionRequestJSON(unsubscriptionReq));
}

void handleSubscriptionResponse(/*rapidjson::Document &document*/) {
	logger::write("Subscription Response Received.");
//	return detectedToSubscription(assignSubResponseVals(document));
}

void handleUnSubscriptionResponse(/*rapidjson::Document &document*/) {
	logger::write("unsubscription response Received.");
	// return detectedToUnsubscription(assignUnsubResponseVals(document));
}

void handleNotifyAdd(rapidjson::Document &document) {

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    document["message"]["ru_description_list"][0].Accept(writer);
    logger::dumpToFile(buffer.GetString());

	logger::write("Notify Add Received.");
	const vector<Detected_Road_User> &roadUsers = assignNotificationVals(document).ru_description_list;
    // we trace the reception as soon as possible and when the message_id stays available
    std::for_each(roadUsers.begin(),
                  roadUsers.end(),
                  [](const auto &roadUser) {
                      // TODO use std:optional instead of use a "placeholder" default value
                      if ("placeholder" != roadUser.message_id) {
                          std::stringstream log;
                          // we may have v2x_gateway into the source_uuid, bu we receive the original one
                          /*log << "traffic_orchestrator ru_description received_from v2x_gateway "
                              << roadUser.message_id
                              << " at "
                              << std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch()).count()
                              << std::endl;*/
                          std::cout << log.str();
                          std::cout.flush();
                      }
                  });
    auto road_users{detectedToRoadUserList(roadUsers)};
    for (const auto &road_user : road_users) {
        database->upsert(road_user);
	}
}

bool handleTrajectoryFeedback(rapidjson::Document &document) {
    auto maneuverFeed = detectedToFeedback(assignTrajectoryFeedbackVals(document));
    // we trace the reception as soon as possible
    std::stringstream log;
    // we may have v2x_gateway into the source_uuid, bu we receive the original one
    /*log << "traffic_orchestrator maneuver_feedback received_from v2x_gateway "
        // TODO use std:optional instead of use a "placeholder" default value
        << ("placeholder" != maneuverFeed->getUuidManeuver() ? maneuverFeed->getUuidManeuver() : "")
        << "/"
        // TODO use std:optional instead of use a "placeholder" default value
        << ("placeholder" != maneuverFeed->getFeedback() ? maneuverFeed->getFeedback() : "")
        << "/"
        // TODO use std:optional instead of use a "placeholder" default value
        << ("placeholder" != maneuverFeed->getReason() ? maneuverFeed->getReason() : "")
        << " at "
        << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()
        << std::endl; */
    std::cout << log.str();
    std::cout.flush();
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

void handleNotifyDelete(rapidjson::Document &document) {
	logger::write("Notify delete Received.");
	auto uuidsVector{assignNotificationDeleteVals(document)};
	for_each(uuidsVector.begin(), uuidsVector.end(), [](std::string uuid) {
         database->deleteRoadUser(uuid);
         logger::write("Deleted road user " + uuid);
	});
}

void inputNorthEast(int longt, int lat){
	northeast = {longt,lat};
}

void inputSouthWest(int longt, int lat){
	southwest = {longt,lat};
}

void computeManeuvers() {
    auto recommendations{ManeuverParser(database, rl_model)};
    if (!recommendations.empty()) {
        logger::write("Sending recommendations.\n");
        sendTrajectoryRecommendations(recommendations);
    } else {
        logger::write("No Trajectories Calculated.\n");
    }
}

void computeSafetyActions(){
	auto recommendations = stabiliseRoad(std::shared_ptr<Database>(database));
	if(!recommendations.empty()) {
			logger::write("Sending Safety Action.\n");
			sendTrajectoryRecommendations(recommendations);
		}
}

// Function Handling the exit of TO
void terminate_to(int signum ){
	logger::write("Sending unsubscription request.\n");
    initiateUnsubscription();
	std::this_thread::sleep_for(std::chrono::milliseconds(15000));
	close(socket_c);
	lstm_model.reset();
	rl_model.reset();
	exit(signum);
}

void handleAIMessage(const std::string &captured_data){
	rapidjson::Document document = parse(captured_data);
	message_type messageType = filterInput(document);
	if (captured_data == "\n" || captured_data.empty()) {
			messageType = message_type::heart_beat;
	}

	switch (messageType) {
			case message_type::notify_add:
				handleNotifyAdd(document);
				computeManeuvers();
				// computeSafetyActions();
				break;
			case message_type::notify_delete:
					handleNotifyDelete(document);
					break;
			case message_type::subscription_response:
					handleSubscriptionResponse();
					break;
			case message_type::unsubscription_response:
					handleUnSubscriptionResponse();
					break;
			case message_type::trajectory_feedback:
					if (!handleTrajectoryFeedback(document)) {
							computeManeuvers();
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
}

void handleGraphMessage(const std::string &captured_data){
    rapidjson::Document document = parse(captured_data);
    message_type messageType = filterInput(document);
    if (captured_data == "\n" || captured_data.empty()) {
        messageType = message_type::heart_beat;
    }

    OptimizerEngine::getEngine()->startManeuverFeedback();
    switch (messageType) {
        case message_type::notify_add:
            handleNotifyAdd(document);
            OptimizerEngine::getEngine()->locker.lock();
            OptimizerEngine::getEngine()->updateSimulationState(database->dump());
            OptimizerEngine::getEngine()->locker.unlock();
            break;
        case message_type::notify_delete:
            handleNotifyDelete(document);
            OptimizerEngine::getEngine()->locker.lock();
            for (auto& uuid : assignNotificationDeleteVals(document)) {
                std::cout << uuid << std::endl;
                OptimizerEngine::getEngine()->removeFromSimulation(uuid);
            }
            OptimizerEngine::getEngine()->locker.unlock();
            break;
        case message_type::subscription_response:
            handleSubscriptionResponse();
            break;
        case message_type::unsubscription_response:
            OptimizerEngine::getEngine()->pauseManeuverFeedback();
            handleUnSubscriptionResponse();
            break;
        case message_type::trajectory_feedback:
            if (!handleTrajectoryFeedback(document)) {
                OptimizerEngine::getEngine()->locker.lock();
                OptimizerEngine::getEngine()->updateSimulationState(database->dump());
                OptimizerEngine::getEngine()->locker.unlock();
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
}


int main() {

    char readBuffer[65536];
    rapidjson::Document args;

    auto file{fopen("include/TO_config.json", "r")};
    if (file == nullptr) {
        logger::write("[ERROR] Config File failed to load -> Abort");
        return 1;
    }
    if (!std::experimental::filesystem::create_directory("logs") && !std::experimental::filesystem::exists("logs")) {
        logger::write("[ERROR] Unable to create the logs directory -> Abort");
        return 2;
    }

    lstm_model = torch::jit::load("include/lstm_model.pt");
    if (lstm_model == nullptr) {
        logger::write("[ERROR] import of lstm model unsuccessful -> Abort\n");
        return 3;
    }
    logger::write("[INFO] import of lstm model successful\n");

    rl_model = torch::jit::load("include/rl_model_deuling.pt");
    if (rl_model == nullptr) {
        logger::write("[ERROR] import of rl model unsuccessful -> Abort\n");
        return 4;
    }
    logger::write("[INFO] import of rl model successful\n");

    rapidjson::FileReadStream is(file, readBuffer, sizeof(readBuffer));
    args.ParseStream(is);
    fclose(file);

    SendInterface::connectionAddress = args["sendAddress"].GetString();
    SendInterface::port = args["sendPort"].GetInt();
    SendInterface::receivePort = args["receivePort"].GetInt();
    SendInterface::receiveAddress = args["receiveAddress"].GetString();

    inputNorthEast(args["northeast"]["longitude"].GetInt(), args["northeast"]["latitude"].GetInt());
    inputSouthWest(args["southwest"]["longitude"].GetInt(), args["southwest"]["latitude"].GetInt());

    initiateSubscription();

    // terminate TO on abortion/interruption
    signal(SIGINT, terminate_to);

    std::thread mainT([=, &args]() {
        while(true) {
            std::string captured_data_end;
            do {
                for (const auto& captured_data : listenDataTCP(socket_c)) {
                    try {
                        if (args["computation_with_ai"].GetBool()) {
                            handleAIMessage(captured_data);
                        } else {
                            handleGraphMessage(captured_data);
                        }
                    } catch (const std::exception &e) {
                        logger::write("[ERROR] Malformed JSON");
                    }
                    captured_data_end = captured_data;
                }
            } while (captured_data_end != "RECONNECT");
            std::cout << "Target seems disconnected -> next attempt in 10 sec." << std::endl;
            OptimizerEngine::getEngine()->pauseManeuverFeedback();
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    });

    OptimizerEngine::getEngine()->getThread()->join();
    mainT.join();

    return 0;
}
