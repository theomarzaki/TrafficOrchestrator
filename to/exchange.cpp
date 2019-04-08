// This is the Main script that brings all the components together

// Obtains TO connection data from configuration file and starts a connection and listens

// Created by : KCL

// Modified by : Omar Nassef(KCL)

#include <torch/torch.h>
#include <torch/script.h>
#include "detection_interface.cpp"
#include "database.cpp"
#include "maneuver_feedback.cpp"
#include "network_interface.cpp"
#include "subscription_response.cpp"
#include "CreateTrajectory.cpp"
#include "unsubscription_response.cpp"
#include <experimental/filesystem>

using namespace std;

using namespace rapidjson;

using namespace std::chrono;

using namespace experimental;

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


vector<shared_ptr<RoadUser>> detectedToRoadUserList(vector<Detected_Road_User> v) {

	write_to_log("Detected number of RoadUsers: " + string(to_string(v.size())) + ".\n");

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
	srand(time(NULL));
	int x = std::rand();
	return x;
}

void sendTrajectoryRecommendations(vector<std::shared_ptr<ManeuverRecommendation>> v,
								   int socket) {
	for (const auto &m : v) {
		cout << createManeuverJSON(m) << endl;
		write_to_log(createManeuverJSON(m));
		sendDataTCP(socket, sendAddress, sendPort, receiveAddress, receivePort, createManeuverJSON(m));
	}
}

int initiateSubscription(string sendAddress, int sendPort,string receiveAddress,int receivePort, bool filter,int radius,uint32_t longitude, uint32_t latitude) {
	milliseconds timeSub = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	SubscriptionRequest * subscriptionReq = new SubscriptionRequest();
	int request_id = generateReqID();
	subscriptionReq->setSourceUUID("traffic_orchestrator_" + to_string(request_id));
	subscriptionReq->setFilter(filter);
	subscriptionReq->setRadius(radius);
	subscriptionReq->setLongitude(longitude);
	subscriptionReq->setLatitude(latitude);
	subscriptionReq->setShape("circle");
	subscriptionReq->setSignature("TEMPLATE");
	subscriptionReq->setRequestId(request_id);
	subscriptionReq->setTimestamp(timeSub.count());
	subscriptionReq->setMessageID(std::string(subscriptionReq->getOrigin()) + "/" + std::string(to_string(subscriptionReq->getRequestId())) + "/" + std::string(to_string(subscriptionReq->getTimestamp())));
	auto socket = sendDataTCP(-999,sendAddress,sendPort,receiveAddress,receivePort,createSubscriptionRequestJSON(subscriptionReq));
	write_to_log("Sent subscription request to " + sendAddress + ":"+ to_string(sendPort));
	return socket;
}

void initiateUnsubscription(string sendAddress, int sendPort, SubscriptionResponse * subscriptionResp) {

	milliseconds timeUnsub = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	UnsubscriptionRequest * unsubscriptionReq = new UnsubscriptionRequest();
	unsubscriptionReq->setSubscriptionId(subscriptionResp->getSubscriptionId());
	unsubscriptionReq->setTimestamp(timeUnsub.count());
	sendDataTCP(-999,sendAddress,sendPort,receiveAddress,receivePort,createUnsubscriptionRequestJSON(unsubscriptionReq));
}

SubscriptionResponse * handleSubscriptionResponse(Document &document) {
	write_to_log("Subscription Response Received.");
	return detectedToSubscription(assignSubResponseVals(document));
}

UnsubscriptionResponse * handleUnSubscriptionResponse(Document &document) {
	write_to_log("unsubscription response Received.");
	return detectedToUnsubscription(assignUnsubResponseVals(document));
}

void handleNotifyAdd(Document &document) {
	write_to_log("Notify Add Received.");
	const vector<Detected_Road_User> &roadUsers = assignNotificationVals(document).ru_description_list;
	int size = roadUsers.size();

	auto road_users{detectedToRoadUserList(roadUsers)};

	for(auto road_user : road_users) {
		database->upsert(road_user);
	}
}

bool handleTrajectoryFeedback(Document &document) {
	cout << "\n*********************************** Received Trajectory Feedback *********************************** \n";
	maneuverFeed = detectedToFeedback(assignTrajectoryFeedbackVals(document));
	write_to_log("Maneuver Feedback: " + maneuverFeed->getFeedback());
	if(maneuverFeed->getFeedback() == "refuse" || maneuverFeed->getFeedback() == "abort") {
		write_to_log("calculating new Trajectory for Vehicle");
		return false;
	}
  return true;
}

void handleNotifyDelete(Document &document) {
	write_to_log("Notify delete Received.");
	auto uuidsVector{assignNotificationDeleteVals(document)};
	for_each(uuidsVector.begin(), uuidsVector.end(),
					 [](string uuid)
					 {
							 database->deleteRoadUser(uuid);
							 write_to_log("Deleted road user " + uuid);
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

void inputMergeLocation(uint32_t longt, uint32_t lat){
	mergingLongitude = longt;
	mergingLatitude = lat;
}

void inputDistanceRadius(int radius) {
	distanceRadius = radius;
}

void initaliseDatabase() {
	database = new Database();
}

void computeManeuvers(const shared_ptr<torch::jit::script::Module> &lstm_model,
					  const shared_ptr<torch::jit::script::Module> &rl_model,
					  int socket) {
	auto recommendations{ManeuverParser(database, distanceRadius, lstm_model, rl_model)};
	if (!recommendations.empty()) {
		write_to_log("\n ***********************************  Sending  *********************************** \n");
		sendTrajectoryRecommendations(recommendations, socket);
	} else {
		write_to_log("No Trajectories Calculated.\n");
	}
}

int main() {
    auto returnCode{0};

    FILE *file = fopen("include/TO_config.json", "r");
    if (file == 0) {
        std::cout << "Config File failed to load." << std::endl;
        returnCode = 1;
    } else if (!filesystem::create_directory("logs") && !filesystem::exists("logs")) {
        std::cout << "Unable to create the logs directory, we stop" << std::endl;
        returnCode = 2;
    } else {
        std::shared_ptr<torch::jit::script::Module> lstm_model = torch::jit::load("include/lstm_model.pt");

        if (lstm_model != nullptr) write_to_log("import of lstm model successful\n");

        std::shared_ptr<torch::jit::script::Module> rl_model = torch::jit::load("include/rl_model_deuling.pt");

        if (rl_model != nullptr) write_to_log("import of rl model successful\n");


        char readBuffer[65536];
        FileReadStream is(file, readBuffer, sizeof(readBuffer));
        Document document;
        document.ParseStream(is);
        fclose(file);

        inputSendAddress(document["sendAddress"].GetString());
        inputSendPort(document["sendPort"].GetInt());
        inputDistanceRadius(document["distanceRadius"].GetInt());
        inputMergeLocation(document["longitude"].GetUint(), document["latitude"].GetUint());
        inputReceivePort(document["receivePort"].GetInt());
        inputReceiveAddress(document["receiveAddress"].GetString());

        auto socket = initiateSubscription(sendAddress, sendPort, receiveAddress, receivePort, filter,
                                           document["distanceRadius"].GetInt(), document["longitude"].GetUint(),
                                           document["latitude"].GetUint());
        initaliseDatabase();
        bool listening = false;
        string reconnect = "RECONNECT";
        string reconnect_flag;


        do {
            auto captured_data = listenDataTCP(socket);
            Document document = parse(captured_data);
            message_type messageType = filterInput(document);
            if (captured_data == "\n" || captured_data == string()) {
                messageType = message_type::heart_beat;
            }

            switch (messageType) {
                case message_type::notify_add:
                    handleNotifyAdd(document);
                    computeManeuvers(lstm_model, rl_model, socket);
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
                        computeManeuvers(lstm_model, rl_model, socket);
                    }
                    break;
                case message_type::heart_beat:
                    write_to_log("Recieved HeartBeat");
                    break;
                case message_type::reconnect:
                    write_to_log("Reconnecting");
                    break;
                default:
                    cout << "Captured Data leading to Error Message:" << captured_data << "End " << endl;
                    write_to_log("error: Couldn't handle message.");
                    break;
            }
            reconnect_flag = captured_data;
        } while (reconnect_flag != reconnect);
        listening = false;
        while (!listening) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10000));
            main();
        }
        //FIXME remove the dead code
        initiateUnsubscription(sendAddress, sendPort, subscriptionResp);
    }
    return returnCode;
}
