#include "nearest_neighbour.cpp"
#include <chrono>

using namespace std::chrono;

vector<ManeuverRecommendation*> calculateTrajectories(Database * database, double distanceRadius, string uuidTo, uint32_t longitudeMerge, uint32_t latitudeMerge) {

	vector<ManeuverRecommendation*> trajectories;
	vector<pair<RoadUser*,vector<RoadUser*>>> neighbours;

	ManeuverRecommendation * mergingManeuver = new ManeuverRecommendation();
	vector<Waypoint*> mergingWaypoints;
	mergingManeuver->setWaypoints(mergingWaypoints);

	ManeuverRecommendation * avoidanceManeuver_1 = new ManeuverRecommendation();
	vector<Waypoint*> avoidanceWaypoints_1;
	avoidanceManeuver_1->setWaypoints(avoidanceWaypoints_1);

	ManeuverRecommendation * avoidanceManeuver_2 = new ManeuverRecommendation();
	vector<Waypoint*> avoidanceWaypoints_2;
	avoidanceManeuver_2->setWaypoints(avoidanceWaypoints_2);

	double distanceFirstToMergePoint;
	double distanceSecondToMergePoint;
	double distanceMergingVehicleToPoint;
	double headingFirstToMergePoint;
	double headingSecondToMergePoint;

	milliseconds timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

	for(RoadUser * r : *database->getDatabase()) {
		if(r->getConnected() == true && r->getLanePosition() == 0) {
			printf("Connected Vehicle in Lane 0.\n");
			neighbours = mapNeighbours(database,distanceRadius);
			printf("Calculating neighbours of merging vehicle.\n");
		}
	}

	if(neighbours.size() == 0) {
		printf("No neighbours calculated for merging vehicle.\n");
		return trajectories;
	}

	else if(neighbours.size() == 3) {
		for(pair<RoadUser*,vector<RoadUser*>> v : neighbours) {

			RoadUser * mergingVehicle = v.first;
			cout << mergingVehicle << endl;
			RoadUser * firstNeighbour = v.second[0];
			cout << firstNeighbour << endl;

			cout << "Merging vehicle has: " << v.second.size() << " neighbours.\n";

			distanceMergingVehicleToPoint = distanceEarth(mergingVehicle->getLatitude(), mergingVehicle->getLongitude(), latitudeMerge, longitudeMerge);
			distanceFirstToMergePoint = distanceEarth(firstNeighbour->getLatitude(), firstNeighbour->getLongitude(), latitudeMerge, longitudeMerge);

			if(mergingVehicle->getConnected() == true && mergingVehicle->getLanePosition() == 0) {

				timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
				mergingManeuver->setTimestamp(timeCalculator.count());
				mergingManeuver->setUuidVehicle(mergingVehicle->getUuid());
				mergingManeuver->setUuidTo(uuidTo);
				mergingManeuver->setTimestampAction(timeCalculator.count());
				mergingManeuver->setLongitudeAction(mergingVehicle->getLongitude());
				mergingManeuver->setLatitudeAction(mergingVehicle->getLatitude());
				mergingManeuver->setSpeedAction(mergingVehicle->getSpeed());
				mergingManeuver->setLanePositionAction(mergingVehicle->getLanePosition());

				Waypoint * waypoint = new Waypoint();
				waypoint->setTimestamp(timeCalculator.count() + (distanceMergingVehicleToPoint/mergingVehicle->getSpeed())*1000);
				waypoint->setLatitude(latitudeMerge);
				waypoint->setLongitude(longitudeMerge);
				waypoint->setSpeed(mergingVehicle->getSpeed());
				waypoint->setLanePosition(mergingVehicle->getLanePosition()+1);
				mergingManeuver->addWaypoint(waypoint);
				trajectories.push_back(mergingManeuver);

				if(v.second.size() == 1) {

					if(headingFirstToMergePoint <= 270 && headingFirstToMergePoint >= 90) {
						timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
						avoidanceManeuver_1->setTimestamp(timeCalculator.count());
						avoidanceManeuver_1->setUuidVehicle(firstNeighbour->getUuid());
						avoidanceManeuver_1->setUuidTo(uuidTo);
						avoidanceManeuver_1->setTimestampAction(timeCalculator.count());
						avoidanceManeuver_1->setLongitudeAction(firstNeighbour->getLongitude());
						avoidanceManeuver_1->setLatitudeAction(firstNeighbour->getLatitude());
						avoidanceManeuver_1->setSpeedAction(firstNeighbour->getSpeed());
						avoidanceManeuver_1->setLanePositionAction(firstNeighbour->getLanePosition());

						Waypoint * waypointFirstNeighbour = new Waypoint();
						waypointFirstNeighbour->setTimestamp(timeCalculator.count());
						waypointFirstNeighbour->setLatitude(firstNeighbour->getLatitude());
						waypointFirstNeighbour->setLongitude(firstNeighbour->getLongitude());
						waypointFirstNeighbour->setSpeed(firstNeighbour->getSpeed());
						waypointFirstNeighbour->setLanePosition(firstNeighbour->getLanePosition());
						avoidanceManeuver_1->addWaypoint(waypointFirstNeighbour);
						trajectories.push_back(avoidanceManeuver_1);

						return trajectories;
					}

					else if(headingFirstToMergePoint >= 270 && headingFirstToMergePoint <= 360 && headingFirstToMergePoint <= 90) {
						timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
						avoidanceManeuver_1->setTimestamp(timeCalculator.count());
						avoidanceManeuver_1->setUuidVehicle(firstNeighbour->getUuid());
						avoidanceManeuver_1->setUuidTo(uuidTo);
						avoidanceManeuver_1->setTimestampAction(timeCalculator.count());
						avoidanceManeuver_1->setLongitudeAction(firstNeighbour->getLongitude());
						avoidanceManeuver_1->setLatitudeAction(firstNeighbour->getLatitude());
						avoidanceManeuver_1->setSpeedAction(distanceFirstToMergePoint/(distanceMergingVehicleToPoint/mergingVehicle->getSpeed()) + mergingVehicle->getLength()/mergingVehicle->getSpeed());
						avoidanceManeuver_1->setLanePositionAction(firstNeighbour->getLanePosition());

						Waypoint * waypointFirstNeighbour = new Waypoint();
						waypointFirstNeighbour->setTimestamp(timeCalculator.count() + (distanceMergingVehicleToPoint/mergingVehicle->getSpeed())*1000 + (mergingVehicle->getLength()/mergingVehicle->getSpeed())*1000);
						waypointFirstNeighbour->setLatitude(latitudeMerge);
						waypointFirstNeighbour->setLongitude(longitudeMerge);
						waypointFirstNeighbour->setSpeed(mergingVehicle->getSpeed());
						waypointFirstNeighbour->setLanePosition(firstNeighbour->getLanePosition());
						avoidanceManeuver_1->addWaypoint(waypointFirstNeighbour);
						trajectories.push_back(avoidanceManeuver_1);

						return trajectories;

					}

					return trajectories;

				}


				else if(v.second.size() == 2) {

					if(headingFirstToMergePoint <= 270 && headingSecondToMergePoint <= 270 && headingFirstToMergePoint >= 90 && headingSecondToMergePoint >= 90) {

						timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
						avoidanceManeuver_1->setTimestamp(timeCalculator.count());
						avoidanceManeuver_1->setUuidVehicle(firstNeighbour->getUuid());
						avoidanceManeuver_1->setUuidTo(uuidTo);
						avoidanceManeuver_1->setTimestampAction(timeCalculator.count());
						avoidanceManeuver_1->setLongitudeAction(firstNeighbour->getLongitude());
						avoidanceManeuver_1->setLatitudeAction(firstNeighbour->getLatitude());
						avoidanceManeuver_1->setSpeedAction(firstNeighbour->getSpeed());
						avoidanceManeuver_1->setLanePositionAction(firstNeighbour->getLanePosition());

						avoidanceManeuver_2->setTimestamp(timeCalculator.count());
						avoidanceManeuver_2->setUuidTo(uuidTo);
						avoidanceManeuver_2->setTimestampAction(timeCalculator.count());

						Waypoint * waypointFirstNeighbour = new Waypoint();
						Waypoint * waypointSecondNeighbour = new Waypoint();

						waypointFirstNeighbour->setTimestamp(timeCalculator.count());
						waypointFirstNeighbour->setLatitude(firstNeighbour->getLatitude());
						waypointFirstNeighbour->setLongitude(firstNeighbour->getLongitude());
						waypointFirstNeighbour->setSpeed(firstNeighbour->getSpeed());
						waypointFirstNeighbour->setLanePosition(firstNeighbour->getLanePosition());
						avoidanceManeuver_1->addWaypoint(waypointFirstNeighbour);
						trajectories.push_back(avoidanceManeuver_1);

						waypointSecondNeighbour->setTimestamp(timeCalculator.count());
						avoidanceManeuver_2->addWaypoint(waypointSecondNeighbour);
						trajectories.push_back(avoidanceManeuver_2);

						return trajectories;

					}

					else if(headingFirstToMergePoint >= 270 && headingSecondToMergePoint >= 270 && headingFirstToMergePoint <= 90 && headingSecondToMergePoint <= 90) {

						timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
						avoidanceManeuver_1->setTimestamp(timeCalculator.count());
						avoidanceManeuver_1->setUuidVehicle(firstNeighbour->getUuid());
						avoidanceManeuver_1->setUuidTo(uuidTo);
						avoidanceManeuver_1->setTimestampAction(timeCalculator.count());
						avoidanceManeuver_1->setLongitudeAction(firstNeighbour->getLongitude());
						avoidanceManeuver_1->setLatitudeAction(firstNeighbour->getLatitude());
						avoidanceManeuver_1->setSpeedAction(distanceFirstToMergePoint/(distanceMergingVehicleToPoint/mergingVehicle->getSpeed()) + mergingVehicle->getLength()/mergingVehicle->getSpeed());
						avoidanceManeuver_1->setLanePositionAction(firstNeighbour->getLanePosition());

						avoidanceManeuver_2->setTimestamp(timeCalculator.count());
						avoidanceManeuver_2->setUuidTo(uuidTo);
						avoidanceManeuver_2->setTimestampAction(timeCalculator.count());
						avoidanceManeuver_2->setSpeedAction(distanceSecondToMergePoint/(distanceMergingVehicleToPoint/mergingVehicle->getSpeed()) + mergingVehicle->getLength()/mergingVehicle->getSpeed());

						Waypoint * waypointFirstNeighbour = new Waypoint();
						Waypoint * waypointSecondNeighbour = new Waypoint();

						waypointFirstNeighbour->setTimestamp(timeCalculator.count() + (distanceMergingVehicleToPoint/mergingVehicle->getSpeed())*1000 + (mergingVehicle->getLength()/mergingVehicle->getSpeed())*1000);
						waypointFirstNeighbour->setLatitude(latitudeMerge);
						waypointFirstNeighbour->setLongitude(longitudeMerge);
						waypointFirstNeighbour->setSpeed(mergingVehicle->getSpeed());
						waypointFirstNeighbour->setLanePosition(firstNeighbour->getLanePosition());
						avoidanceManeuver_1->addWaypoint(waypointFirstNeighbour);
						trajectories.push_back(avoidanceManeuver_1);

						waypointSecondNeighbour->setTimestamp(timeCalculator.count() + (distanceMergingVehicleToPoint/mergingVehicle->getSpeed())*1000 + (mergingVehicle->getLength()/mergingVehicle->getSpeed())*1000);
						waypointSecondNeighbour->setLatitude(latitudeMerge);
						waypointSecondNeighbour->setLongitude(longitudeMerge);
						waypointSecondNeighbour->setSpeed(mergingVehicle->getSpeed());
						avoidanceManeuver_2->addWaypoint(waypointSecondNeighbour);
						trajectories.push_back(avoidanceManeuver_2);

						return trajectories;
					}

					else if(headingFirstToMergePoint <= 270 && headingFirstToMergePoint >= 90 && headingSecondToMergePoint >= 270 && headingSecondToMergePoint <= 90) {

						timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
						avoidanceManeuver_1->setTimestamp(timeCalculator.count());
						avoidanceManeuver_1->setUuidVehicle(firstNeighbour->getUuid());
						avoidanceManeuver_1->setUuidTo(uuidTo);
						avoidanceManeuver_1->setTimestampAction(timeCalculator.count());
						avoidanceManeuver_1->setLongitudeAction(firstNeighbour->getLongitude());
						avoidanceManeuver_1->setLatitudeAction(firstNeighbour->getLatitude());
						avoidanceManeuver_1->setSpeedAction(firstNeighbour->getSpeed());
						avoidanceManeuver_1->setLanePositionAction(firstNeighbour->getLanePosition());

						Waypoint * waypointFirstNeighbour = new Waypoint();
						waypointFirstNeighbour->setTimestamp(timeCalculator.count());
						waypointFirstNeighbour->setLatitude(firstNeighbour->getLatitude());
						waypointFirstNeighbour->setLongitude(firstNeighbour->getLongitude());
						waypointFirstNeighbour->setSpeed(firstNeighbour->getSpeed());
						waypointFirstNeighbour->setLanePosition(firstNeighbour->getLanePosition());
						avoidanceManeuver_1->addWaypoint(waypointFirstNeighbour);
						trajectories.push_back(avoidanceManeuver_1);

						avoidanceManeuver_2->setTimestamp(timeCalculator.count());
						avoidanceManeuver_2->setUuidTo(uuidTo);
						avoidanceManeuver_2->setTimestampAction(timeCalculator.count());
						avoidanceManeuver_2->setSpeedAction(distanceSecondToMergePoint/(distanceMergingVehicleToPoint/mergingVehicle->getSpeed()) + mergingVehicle->getLength()/mergingVehicle->getSpeed());

						Waypoint * waypointSecondNeighbour = new Waypoint();
						waypointSecondNeighbour->setTimestamp(timeCalculator.count() + (distanceMergingVehicleToPoint/mergingVehicle->getSpeed())*1000 + (mergingVehicle->getLength()/mergingVehicle->getSpeed())*1000);
						waypointSecondNeighbour->setLatitude(latitudeMerge);
						waypointSecondNeighbour->setLongitude(longitudeMerge);
						waypointSecondNeighbour->setSpeed(mergingVehicle->getSpeed());
						avoidanceManeuver_2->addWaypoint(waypointSecondNeighbour);
						trajectories.push_back(avoidanceManeuver_2);


						return trajectories;


					}

					else if(headingFirstToMergePoint >= 270 && headingFirstToMergePoint <= 90 && headingSecondToMergePoint <= 270 && headingSecondToMergePoint >= 90) {

						timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
						avoidanceManeuver_1->setTimestamp(timeCalculator.count());
						avoidanceManeuver_1->setUuidVehicle(firstNeighbour->getUuid());
						avoidanceManeuver_1->setUuidTo(uuidTo);
						avoidanceManeuver_1->setTimestampAction(timeCalculator.count());
						avoidanceManeuver_1->setLongitudeAction(firstNeighbour->getLongitude());
						avoidanceManeuver_1->setLatitudeAction(firstNeighbour->getLatitude());
						avoidanceManeuver_1->setSpeedAction(distanceFirstToMergePoint/(distanceMergingVehicleToPoint/mergingVehicle->getSpeed()) + mergingVehicle->getLength()/mergingVehicle->getSpeed());
						avoidanceManeuver_1->setLanePositionAction(firstNeighbour->getLanePosition());

						Waypoint * waypointFirstNeighbour = new Waypoint();
						waypointFirstNeighbour->setTimestamp(timeCalculator.count() + (distanceMergingVehicleToPoint/mergingVehicle->getSpeed())*1000 + (mergingVehicle->getLength()/mergingVehicle->getSpeed())*1000);
						waypointFirstNeighbour->setLatitude(latitudeMerge);
						waypointFirstNeighbour->setLongitude(longitudeMerge);
						waypointFirstNeighbour->setSpeed(mergingVehicle->getSpeed());
						waypointFirstNeighbour->setLanePosition(firstNeighbour->getLanePosition());
						avoidanceManeuver_1->addWaypoint(waypointFirstNeighbour);
						trajectories.push_back(avoidanceManeuver_1);

						avoidanceManeuver_2->setTimestamp(timeCalculator.count());
						avoidanceManeuver_2->setUuidTo(uuidTo);
						avoidanceManeuver_2->setTimestampAction(timeCalculator.count());

						Waypoint * waypointSecondNeighbour = new Waypoint();
						waypointSecondNeighbour->setTimestamp(timeCalculator.count());
						avoidanceManeuver_2->addWaypoint(waypointSecondNeighbour);
						trajectories.push_back(avoidanceManeuver_2);

						return trajectories;

					}
					return trajectories;
				}
			} else {
				cout << "There is no vehicle attempting to merge.\n";
			}
		}
	} else {
	return trajectories;
	}
}
