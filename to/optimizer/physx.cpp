//
// Created by jab on 11/07/19.
//

#include <cmath>
#include "include/physx.h"

#define AVERAGE_CAR_MASS 1200 // kg
#define AVERAGE_CAR_DENSITY 110 // Average mass per square meter in kg
#define BRAKE_PERFORMANCE_COEF 0.6

#define BRAKE_AVERAGE_DISSIPATION_CONSTENT 12000 // Average number of Joules dissipated per meter.
#define HUMAN_REACTION_TIME 1

double Physx::getSecurityDistance(double carSpeed, double objectSpeed, double carMass) {
    auto minDistance{carSpeed*HUMAN_REACTION_TIME};
    auto distance{0.0};
    if (carSpeed > objectSpeed) {
        distance = 0.5 * carMass * (std::pow(carSpeed, 2) - std::pow(objectSpeed, 2))  / (BRAKE_AVERAGE_DISSIPATION_CONSTENT*(carMass/AVERAGE_CAR_MASS)*BRAKE_PERFORMANCE_COEF) + minDistance;
    }
    return distance > minDistance ? distance : minDistance;
}

double Physx::getCarMass(double height, double length, double width) {
    if (height < 0.1 or length < 0.1 or width < 0.1) {
        return AVERAGE_CAR_MASS;
    } else {
        return height*length*width*AVERAGE_CAR_DENSITY;
    }
}

//double Physx::getDistanceWithSpeedLimitAndjustAndLinearAcceleration(double deltaT,
//                                                                    double acceleration,
//                                                                    double speed,
//                                                                    double speedLimit) {
//    auto deltaV{speedLimit - speed};
//    auto needTime{std::fabs(deltaV/acceleration)};
//
//    if (needTime < deltaT) {
//        (deltaT-needTime)*speedLimit+(deltaT-(deltaT-needTime))*speed+0.5*acceleration*needTime*needTime;
//    } else {
//
//    }
//}