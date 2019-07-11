//
// Created by jab on 11/07/19.
//

#ifndef TO_PHYSX_H
#define TO_PHYSX_H

namespace Physx {
    double getSecurityDistance(double carSpeed, double objectSpeed, double carMass);
    double getCarMass(double height, double length, double width);
}


#endif //TO_PHYSX_H
