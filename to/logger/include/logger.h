//
// Created by Frédéric Gardes<frederic.gardes@orange.com> on 5/27/19.
//

#ifndef COMMUNICATION_LAYER_LOGGER_H
#define COMMUNICATION_LAYER_LOGGER_H

#include <string>
#include <fstream>


namespace logger {
    auto write(const std::string &text) -> void;
}

#endif //COMMUNICATION_LAYER_LOGGER_H
