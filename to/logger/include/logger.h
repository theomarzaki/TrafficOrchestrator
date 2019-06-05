//
// Created by Frédéric Gardes<frederic.gardes@orange.com> on 5/27/19.
//

#ifndef TO_LOGGER_H
#define TO_LOGGER_H

#include <string>
#include <fstream>


namespace logger {
    auto write(const std::string& text) -> void;
}

#endif //TO_LOGGER_H
