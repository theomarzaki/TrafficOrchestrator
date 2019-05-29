//
// Created by Frédéric Gardes<frederic.gardes@orange.com> on 5/27/19.
//

#include "logger.h"

auto logger::write(const std::string &text) -> void {
    std::ofstream log_file("logs/to_logs.log", std::ios_base::out | std::ios_base::app);
    log_file << text << std::endl;
}