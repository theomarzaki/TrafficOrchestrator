//
// Created by Frédéric Gardes<frederic.gardes@orange.com> on 5/27/19.
//

#include "include/logger.h"

auto logger::write(const std::string &text) -> void {
    std::ofstream log_file("logs/to_logs.log", std::ios_base::out | std::ios_base::app);
    log_file << text << std::endl;
}

auto logger::wipeOutDumpFile() -> void {
    std::ofstream log_file("logs/Montlhery_dump.json", std::ios_base::out | std::ios_base::trunc);
}

// for str in $(cat Montlhery_dump.json| pcregrep -o1 "timestamp\":([0-9]*)," | sort -n); do; cat Montlhery_dump.json | grep $str > Montlhery_dump_final.json; done
auto logger::dumpToFile(const std::string &text) -> void {
    std::ofstream log_file("logs/Montlhery_dump.json", std::ios_base::out | std::ios_base::app);
    log_file << text << "," << std::endl;
}