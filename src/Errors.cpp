#include "Errors.h"

#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace {

std::mutex err_mutex;

}

void npuemulator::GreaterZeroOrDie(const std::string &op, const std::string &name, int val)
{
    if (val <= 0) {
        std::lock_guard<std::mutex> lg(err_mutex);
        std::ostringstream msg;
        msg << "npuemulator:" << op << ": " << name << " is less than zero!";
        std::cerr << msg.str() << std::endl;
        throw std::invalid_argument(msg.str());
    }
}

void npuemulator::EqualOrDie(const std::string &op, const std::string &name1, int val1, const std::string &name2, int val2)
{
    if (val1 != val2) {
        std::lock_guard<std::mutex> lg(err_mutex);
        std::ostringstream msg;
        msg << "npuemulator: " << op << ": " << name1 << "(" << val1 << ") != " << name2 << "(" << val2 << ")!";
        std::cerr << msg.str() << std::endl;
        throw std::invalid_argument(msg.str());
    }
}

void npuemulator::GreaterOrEqualOrDie(const std::string &op, const std::string &name1, int val1, const std::string &name2, int val2)
{
    if (val1 < val2) {
        std::lock_guard<std::mutex> lg(err_mutex);
        std::ostringstream msg;
        msg << "npuemulator: " << op << ": " << name1 << "(" << val1 << ") < " << name2 << "(" << val2 << ")!";
        std::cerr << msg.str() << std::endl;
        throw std::invalid_argument(msg.str());
    }
}