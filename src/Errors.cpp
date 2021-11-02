#include "Errors.h"

#include <sstream>
#include <stdexcept>

void npuemulator::GreaterZeroOrDie(const std::string &op, const std::string &name, int val)
{
    if (val <= 0) {
        throw std::invalid_argument("npuemulator:" + op + ": " + name + " is less than zero!");
    }
}

void npuemulator::EqualOrDie(const std::string &op, const std::string &name1, int val1, const std::string &name2, int val2)
{
    if (val1 != val2) {
        std::ostringstream msg;
        msg << "npuemulator: " << op << ": " << name1 << "(" << val1 << ") != " << name2 << "(" << val2 << ")!";
        throw std::invalid_argument(msg.str());
    }
}

void npuemulator::GreaterOrEqualOrDie(const std::string &op, const std::string &name1, int val1, const std::string &name2, int val2)
{
    if (val1 < val2) {
        std::ostringstream msg;
        msg << "npuemulator: " << op << ": " << name1 << "(" << val1 << ") < " << name2 << "(" << val2 << ")!";
        throw std::invalid_argument(msg.str());
    }
}