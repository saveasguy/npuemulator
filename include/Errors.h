#ifndef ERRORS_H
#define ERRORS_H

#include <string>

namespace npuemulator {

void GreaterZeroOrDie(const std::string &op, const std::string &name, int val);

void EqualOrDie(const std::string &op, const std::string &name1, int val1, const std::string &name2, int val2);

void GreaterOrEqualOrDie(const std::string &op, const std::string &name1, int val1, const std::string &name2, int val2);

}

#endif
