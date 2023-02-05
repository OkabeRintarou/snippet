#include <fstream>
#include <streambuf>
#include "util.h"

bool ReadFile(const char *filename, std::string &content) {
    std::ifstream fin(filename, std::ios_base::in);
    if (fin.fail()) {
        return false;
    }
    std::string str{std::istreambuf_iterator<char>(fin),
        std::istreambuf_iterator<char>()};
    content.swap(str);
    return true;
}
