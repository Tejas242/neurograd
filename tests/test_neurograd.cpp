#include <iostream>
#include "neurograd.h"

int main() {
    Value a(2.0);
    Value b(3.0);

    Value c = a + b;
    c.backward();

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;

    Value d = a.tanh();
    d.backward();

    std::cout << d << std::endl;

    return 0;
}
