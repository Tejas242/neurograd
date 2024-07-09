#pragma once

#include <iostream>
#include <cmath>
#include <set>
#include <functional>
#include <vector>
#include <cassert>

class Value {
public:
    double data;
    double grad;
    std::function<void()> _backward;
    std::set<Value*> _prev;
    std::string _op;

    Value(double data, std::initializer_list<Value*> children = {}, std::string op = "");
    Value operator+(const Value& other) const;
    Value operator*(const Value& other) const;
    Value pow(double other) const;
    Value relu() const;
    Value tanh() const;
    void backward();
    Value operator-() const;
    Value operator-(const Value& other) const;
    Value operator/(const Value& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Value& v);
};

Value operator+(double scalar, const Value& v);
Value operator*(double scalar, const Value& v);
Value operator-(double scalar, const Value& v);
Value operator/(double scalar, const Value& v);
