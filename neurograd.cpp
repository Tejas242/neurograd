#include "neurograd.h"

Value::Value(double data, std::initializer_list<Value*> children, std::string op)
    : data(data), grad(0), _op(op), _backward([]() {}) {
    for (auto child : children) {
        _prev.insert(child);
    }
}

Value Value::operator+(const Value& other) const {
    Value out(this->data + other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "+");

    out._backward = [this, &other, &out]() {
        const_cast<Value*>(this)->grad += out.grad;
        const_cast<Value&>(other).grad += out.grad;
    };

    return out;
}

Value Value::operator*(const Value& other) const {
    Value out(this->data * other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "*");

    out._backward = [this, &other, &out]() {
        const_cast<Value*>(this)->grad += other.data * out.grad;
        const_cast<Value&>(other).grad += this->data * out.grad;
    };

    return out;
}

Value Value::pow(double other) const {
    assert((std::is_same<decltype(other), int>::value || std::is_same<decltype(other), double>::value) && "only supporting int/float powers for now");
    Value out(std::pow(this->data, other), {const_cast<Value*>(this)}, "**" + std::to_string(other));

    out._backward = [this, other, &out]() {
        const_cast<Value*>(this)->grad += (other * std::pow(this->data, other - 1)) * out.grad;
    };

    return out;
}

Value Value::relu() const {
    Value out(this->data < 0 ? 0 : this->data, {const_cast<Value*>(this)}, "ReLU");

    out._backward = [this, &out]() {
        const_cast<Value*>(this)->grad += (out.data > 0) * out.grad;
    };

    return out;
}

Value Value::tanh() const {
    double x = this->data;
    double t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
    Value out(t, {const_cast<Value*>(this)}, "tanh");

    out._backward = [this, t, &out]() {
        const_cast<Value*>(this)->grad += (1 - t * t) * out.grad;
    };

    return out;
}

void Value::backward() {
    std::vector<Value*> topo;
    std::set<Value*> visited;
    std::function<void(Value*)> build_topo;

    build_topo = [&build_topo, &visited, &topo](Value* v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (auto child : v->_prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(this);
    this->grad = 1;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}

Value Value::operator-() const {
    return *this * -1;
}

Value Value::operator-(const Value& other) const {
    return *this + (-other);
}

Value Value::operator/(const Value& other) const {
    return *this * other.pow(-1);
}

std::ostream& operator<<(std::ostream& os, const Value& v) {
    os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
    return os;
}

Value operator+(double scalar, const Value& v) {
    return v + scalar;
}

Value operator*(double scalar, const Value& v) {
    return v * scalar;
}

Value operator-(double scalar, const Value& v) {
    return Value(scalar) - v;
}

Value operator/(double scalar, const Value& v) {
    return Value(scalar) / v;
}
