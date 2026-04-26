#include "value.h"
#include <iostream>
#include <utility>
#include <functional>

namespace ml
{
    Value::Value(double data, std::vector<Value> previous, Op operation, std::string label)
    {
        this->data = data;
        this->previous_ = previous;
        this->operation_ = operation;
        this->label_ = label;
    }

    Value::~Value() = default;

    // Implement the operator overloads on the data (i.e. doubles)
    // const as we're not modifying this!
    Value Value::operator+(const Value& rhs) const {
        return Value(this->data + rhs.data, {*this, rhs}, Op::Add, "");
    }
    Value Value::operator-(const Value& rhs) const {
        return Value(this->data - rhs.data, {*this, rhs}, Op::Sub, "");
    }
    Value Value::operator*(const Value& rhs) const {
        return Value(this->data * rhs.data, {*this, rhs}, Op::Mul, "");
    }
    Value Value::operator/(const Value& rhs) const {
        return Value(this->data / rhs.data, {*this, rhs}, Op::Div, "");
    }

    void Value::backward()
    {
        backward_();
    }

    // Overload the << operator for easy printing of Value objects
    std::ostream& operator<<(std::ostream& os, const Value& v)
    {
        os << "Value(" << v.label_ << " | data=" << v.data << ", grad=" << v.grad << ")";
        return os;
    }
}