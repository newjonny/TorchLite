#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include "ops.h"

namespace ml
{
    class Value
    {
        private:
            std::vector<Value> previous_;
            Op operation_;
            std::string label_;
            std::function<void()> backward_ = []{};
        public:
            double data;
            double grad = 0.0;

            Value(double data, std::vector<Value> previous, Op operation, std::string label);
            ~Value();

            Value operator+(const Value& rhs) const;
            Value operator-(const Value& rhs) const;
            Value operator*(const Value& rhs) const;
            Value operator/(const Value& rhs) const;

            void backward();

            // friend gives access to priv members w/o
            friend std::ostream& operator<<(std::ostream& os, const Value& v);
    };
}