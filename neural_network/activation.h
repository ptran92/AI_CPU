#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include <iostream>
#include <cmath>
#include "matrix.h"

/* This is a functor which handle Singmoid activation function */
class Sigmoid
{
public:
  float operator()(float a)
  {
    return (1.0 / (1.0 + exp(-a)));
  }
};

class Sigmoid_Derivative
{
public:
  float operator()(float a)
  {
    return (a * (1.0 - a));
  }
};

class Exponent
{
public:
  float operator()(float a)
  {
    return exp(a);
  }
};

class Relu
{
public:
  float operator()(float a)
  {
    return ( (a > 0)? a : 0 );
  }
};

class Relu_Derivative
{
public:
  float operator()(float a)
  {
    return ( (a > 0)? 1 : 0 );
  }
};

#endif
