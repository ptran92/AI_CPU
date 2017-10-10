#include <iostream>
#include <cmath>
#include "matrix.h"

class LossFunction
{
public:
  virtual float CalcLoss(Matrix& e_out, Matrix& neural_out) = 0;
  virtual Matrix Derivative(Matrix& e_out, Matrix& neural_out) = 0;
};

class Quadratic : public LossFunction
{
public:
  float CalcLoss(Matrix& e_out, Matrix& neural_out)
  {
    float loss = 0.0;
    int total_output = e_out.Cols();

    for(int i = 0; i < total_output; i++)
      loss += (e_out.m[0][i] - neural_out.m[0][i]) * (e_out.m[0][i] - neural_out.m[0][i]) * 0.5;

    return loss;
  }

  Matrix Derivative(Matrix& e_out, Matrix& neural_out)
  {
    Matrix temp(e_out.Rows(), e_out.Cols());

    temp = (e_out - neural_out).ConstantMult(-1);

    return temp;
  }

};

class CrossEntropy : public LossFunction
{
public:
  float CalcLoss(Matrix& e_out, Matrix& neural_out)
  {
    float loss = 0.0;
    int total_output = e_out.Cols();

    for(int i = 0; i < total_output; i++)
      loss += e_out.m[0][i] * log(neural_out.m[0][i]) + (1 - e_out.m[0][i]) * log(1 - neural_out.m[0][i]) ;

    return (-loss);
  }

  Matrix Derivative(Matrix& e_out, Matrix& neural_out)
  {
    Matrix temp(e_out.Rows(), e_out.Cols());

    temp = (e_out.CoeffDiv(neural_out)).ConstantMult(-1) + e_out.ConstantSubstract(1).CoeffDiv(neural_out.ConstantSubstract(1));

    return temp;
  }

};
