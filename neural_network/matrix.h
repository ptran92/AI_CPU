#ifndef _MATRIX_H
#define _MATRIX_H

#include <iostream>
#include <random>
#define RANDOM_VAL_RANGE 20

class Matrix
{
public:
  /*****************************
   *    CONSTRUCTOR FOR ZERO MATRIX AND RANDOM MATRIX
   *****************************/
  Matrix(int rows, int cols):
    rows(rows),
    cols(cols)
  {
    /* Create a 2D array [rows x cols] */
    m = new float*[rows];
    for(int i = 0; i < rows; i++)
    {
      m[i] = new float[cols];

      // Reset all coefficient
      for(int j = 0; j < cols; j++)
      {
        m[i][j] = 0;
      }
    }

  }

  Matrix(int rows, int cols, int rand_seed):
    rows(rows),
    cols(cols)
  {
    std::uniform_real_distribution<float> uni_distribute(-0.5, 0.5);
    std::default_random_engine rand_gen;

    /* Create a 2D array [rows x cols] */
    m = new float*[rows];
    for(int i = 0; i < rows; i++)
    {
      m[i] = new float[cols];

      // Randomly initiate all coefficients
      for(int j = 0; j < cols; j++)
      {
        m[i][j] = uni_distribute(rand_gen);
      }

    }

  }

  /*****************************
   *    DESTRUCTOR
   *****************************/
  ~Matrix()
  {
    for(int i = 0; i < rows; i++)
    {
      delete[] m[i];
    }

    delete[] m;
  }

  /*****************************
   *    COPY CONSTRUCTOR
   *****************************/
  Matrix(const Matrix& obj)
  {
    rows = obj.rows;
    cols = obj.cols;

    // allocate memory and copy all coefficients
    m = new float* [rows];
    for(int i = 0; i < rows; i++)
    {
      m[i] = new float[cols];
      for(int j = 0; j < cols; j++)
      {
        m[i][j] = obj.m[i][j];
      }
    }
  }

  /*****************************
   *    OPERATORS
   *****************************/
  Matrix operator +(const Matrix& obj) const
  {
    Matrix temp(rows, cols);

    for(int i = 0; i < rows; i++)
      for(int j = 0; j < cols; j++)
      {
        temp.m[i][j] = m[i][j] + obj.m[i][j];
      }

    return temp;
  }

  Matrix operator -(const Matrix& obj) const
  {
    Matrix temp(rows, cols);

    for(int i = 0; i < rows; i++)
      for(int j = 0; j < cols; j++)
      {
        temp.m[i][j] = m[i][j] - obj.m[i][j];
      }

    return temp;
  }

  Matrix& operator =(const Matrix& obj)
  {
    for(int i = 0; i < rows; i++)
      for(int j = 0; j < cols; j++)
      {
        m[i][j] = obj.m[i][j];
      }

    return *this;
  }

  Matrix operator *(const Matrix& obj) const
  {
    int output_rows = rows;
    int output_cols = obj.cols;

    Matrix temp(output_rows, output_cols);

    for(int i = 0; i < output_rows; i++)
      for(int j = 0; j < output_cols; j++)
      {
        float sum = 0.0;

        for(int k = 0; k < cols; k++)
          sum += m[i][k] * obj.m[k][j];

        temp.m[i][j] = sum;
      }

    return temp;
  }

  /*****************************
   *    COEFFICIENT SUBTRACTED
   *****************************/
  Matrix ConstantSubstract(float num)
  {
    Matrix temp(rows, cols);

    for(int i = 0; i < rows; i++)
      for(int j = 0; j < cols; j++)
      {
        temp.m[i][j] = num - m[i][j];
      }

    return temp;
  }

  /*****************************
   *    COEFFICIENT MULTIPLICATION
   *****************************/
  Matrix CoeffMult(const Matrix& obj)
  {
    Matrix temp(rows, cols);

    for(int i = 0; i < rows; i++)
      for(int j = 0; j < cols; j++)
      {
        temp.m[i][j] = m[i][j] * obj.m[i][j];
      }

    return temp;
  }

  /*****************************
   *    COEFFICIENT DIVISION
   *****************************/
  Matrix CoeffDiv(const Matrix& obj)
  {
    Matrix temp(rows, cols);

    for(int i = 0; i < rows; i++)
      for(int j = 0; j < cols; j++)
      {
        temp.m[i][j] = m[i][j] / obj.m[i][j];
      }

    return temp;
  }

  /*****************************
   *    SPECIAL FUNCTION ON COEFFICIENTS
   *****************************/
  template<typename FuncType>
  Matrix CoeffFunc(FuncType& func)
  {
    Matrix temp(rows, cols);

    for(int i = 0; i < rows; i++)
      for(int j = 0; j < cols; j++)
      {
        temp.m[i][j] = func(m[i][j]);
      }

    return temp;
  }

  /*****************************
   *    CONSTANT MULTIPLICATION
   *****************************/
  Matrix ConstantMult(float val)
  {
    Matrix temp(rows, cols);

    for(int i = 0; i < rows; i++)
      for(int j = 0; j < cols; j++)
      {
        temp.m[i][j] = m[i][j] * val;
      }

    return temp;
  }

  /*****************************
   *    TRANSPOSE
   *****************************/
  Matrix Transpose() const
  {
    Matrix temp(cols, rows);

    for(int i = 0; i < cols; i++)
      for(int j = 0; j < rows; j++)
        temp.m[i][j] = m[j][i];

    return temp;
  }

  /*****************************
   *    MISCELLANEOUS
   *****************************/
   int Rows() const
   {
     return rows;
   }

   int Cols() const
   {
     return cols;
   }

   int Size() const
   {
     return rows * cols;
   }
   void Print() const
   {
     for(int i = 0; i < rows; i++)
     {
      for(int j = 0; j < cols; j++)
      {
        std::cout << m[i][j] << " ";
      }
      std::cout << std::endl;
     }
   }
   void Clear()
   {
     for(int i = 0; i < rows; i++)
      for(int j = 0; j < cols; j++)
        m[i][j] = 0.0;
   }

public:
  int rows;
  int cols;
  float **m;
};

#endif
