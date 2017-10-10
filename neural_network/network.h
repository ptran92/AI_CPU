#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "matrix.h"
#include "activation.h"
#include "loss_function.h"

class Network
{
private:
  struct layer_t
  {
  public:
    layer_t(int in_size, int out_size):
      w(in_size, out_size, 1),
      b(1, out_size, 1),
      w_grad(in_size, out_size),
      b_grad(1, out_size),
      out(1, out_size),
      error(1, out_size)
    {}

  public:
    Matrix w;
    Matrix b;
    Matrix w_grad;
    Matrix b_grad;
    Matrix out;
    Matrix error;
    const Matrix* in;
  };

public:
  /**********************************
   *  CONSTRUCTOR AND DESTRUCTOR
   **********************************/
  Network(int input_size, int output_size, int n_hidd_layers, int n_hid_neurons, LossFunction* loss_func):
    input_size(input_size),
    output_size(output_size),
    no_layers(n_hidd_layers + 1), // Plus 1 for output layer
    no_hidden_neurons(n_hid_neurons),
    v_layers(n_hidd_layers + 1), // Plus 1 for output layer
    loss(loss_func)
  {
    std::cout << "Initialize network, input " << input_size << " output " << output_size << std::endl;

    /* Initiate layer */
    for(size_t l = 0; l < no_layers; l++)
    {
      int in_neurons;
      int out_neurons;

      if(l == 0) // first hidden layer
      {
        in_neurons = input_size;
        out_neurons = no_hidden_neurons;
      }
      else if(l == (no_layers - 1)) // output layer
      {
        in_neurons = no_hidden_neurons;
        out_neurons = output_size;
      }
      else // others layers
      {
        in_neurons = no_hidden_neurons;
        out_neurons = no_hidden_neurons;
      }

      v_layers[l].reset(new layer_t(in_neurons, out_neurons));

    }

  }

  virtual ~Network(){}

  /**********************************
   *  TRAINING METHOD
   **********************************/
  void Train(const float* input, const float* expect_output, int training_size, int b_size, float lr, int epoch = 10)
  {
    std::cout << "Train the network" << std::endl;

    // Save away training parameters
    batch_size  = b_size;
    eta         = lr;

    // Calculate total of training batches to feed into the network
    int n_batches = training_size / batch_size;

    // Allocate an array to store all loss values through training process
    // std::shared_ptr<float>loss_array(new float[n_batches]);
    float loss_val = 0.0;

    /*
      Training steps:
      For epoch time:
        For each batch in input training set:
          For sample in batch:
            + Do forward propagation
            + Calculate error signal of output layer
            + Back propagate error signal to inner layers
            + Accumulate weight and bias gradients for all layers
          End for

          Average the gradients and update weights and biases
        End for
      End for
     */
     while(epoch--) // Loop for epoch times
     {
       // Look through all batches
       for(int batch_idx = 0; batch_idx < n_batches; batch_idx++)
       {
         int random_batch_idx = rand() % n_batches;
         loss_val = 0.0;

         // Loop through all samples in a batch
         for(int b = 0; b < batch_size; b++)
         {
           /* Prepare a training sample */
          //  int sample_idx = batch_idx * batch_size + b;
           int sample_idx = random_batch_idx * batch_size + b; // select random batch
           Matrix in(1, input_size);
           Matrix e_out(1, output_size);

           for(int idx = 0, input_pos = sample_idx*input_size; idx < input_size; idx++)
              in.m[0][idx] = input[input_pos + idx];

           for(int idx = 0, output_pos = sample_idx*output_size; idx < output_size; idx++)
              e_out.m[0][idx] = expect_output[output_pos + idx];

           /* Forward propagation */
           Matrix& neural_output = forward_propagation(in);

           /* Backward propagation */
           backward_propagation(in, e_out, neural_output);

           /* Calculate loss value */
           loss_val += loss->CalcLoss(e_out, neural_output);
         }

         /* After examining a batch, update network weights and biases */
         network_update();

         /* Average this batch's loss value and save it */
         loss_val /= batch_size;

         if(batch_idx % 500 == 0)
         {
           std::cout << "Loss value: " << loss_val << std::endl;
         }

       }

       std::cout << "\t\tEpoch: " << epoch << std::endl;

     }
  }

  /**********************************
   *  PREDICT METHOD
   **********************************/
  void Predict(const float* input, float* output)
  {
    // Create input matrix whose size 1 x number of input neurons
    Matrix in(1, input_size);
    for(int col_idx = 0; col_idx < input_size; col_idx++)
    {
      in.m[0][col_idx] = input[col_idx];
    }

    // Do forward propagation
    Matrix& network_output = forward_propagation(in);

    // Copy to output buffer
    for(int row_idx = 0; row_idx < output_size; row_idx++)
    {
      output[row_idx] = network_output.m[0][row_idx];
    }
  }

private:
    /**********************************
    *  FORWARD PROPAGATION
    **********************************/
    Matrix& forward_propagation(const Matrix& input)
    {
      // Go through all layers
      for(size_t l = 0; l < no_layers; l++)
      {
        if(l != 0)
        {
          // If this layer is not the first hidden layer,
          // then input of current layer is the output of previous layer
          v_layers[l].get()->in = &v_layers[l-1].get()->out;
        }
        else
        {
          // If this is the first hidden layer, its input is the data
          v_layers[0].get()->in = &input;
        }

        Sigmoid activate_function;
        Matrix z = *(v_layers[l].get()->in) * v_layers[l].get()->w + v_layers[l].get()->b;
        if(l != (no_layers - 1))
        {
          // If this is not the output layer, use activation on output of neurons
          v_layers[l].get()->out = z.CoeffFunc<Sigmoid>(activate_function);
        }
        else
        {
          // If this is the output layer, apply softmax function since we are doing classification
          v_layers[l].get()->out = Softmax(z);
        }
      }

      return v_layers[no_layers - 1].get()->out;
    }

    /**********************************
    *  BACKWARD PROPAGATION
    **********************************/
    void backward_propagation(Matrix& in, Matrix& e_out, Matrix& neural_output)
    {
      /* Calculate error signal of output layer */
      Sigmoid_Derivative    output_derivative;
      Sigmoid_Derivative    hidden_derivative;
      for(int l = (no_layers - 1); l >= 0; l--)
      {
        if( l == (no_layers - 1) ) // for output layer
        {
          v_layers[l].get()->error = loss->Derivative(e_out, neural_output).CoeffMult(neural_output.CoeffFunc<Sigmoid_Derivative>(output_derivative));
        }
        else // for inner layers
        {
          v_layers[l].get()->error = (v_layers[l+1].get()->w * v_layers[l+1].get()->error.Transpose())\
                                      .Transpose()\
                                      .CoeffMult(v_layers[l].get()->out.CoeffFunc<Sigmoid_Derivative>(hidden_derivative));
        }

        /* Calculate gradients */
        v_layers[l].get()->w_grad = v_layers[l].get()->w_grad + \
                                    (v_layers[l].get()->in->Transpose() * v_layers[l].get()->error);
        v_layers[l].get()->b_grad = v_layers[l].get()->b_grad + v_layers[l].get()->error;
      }
    }

    /**********************************
    *  UPDATE WEIGHTS AND BIASES
    **********************************/
    void network_update()
    {
      // Loop through all layers and update
      for(size_t l = 0; l < v_layers.size(); l++)
      {
        v_layers[l].get()->w = v_layers[l].get()->w - v_layers[l].get()->w_grad.ConstantMult(eta / batch_size);
        v_layers[l].get()->b = v_layers[l].get()->b - v_layers[l].get()->b_grad.ConstantMult(eta / batch_size);

        // Clear the gradients for the next training session
        v_layers[l].get()->w_grad.Clear();
        v_layers[l].get()->b_grad.Clear();
      }
    }

public:
  /**********************************
   *  SOFTMAX FUNCTION
   **********************************/
   static Matrix Softmax(Matrix in)
   {
     int n_output = in.Cols();
     Matrix temp(in.Rows(), in.Cols());
     float sum = 0.0;
     float inversed_sum;
     Exponent exponent;

     for(int i = 0; i < n_output; i++)
        sum += exp(in.m[0][i]);

     inversed_sum = 1 / sum;

     temp = in.CoeffFunc<Exponent>(exponent).ConstantMult(inversed_sum);

     return temp;
   }

private:
  /**********************************
   *  NETWORK INTERNAL VARIABLES
   **********************************/
  /* Number of layers and neurons in input & output layer */
  int input_size;
  int output_size;
  int no_layers;
  int no_hidden_neurons;
  /* Store the layers */
  std::vector<std::shared_ptr<layer_t>> v_layers;
  /* Store parameters for backward propagation */
  float eta; // learning rate
  int batch_size;
  LossFunction* loss;
};
