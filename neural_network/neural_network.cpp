/****************************************************
 *    INCLUDES
 ****************************************************/
#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <memory>
#include <chrono>
#include "network.h"
#include "mnist_data.h"

/****************************************************
 *    DIRECTORIES
 ****************************************************/
#define TRAINING_DATA_FILE      "train-images.idx3-ubyte"
#define TRAINING_LABEL_FILE     "train-labels.idx1-ubyte"

#define TESTING_DATA_FILE       "t10k-images.idx3-ubyte"
#define TESTING_LABEL_FILE      "t10k-labels.idx1-ubyte"

/****************************************************
 *    PARAMETERS
 ****************************************************/
/* network parameter */
#define BATCH_SIZE                        20
#define LEARNING_RATE                     0.1
#define EPOCH_TIME                        10
#define HIDDEN_LAYERS                     2
#define NEURONS_PER_HIDDEN_LAYER          64

/****************************************************
 *    MAIN
 ****************************************************/
void network_validation(Network& net)
{
  Mnist_Parser test_set(TESTING_DATA_FILE, TESTING_LABEL_FILE);
  std::shared_ptr<float> single_neural_output(new float[test_set.total_class]);
  int total_test_images = test_set.total_image;
  int total_pxs         = test_set.img_rows * test_set.img_cols;
  int total_class       = test_set.total_class;
  float *image         = test_set.image.get();
  float *res           = test_set.label.get();
  float accuracy       = 0.0;

  std::cout << "/**************** TEST RESULT ****************/" << std::endl;

  for(int i = 0; i < total_test_images; i++)
  {
    float *input          = image + (i * total_pxs);
    float *e_output       = res + (i * total_class);
    float *neural_output  = single_neural_output.get();

    // Feed the network with test data
    net.Predict(input, neural_output);

    // Calculate error percentage
    float loss             = 0.0;
    int    predict_num      = 0;
    int    true_num         = 0;
    float predict_max_prob = neural_output[0];
    float true_max_prob    = e_output[0];

    for(int c = 0; c < total_class; c++)
    {
      loss += 0.5 * (e_output[c] - neural_output[c]) * (e_output[c] - neural_output[c]);

      if(predict_max_prob < neural_output[c])
      {
        predict_max_prob = neural_output[c];
        predict_num      = c;
      }

      if(true_max_prob < e_output[c])
      {
        true_max_prob = e_output[c];
        true_num      = c;
      }
    }

    loss /= total_class;

    if(predict_num == true_num)
    {
      accuracy += 1.0;
    }

    if(i < 10)
    {
      std::cout << "\tExpect : " << true_num << std::endl;
      std::cout << "\tPredict: " << predict_num << std::endl <<  std::endl;
    }
  }

  std::cout << "Average accuracy: " << (accuracy * 100.0 / total_test_images) << std::endl;

}

int main(int argc, char const *argv[])
{
  /* code */
  srand(time(NULL));

  /* Initialize training data */
  Mnist_Parser training_set(TRAINING_DATA_FILE, TRAINING_LABEL_FILE);

  /* initialize network */
  CrossEntropy loss_func;
  int input_size = training_set.img_cols * training_set.img_rows;
  int output_size = training_set.total_class;

  Network net(input_size, output_size, HIDDEN_LAYERS, NEURONS_PER_HIDDEN_LAYER, &loss_func);

  // Measure time
  /* train network with training data */
  std::chrono::system_clock::time_point start   = std::chrono::system_clock::now();
  net.Train(training_set.image.get(), training_set.label.get(), training_set.total_image, BATCH_SIZE, LEARNING_RATE, EPOCH_TIME);
  std::chrono::system_clock::time_point end     = std::chrono::system_clock::now();
  std::chrono::milliseconds             elapsed_millisecs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  /* test the network */
  network_validation(net);

  // Print the elapsed time
  std::cout << "Time elapsed: " << (double)(elapsed_millisecs.count())/1000/60 << " minutes" << std::endl;

  return 0;
}
