//
// Created by dalae on 2023-04-10.
//

#ifndef CNN_2_NODES_H
#define CNN_2_NODES_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_LAYER_NODES 1
#define OUTPUT_LAYER_NODES 1
#define model_func(num) (sinl(num))
#define NUM_LAYERS 8
#define NODES_PER_LAYER 20
#define BATCH_SIZE 1000 //training samples per batch
#define EPOCHS 50
#define relu(num) (num > 0 ? num : 0)
#define dx_relu(num) (num > 0 ? 1 : 0)
#define rev_sigmoid(num) (-1*logl(1/(double)num - 1))
#define absl(x) (x > 0 ? x : -x)
#define rand_1_0 (rand()%2 == 0 ? 1 : -1)
#define SCALE_FACT (M_PI)
#define INPUT 0
#define PRED 1
#define VALID 2

struct Node{
    long double value;
    long double weights[NODES_PER_LAYER]; //stored as reference to the scalar multiplier for the value of the output of the node at that index in the previous layer
    long double weight_gradients[NODES_PER_LAYER]; //first node has no weights
    long double gradient;
    //long double m[NODES_PER_LAYER];
    //long double v[NODES_PER_LAYER];
};

struct Network{
    struct Node node[NUM_LAYERS][NODES_PER_LAYER];
};

long double sigmoid(long double input);
long double dx_sigmoid(long double input);
long double dx_tanhl(long double input);
void initialize_weight(struct Node* node); //initialize randomized weights for nodes to start training
void save_weights(struct Network* network);
void init_train_data(long double pred_valid[3][BATCH_SIZE][INPUT_LAYER_NODES]);
void initialize_network(struct Network* network);
void scale_inputs(long double train_pred_valid[3][BATCH_SIZE]);
void propagate_node(struct Node* node, struct Node* target_node, int pos_x); //propagate one node's value to the next
void predict(struct Network* network, long double input[INPUT_LAYER_NODES]);
void compute_gradients(struct Network* network, long double expected_output[OUTPUT_LAYER_NODES]);
void update_weights(struct Network* network);
void update_weights_adam(struct Network* network, int t);
long double error(long double pred[OUTPUT_LAYER_NODES], long double valid[OUTPUT_LAYER_NODES]); //calculate the mse of a training batch from prediction and validation arrays.
long double mse(long double data[3][BATCH_SIZE][OUTPUT_LAYER_NODES]);
void print_weights(struct Network* network);

#endif //CNN_2_NODES_H
