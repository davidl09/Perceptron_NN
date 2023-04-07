//
// Created by dalae on 2023-04-06.
//

#ifndef CNN_1_NODES_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define NUM_LAYERS 20
#define NODES_PER_LAYER 20
#define BATCH_SIZE 20
#define relu(num) num > 0 ? num : 0
#define sigmoid(num) (1/(long double)(1+exp(-num)))
#define dx_sigmoid(num) sigmoid(num)*(1-sigmoid(num))
#define rev_sigmoid(num) (-1*logl(1/(long double)num - 1))
#define rand_1_0 (rand()%2 == 0 ? 1 : -1)
#define INPUT 0
#define PRED 1
#define VALID 2

struct Node{
    long double value;
    double weights[NODES_PER_LAYER]; //stored as reference to the scalar multiplier for the node at index in next layer (points downward). Last layer has all weights stored at 0 (all pointing to single output node).
    double weight_gradients[NODES_PER_LAYER];
    double gradient;
};

struct Network{
    struct Node node1;
    struct Node node[NUM_LAYERS][NODES_PER_LAYER];
    struct Node node_out;
};

void initialize_weight(struct Node* node); //initialize randomized weights for nodes to start training
void save_weights(struct Network* network);
void initialize_network(struct Network* network);
void propagate_node(struct Node* node, struct Node* target_node, int pos_x); //propagate one node's value to the next
long double predict(struct Network* network, long double input);
void compute_gradients(struct Network* network, double error_out);
void update_weights(struct Network* network);
long double mse(long double pred_valid[3][BATCH_SIZE]); //calculate the mse of a training batch from prediction and validation arrays.
void print_weights(struct Network* network);

#define CNN_1_NODES_H

#endif //CNN_1_NODES_H
