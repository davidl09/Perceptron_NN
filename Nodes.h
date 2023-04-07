//
// Created by dalae on 2023-04-06.
//

#ifndef CNN_1_NODES_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define NUM_LAYERS 10
#define NODES_PER_LAYER 10
#define BATCH_SIZE 20
#define relu(num) num > 0 ? num : 0
#define sigmoid(num) (1/(long double)(1+exp(-num)))
#define rev_sigmoid(num) (-1*logl(1/(long double)num - 1))
#define rand_1_0 (rand()%2 == 0 ? 1 : -1)

struct Node{
    long double value;
    long double weights[NODES_PER_LAYER];
};

struct Network{
    struct Node node1;
    struct Node node[NUM_LAYERS][NODES_PER_LAYER];
    struct Node node_out;
};

void initialize_weight(struct Node* node); //initialize randomized weights for nodes to start training
void initialize_network(struct Network* network);
void propagate_main(struct Node* node, struct Node* target_node, int pos_x); //propagate one node's value to the next
long double predict(struct Network* network, long double input);
long double mse(long double validation[BATCH_SIZE], long double predictions[BATCH_SIZE]); //calculate the mse of a training batch from prediction and validation arrays.
void print_weights(struct Network* network);

#define CNN_1_NODES_H

#endif //CNN_1_NODES_H
