//
// Created by dalae on 2023-04-06.
//

#include "Nodes.h"


void initialize_weight(struct Node* node){
    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        node->weights[i] = (double)rand()/5000;
        node->weights[i] *= rand_1_0;
    }
}

void print_weights(struct Network* network){
    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        printf("%F", network->node1.weights[i]);
    }

    for (int i = 0; i < NUM_LAYERS - 1; ++i) {
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            for (int k = 0; k < NODES_PER_LAYER; ++k) {
                printf("%f,", network->node[i][j].weights[k]);
            }
            printf("\n");
        }
        printf("\n\n\n");
    }
}

void initialize_network(struct Network* network){
    initialize_weight(&network->node1);
    network->node1.value = 0;
    for (int i = 0; i < NUM_LAYERS; ++i) {
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            initialize_weight(&network->node[i][j]);
            network->node[i][j].value = 0;
        }
    }
    initialize_weight(&network->node_out);
}

void propagate_node(struct Node* node, struct Node* target_node, int pos_x){
    node->value = sigmoid(node->value);
    target_node->value = target_node->value + (node->value)*(node->weights[pos_x]);
}

void compute_gradients(struct Network* network, double error_out){
    //compute the gradient of the output node
    network->node_out.gradient = error_out * dx_sigmoid(network->node_out.value);

    //compute the gradients of the hidden nodes
    for (int i = NUM_LAYERS - 1; i >= 0; ++i) {
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            long double gradient = 0;
            for (int k = 0; k < NODES_PER_LAYER; ++k) {
                if(i == NUM_LAYERS - 1){
                    gradient += network->node_out.gradient * network->node_out.weights[0];
                }else{
                    gradient += network->node[i+1][k].gradient * network->node[i+1][k].weights[j];
                }
            }
        }
    }
};

long double predict(struct Network* network, long double predict){

    network->node1.value = predict;

    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        propagate_node(&network->node1, &network->node[0][i], i);
    }
    for (int i = 0; i < NUM_LAYERS - 1; ++i) {
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            for (int k = 0; k < NODES_PER_LAYER; ++k) {
                propagate_node(&network->node[i][j], &network->node[i+1][k], j);
            }
        }
    }
    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        propagate_node(&network->node[NUM_LAYERS-1][i], &network->node_out, 0);
    }
    return network->node_out.value;
}

long double mse(long double pred_valid[3][BATCH_SIZE]){
    long double error = 0;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        error += powl(pred_valid[PRED][i] - pred_valid[VALID][i], 2);
    }
    error /= BATCH_SIZE;
    return sqrtl(error);
}

