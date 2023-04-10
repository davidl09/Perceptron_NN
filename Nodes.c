//
// Created by dalae on 2023-04-10.
//

#include "Nodes.h"

long double learning_rate;

void init_train_data(long double pred_valid[3][BATCH_SIZE]){
    for (int i = 0; i < BATCH_SIZE; ++i) {
        pred_valid[INPUT][i] = rand_1_0*(rand()%6300)/(float)1000;
        pred_valid[VALID][i] = 3*(pred_valid[INPUT][i]);
    }
}

void initialize_weight(struct Node* node){
    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        node->weights[i] = sqrtl(6)/sqrtl(2);
    }
}

void initialize_network(struct Network* network){
    for (int i = 0; i < NUM_LAYERS; ++i) {
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            initialize_weight(&network->node[i][j]);
            network->node[i][j].value = 0;
        }
    }
}

void scale_inputs(long double train_pred_valid[3][BATCH_SIZE]){
    float max_size = 0;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        absl(train_pred_valid[INPUT][i]) > max_size ? max_size = (float)train_pred_valid[INPUT][i] : max_size; //find largest element of input array and set scale factor to scale data within range[-1, 1]
    }
    for (int i = 0; i < BATCH_SIZE; ++i) {
        train_pred_valid[INPUT][i] /= max_size;
    }
}

void propagate_node(struct Node* node, struct Node* target_node, int pos_x){
    target_node->value += (relu(node->value))*(target_node->weights[pos_x]);
}

void compute_gradients(struct Network* network, long double error_out){
    //compute the gradients of the hidden nodes

    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) {
        network->node[NUM_LAYERS - 1][i].gradient = error_out * dx_relu(network->node[NUM_LAYERS - 1][i].value);
    }

    long double gradient;
    for (int i = NUM_LAYERS - 2; i > 0; --i) {//i never = 0 since input nodes have no weights
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            gradient = 0;
            if(i == 1){
                for (int k = 0; k < INPUT_LAYER_NODES; ++k) {
                    gradient += network->node[i+1][k].gradient * network->node[i+1][k].weights[j];//compute weights for vectors from input layer
                }
            }
            else if(i == NUM_LAYERS - 2){
                for (int k = 0; k < OUTPUT_LAYER_NODES; ++k) {
                    gradient += network->node[i+1][k].gradient * network->node[i+1][k].weights[j];//compute weights for vectors from output
                }
            }
            else{
                for (int k = 0; k < NODES_PER_LAYER; ++k) {
                    gradient += network->node[i+1][k].gradient * network->node[i+1][k].weights[j];//compute weights between hidden layers
                }
            }
            network->node[i][j].gradient = gradient * dx_relu(network->node[i][j].value);
        }
    }

    for (int i = NUM_LAYERS - 2; i > 0 ; --i) {    //compute a gradient for weights of each hidden node
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            for (int k = 0; k < NODES_PER_LAYER; ++k) {
                if(k >= INPUT_LAYER_NODES){
                    continue;
                }
                network->node[i][j].weight_gradients[k] = network->node[i][j].gradient * network->node[i][j].value;
            }
        }
    }

    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) { //compute a gradient for each weight of output layer node
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            network->node[NUM_LAYERS - 1][i].weight_gradients[j] = network->node[NUM_LAYERS - 1][i].gradient * network->node[NUM_LAYERS-1][i].value;
        }
    }
}

void update_weights(struct Network* network){

    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) { //adjust weights for output nodes
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            network->node[NUM_LAYERS - 1][i].weights[j] -= learning_rate * network->node[NUM_LAYERS - 1][i].weight_gradients[j];
        }
    }

    for (int i = 0; i < NODES_PER_LAYER; ++i) { //adjust weights for first hidden layer
        for (int j = 0; j < INPUT_LAYER_NODES; ++j) {
            network->node[1][i].weights[j] -= learning_rate * network->node[1][i].weight_gradients[j];
        }
    }

    for (int i = 1; i < NUM_LAYERS; ++i) { //update weights for second through last hidden nodes
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            for (int k = 0; k < NODES_PER_LAYER; ++k) {
                network->node[i][j].weights[k] -= learning_rate * network->node[i][j].weight_gradients[k];
            }
        }
    }

}
