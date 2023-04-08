//
// Created by dalae on 2023-04-06.
//

#include "Nodes.h"


void initialize_weight(struct Node* node){
    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        node->weights[i] = rand_1_0*(double)rand()/10000;
    }
}

void print_weights(struct Network* network) {
    printf("Weights for layer 2:\n");
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            printf("Layer 3 Node 4 weight %d: %LF\n", j, network->node[2][4].weights[j]);
        }
        printf("\n");

}

void init_train_data(long double pred_valid[3][BATCH_SIZE]){
    for (int i = 0; i < BATCH_SIZE; ++i) {
        pred_valid[INPUT][i] = rand_1_0*rand()%101;
        pred_valid[VALID][i] = sinl(pred_valid[INPUT][i]);
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
    target_node->value += (node->value)*(target_node->weights[pos_x]);
}



void compute_gradients(struct Network* network, long double error_out){
    //compute the gradient of the output node
    network->node_out.gradient = error_out * dx_sigmoid(network->node_out.value);

    //compute the gradients of the hidden nodes
    for (int i = NUM_LAYERS - 1; i >= 0; --i) {
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            long double gradient = 0;
            for (int k = 0; k < NODES_PER_LAYER; ++k) {
                if(i == NUM_LAYERS - 1){
                    gradient += network->node_out.gradient * network->node_out.weights[0];
                }else {
                    gradient += network->node[i + 1][k].gradient * network->node[i + 1][k].weights[j];
                }
            }
            network->node[i][j].gradient = gradient * dx_sigmoid(network->node[i][j].value);
        }
    }

    //compute a gradient for weights of each node
    for (int i = NUM_LAYERS - 1; i >= 0 ; --i) {
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            for (int k = 0; k < NODES_PER_LAYER; ++k) {
                network->node[i][j].weight_gradients[k] = network->node[i][j].gradient * network->node[i][j].value;
            }
        }
    }

    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        network->node_out.weight_gradients[i] = network->node_out.gradient * network->node[NUM_LAYERS-1][i].value;
    }
}

void update_weights(struct Network* network){

    for (int i = 0; i < NODES_PER_LAYER; ++i) {//update weights for first and last nodes
        network->node[0][i].weights[0] -= LEARNING_RATE * network->node[0][i].weight_gradients[0];
        network->node_out.weights[i] -= LEARNING_RATE * network->node_out.weight_gradients[i];

    }
    for (int i = 1; i < NUM_LAYERS; ++i) { //update weights for second through last hidden nodes
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            for (int k = 0; k < NODES_PER_LAYER; ++k) {
                network->node[i][j].weights[k] -= LEARNING_RATE * network->node[i][j].weight_gradients[k];
            }
        }
    }

}

long double predict(struct Network* network, long double predict){

    network->node1.value = predict;

    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        propagate_node(&network->node1, &network->node[0][i], 0);
    }
    for (int i = 0; i < NUM_LAYERS - 1; ++i) {
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            for (int k = 0; k < NODES_PER_LAYER; ++k) {
                propagate_node(&network->node[i][j], &network->node[i+1][k], j);
            }
        }
    }
    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        propagate_node(&network->node[NUM_LAYERS-1][i], &network->node_out, i);
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

