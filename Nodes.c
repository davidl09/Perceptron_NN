//
// Created by dalae on 2023-04-10.
//

#include "Nodes.h"

long double learning_rate;

void init_train_data(long double pred_valid[3][BATCH_SIZE][INPUT_LAYER_NODES]){
    //reinitializes the entire data array for new train values
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < INPUT_LAYER_NODES; ++j) {
            pred_valid[INPUT][i][j] = rand_1_0 * (long double)(rand()%10000)/300;
            while(absl(pred_valid[INPUT][i][j]) > M_PI){
                if(pred_valid[INPUT][i][j] > 0) //input data in range +- PI
                    pred_valid[INPUT][i][j] -= M_PI;
                else
                    pred_valid[INPUT][i][j] += M_PI;
            }
            pred_valid[VALID][i][j] = model_func(pred_valid[INPUT][i][j]);
        }
    }
}

void initialize_weight(struct Node* node){
    for (int i = 0; i < NODES_PER_LAYER; ++i) {
        node->weights[i] =  rand_1_0 * (long double)(rand()%1001)/500;
    }
}

long double sigmoid(long double input){
    return (long double)1/(1 + expl(-input));
}

long double dx_sigmoid(long double input){
    return tanhl(input) * ((long double)1 - tanhl(input));
}

long double dx_tanhl(long double input){
    return 1 - powl(tanhl(input), 2);
}

void initialize_network(struct Network* network){
    for (int i = 1; i < NUM_LAYERS; ++i) {
        for (int j = 0; j < (i == NUM_LAYERS - 1 ? OUTPUT_LAYER_NODES : NODES_PER_LAYER); ++j) {
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
    target_node->value += tanhl(node->value) * target_node->weights[pos_x];
}

void compute_gradients(struct Network* network, long double expected_output[OUTPUT_LAYER_NODES]){
    long double output;
    long double error;
    long double gradient_out;

    for (int i = 0; i < OUTPUT_LAYER_NODES; ++i) { // compute error and gradients for output nodes
        output = network->node[NUM_LAYERS - 1][i].value;
        error = output - expected_output[i];
        gradient_out = error * dx_tanhl(output);
        network->node[NUM_LAYERS - 1][i].gradient = gradient_out;
    }

    for (int i = NUM_LAYERS - 2; i > 0; --i) { //compute a gradient for each node
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            output = network->node[i][j].value;
            long double weighted_sum = 0;
            for (int k = 0; k < (i == NUM_LAYERS - 2 ? OUTPUT_LAYER_NODES : NODES_PER_LAYER); ++k) {
                weighted_sum += network->node[i + 1][k].gradient * network->node[i + 1][k].weights[j];
            }
            network->node[i][j].gradient = weighted_sum * dx_tanhl(output);
        }
    }

    for (int i = 1; i < NUM_LAYERS; ++i) {
        for (int j = 0; j < (i == NUM_LAYERS - 1 ? OUTPUT_LAYER_NODES : NODES_PER_LAYER); ++j) {
            for (int k = 0; k < (i == 1 ? INPUT_LAYER_NODES : NODES_PER_LAYER); ++k) {
                output = network->node[i - 1][k].value;
                network->node[i][j].weight_gradients[k] += network->node[i][j].gradient * output;
            }
        }
    }
    
}

void update_weights(struct Network* network){

    for (int i = 1; i < NUM_LAYERS; ++i) { //update weights for layers, with conditions for first/last layers
        for (int j = 0; j < (i == NUM_LAYERS - 1 ? OUTPUT_LAYER_NODES : NODES_PER_LAYER); ++j) {
            for (int k = 0; k < (i == 1 ? INPUT_LAYER_NODES : NODES_PER_LAYER); ++k) {
                network->node[i][j].weights[k] -= learning_rate * network->node[i][j].weight_gradients[k];
                network->node[i][j].weight_gradients[k] = 0;
            }
        }
    }
}

void predict(struct Network* network, long double input[INPUT_LAYER_NODES]){

    for (int i = 0; i < NUM_LAYERS; ++i) { //initialize all values to 0
        for (int j = 0; j < NODES_PER_LAYER; ++j) {
            network->node[i][j].value = 0;
        }
    }

    for (int i = 0; i < INPUT_LAYER_NODES; ++i) { //feed input to input layer nodes
        network->node[0][i].value = input[i];
    }

    for (int i = 0; i < NUM_LAYERS - 1; ++i) { //iterate over layers
        for (int j = 0; j < (i == 0 ? INPUT_LAYER_NODES : NODES_PER_LAYER); ++j) { //check whether iteration is at bottom or top of network
            for (int k = 0; k < (i == NUM_LAYERS - 2 ? OUTPUT_LAYER_NODES : NODES_PER_LAYER); ++k) {
                propagate_node(&network->node[i][j], &network->node[i + 1][k], j);
            }
        }
    }
}

long double mse(long double data[3][BATCH_SIZE][OUTPUT_LAYER_NODES]){
    long double error = 0;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_LAYER_NODES; ++j) {
            error += powl(data[PRED][i][j] - data[VALID][i][j], 2);
        }
    }
    error /= (BATCH_SIZE * OUTPUT_LAYER_NODES);
    return sqrtl(error);
}
