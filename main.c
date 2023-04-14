#include "Nodes.h"


extern long double learning_rate;

struct Network network;


int main() {
    srand(time(NULL)); //initialize basics such as seed rand(), create train data array
    long double data[3][BATCH_SIZE][INPUT_LAYER_NODES];


    init_train_data(data);
    initialize_network(&network);
    learning_rate = 0.006;


    for (int i = 0; i < EPOCHS; ++i) { //num of times network is trained
        for (int j = 0; j < BATCH_SIZE; ++j) {
            predict(&network, data[INPUT][j]);
            for (int k = 0; k <OUTPUT_LAYER_NODES; ++k) {
                data[PRED][j][k] = network.node[NUM_LAYERS - 1][k].value; //save predictions
            }
            compute_gradients(&network, data[VALID][j]);
            update_weights(&network);
        }
        printf("MSE for Epoch %d: %.20Lf\n", i, mse(data));
        if(i%10 == 0)
            init_train_data(data);
    }

    long double input;

    while(1){
        printf("Enter the input value to predict\n");
        scanf("%Lf", &input);
        input = fmodl(input, 1);
        predict(&network, &input);
        printf("Input: %Lf, Prediction: %.20Lf\n", input, network.node[NUM_LAYERS - 1][0].value);
    }
}