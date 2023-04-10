#include "Nodes.h"

struct Network network;

extern long double learning_rate;

int main() {
    srand(time(NULL)); //initialize basics such as seed rand(), create train data array
    long double data[3][BATCH_SIZE][INPUT_LAYER_NODES];
    long double error;

    init_train_data(data);
    initialize_network(&network);
    learning_rate = 0.1;

    for (int i = 0; i < EPOCHS; ++i) { //num of times network is trained
        for (int j = 0; j < BATCH_SIZE; ++j) {
            predict(&network, data[INPUT][j], data[PRED][j]);
            if(OUTPUT_LAYER_NODES > 1){
                error = 0;
                for (int k = 0; k < OUTPUT_LAYER_NODES; ++k) { //getting mse if more than one output node
                    error += data[PRED][j][k] - data[VALID][j][k];
                }
                error /= OUTPUT_LAYER_NODES;
            }
            else{
                error = data[PRED][j][0] - data[VALID][j][0];
            }
            compute_gradients(&network, error);
            update_weights(&network);

        }
        printf("MSE for this Epoch: %Lf\n", mse(data));
        //init_train_data(data);
    }
    while(1){
        long double input;
        long double output;
        printf("Enter the input value to predict\n");
        scanf("%Lf", &input);
        predict(&network, &input, &output);
        printf("Input: %Lf, Prediction: %Lf\n", input, output);
    }
}