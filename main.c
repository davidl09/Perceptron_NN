#include "Nodes.h"


extern long double learning_rate;

struct Network network;

FILE* fp;

int main() {
    srand(time(NULL)); //initialize basics such as seed rand(), create train data array
    //long double data[3][BATCH_SIZE][INPUT_LAYER_NODES];
    long double input;
    long double prediction;
    long double validation;
    long double error;


    //init_train_data(data);
    initialize_network(&network);
    learning_rate = 0.01;

    for (int j = 0; j < EPOCHS; ++j) {
        error = 0;
        for (int i = -BATCH_SIZE/2; i < BATCH_SIZE/2; ++i) {
            input = 2 * (long double)i/BATCH_SIZE;
            predict(&network, &input);
            validation = model_func(input);
            error += powl(network.node[NUM_LAYERS - 1][0].value - validation, 2);
            //printf("Input: %Lf, Prediction: %Lf, Validation: %Lf\n", input, network.node[NUM_LAYERS - 1][0].value, validation);
            compute_gradients(&network, &validation);
            update_weights(&network);
        }
        for (int i = BATCH_SIZE/2; i > -BATCH_SIZE/2; --i) {
            input = 2 * (long double)i/BATCH_SIZE;
            predict(&network, &input);
            validation = model_func(input);
            error += powl(network.node[NUM_LAYERS - 1][0].value - validation, 2);
            //printf("Input: %Lf, Prediction: %Lf, Validation: %Lf\n", input, network.node[NUM_LAYERS - 1][0].value, validation);
            compute_gradients(&network, &validation);
            update_weights(&network);
        }
        printf("Epoch: %d, Error: %.17Lf, learning rate: %.10Lf\n", j + 1, sqrtl(error/(2 * BATCH_SIZE)), learning_rate);
        if(learning_rate > 1e-7)
            learning_rate *= 0.9998;
    }

    /*
    for (int i = 0; i < EPOCHS; ++i) { //num of times network is trained
        for (int j = 0; j < BATCH_SIZE; ++j) {
            predict(&network, data[INPUT][j]);
            for (int k = 0; k <OUTPUT_LAYER_NODES; ++k) {
                data[PRED][j][k] = network.node[NUM_LAYERS - 1][k].value; //save predictions
            }
            compute_gradients(&network, data[VALID][j]);
            update_weights(&network);
        }
        printf("MSE for Epoch %d: %.20Lf\n", i+1, mse(data));
        if(i%10 == 0) {
            init_train_data(data);
            continue;
        }
    }
    */

    char filename[] = "perceptron_res_%depochs";

    fp = fopen("C:\\Users\\dalae\\Desktop\\sin_.csv", "w");
    for (int i = -1000; i < 1000; ++i) {
        input = i/1000.0;
        predict(&network, &input);
        fprintf(fp, "%.15Lf,%.15Lf\n", input, network.node[NUM_LAYERS - 1][0].value);
    }
    fprintf(fp, "eof");

    system("python C:\\Users\\dalae\\Desktop\\visualize_output.py");


    while(1){
        printf("Enter the input value to predict\n");
        scanf("%Lf", &input);
        while(absl(input) > 1){
            if(input > 0)
                input -= 1;
            else
                input += 1;
        }
        predict(&network, &input);
        prediction = network.node[NUM_LAYERS - 1][0].value;
        validation = model_func(input);
        printf("Prediction: %.20Lf, Validation: %Lf, Error: %Lf\n", prediction, validation, prediction - validation);
    }
}