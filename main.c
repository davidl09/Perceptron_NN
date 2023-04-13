#include "Nodes.h"


extern long double learning_rate;

struct Network network;

FILE* fp;

int main() {
    srand(time(NULL)); //initialize basics such as seed rand(), create train data array
    long double data[3][BATCH_SIZE][INPUT_LAYER_NODES];


    init_train_data(data);
    initialize_network(&network);
    learning_rate = 0.0005;


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
        if(i%10 == 0) {
            init_train_data(data);
            continue;
        }
    }

    long double input;
    long double prediction;
    long double validation;
    fp = fopen("sin.txt", "w");
    fprintf(fp, "input,prediction");
    for (int i = (int)(- 100.0 * M_PI); i < (int)(100 * M_PI); ++i) {
        input = i/100.0;
        predict(&network, &input);
        fprintf(fp, "%.15Lf,%.15Lf\n", input, network.node[NUM_LAYERS - 1][0].value);
    }
    fprintf(fp, "eof");

    while(1){
        printf("Enter the input value to predict\n");
        scanf("%Lf", &input);
        while(absl(input) > M_PI){
            if(input > 0)
                input -= M_PI;
            else
                input += M_PI;
        }
        predict(&network, &input);
        prediction = network.node[NUM_LAYERS - 1][0].value;
        validation = model_func(input);
        printf("Prediction: %.20Lf, Validation: %Lf, Error: %Lf\n", prediction, validation, prediction - validation);
    }
}