#include "Nodes.h"


struct Network network;

extern long double learning_rate;



int main() {
    srand(time(NULL));

    initialize_network(&network);

    long double train_pred_valid[3][BATCH_SIZE];
    long double error;

    float scale_fact;

    //print_weights(&network);

    init_train_data(train_pred_valid);
    scale_fact = scale_inputs(train_pred_valid);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        train_pred_valid[INPUT][i] /= scale_fact;
    }

    time_t start = clock();

    for(int k = 0; k < EPOCHS; ++k){
        for (int i = 0; i < BATCH_NUM; ++i) {
            for (int j = 0; j < BATCH_SIZE; ++j) {
                learning_rate = 0.001;
                //learning_rate = 0.001 - 0.0001 * (k/12);
                train_pred_valid[PRED][j] = predict(&network, train_pred_valid[INPUT][j]);
                error = train_pred_valid[PRED][j] - train_pred_valid[VALID][j];
                compute_gradients(&network, error);
                update_weights(&network);
            }
            //printf("MSE for this run: %Lf\n", mse(train_pred_valid));
        }
        printf("MSE for this run: %Lf\n", mse(train_pred_valid));
        //init_train_data(train_pred_valid);
        //scale_fact = scale_inputs(train_pred_valid);
        for (int i = 0; i < BATCH_SIZE; ++i) {
            //train_pred_valid[INPUT][i] /= scale_fact;
        }
    }


    start = clock() - start;

    //print_weights(&network);
    printf("Runtime: %lldms\n", start);

    long double input;
    while (1){
        printf("Enter a value to predict:");
        scanf("%Lf", &input);
        printf("Prediction: %.10Lf\n", predict(&network, input));
    }

    return 0;
}

