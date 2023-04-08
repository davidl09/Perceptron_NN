#include "Nodes.h"


struct Network network;

int main() {
    srand(time(NULL));

    initialize_network(&network);

    long double pred_valid[3][BATCH_SIZE];
    long double error;

    init_train_data(pred_valid);

    time_t start = clock();

        for (int i = 0; i < BATCH_SIZE; ++i) {//predicting sin(x) function, initialize train data array
            pred_valid[PRED][i] = predict(&network, pred_valid[INPUT][i]);
            error = pred_valid[PRED][i] - pred_valid[VALID][i];
            compute_gradients(&network, error);
            update_weights(&network);
            init_train_data(pred_valid);
            printf("Output node gradient: %.17Lf, Prediction: %Lf, Error: %Lf\n", network.node_out.gradient, pred_valid[PRED][i], error);
        }
        printf("MSE for this run: %Lf\n", mse(pred_valid));




    start = clock() - start;

    printf("Runtime: %lldms, mse: %Lf", start, mse(pred_valid));

    return 0;
}

