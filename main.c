#include "Nodes.h"


struct Network network;

int main() {
    srand(time(NULL));

    initialize_network(&network);

    long double pred_valid[3][BATCH_SIZE];

    time_t start = clock();

    for (int i = 0; i < BATCH_SIZE; ++i) {//predicting sin(x) function

        pred_valid[INPUT][i] = rand()%101;
        pred_valid[VALID][i] = sinl(pred_valid[INPUT][i]);
        pred_valid[PRED][i] = predict(&network, pred_valid[INPUT][i]);
    }
    start = clock() - start;

    printf("Runtime: %lldms, mse: %Lf", start, mse(pred_valid));

    return 0;
}

