#include "Nodes.h"


struct Network network;

int main() {
    srand(time(NULL));

    initialize_network(&network);

    long double train_pred_valid[BATCH_NUM][3][BATCH_SIZE];
    long double error;



    time_t start = clock();

    for (int i = 0; i < BATCH_NUM; ++i) {
        init_train_data(train_pred_valid[i]);
    }

    print_weights(&network);

    for(int k = 0; k < EPOCHS; ++k){
        for (int i = 0; i < BATCH_NUM; ++i) {
            for (int j = 0; j < BATCH_SIZE; ++j) {
                train_pred_valid[i][PRED][j] = predict(&network, train_pred_valid[i][INPUT][j]);
                error = train_pred_valid[i][PRED][j] - train_pred_valid[i][VALID][j];
                compute_gradients(&network, error);
                update_weights(&network);
                //printf("%.5Lf\n", error);
            }
            printf("MSE for this run: %Lf\n", mse(train_pred_valid[i]));
        }
    }


    start = clock() - start;

    print_weights(&network);
    printf("Runtime: %lldms\n", start);

    long double input;
    while (1){
        printf("Enter a value to predict:");
        scanf("%Lf", &input);
        printf("Prediction: %.10Lf\n", predict(&network, input));
    }

    return 0;
}

