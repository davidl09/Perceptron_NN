#include "Nodes.h"


int main() {

    struct Network network;


    srand(time(NULL));


    initialize_network(&network);

    print_weights(&network);

    long double num = predict(&network, 2*M_PI);
    printf("%Lf", num);

    return 0;
}

