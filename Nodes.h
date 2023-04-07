//
// Created by dalae on 2023-04-06.
//

#ifndef CNN_1_NODES_H

#define NUM_LAYERS 5
#define relu(num) num > 0 ? num : 0

struct Node;
void propagate_main(struct Node* node, double value);

#define CNN_1_NODES_H

#endif //CNN_1_NODES_H
