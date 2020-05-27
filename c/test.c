#include <stdio.h>

#include "bf/bf.h"

void go() {
    printf("rand:\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2 * 3; ++j) {
            printf(" %8.5f", prng_rand(-1, 2));
        }
        printf("\n");
    }
    printf("\n");
    printf("randint:\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2 * 3; ++j) {
            printf(" %2d", prng_randint(-1, 2));
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    go();
    prng_set_seed(42);
    go();
}
