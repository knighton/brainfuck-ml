#ifndef BF_PRNG_H_
#define BF_PRNG_H_

#include <stdint.h>

#define BF_RAND_MAX (1u << 31)

typedef struct xorshift32_t {
    uint32_t a;
} xorshift32_t;

void xorshift32_free(xorshift32_t* state);
void xorshift32_set_seed(xorshift32_t* state, uint32_t seed);
uint32_t xorshift32_get(xorshift32_t* state);

void prng_free(void);
void prng_set_seed(int seed);
float prng_rand(float low, float high);
int prng_randint(int low, int high);

#endif  /* BF_PRNG_H_ */
