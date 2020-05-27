from .common.bf.prng import set_seed, rand, randint


def go_f(f):
    print('%s:' % f.__name__)
    for i in range(4):
        x = f(-1, 2, (2, 3))
        print(x.flatten())
    print()


def go():
    go_f(rand)
    go_f(randint)


def main():
    go()
    set_seed(42)
    go()


if __name__ == '__main__':
    main()
