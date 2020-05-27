FLAGS = \
    -Ic/ \
    -lm \
    -O3 \
    -std=c99 \
    -Wall -Werror -Wpedantic -Weverything -Wextra \
    -Wno-missing-prototypes -Wno-conversion -Wno-sign-conversion -Wno-padded \
    -Wno-cast-align -Wno-switch-enum -Wno-double-promotion \
    -Wno-covered-switch-default -Wno-unused-macros -Wno-unused-parameter

all: clean
	mkdir -p bin/
	clang c/test.c c/bf/*.c -o bin/test $(FLAGS)

clean:
	rm -rf bin/
