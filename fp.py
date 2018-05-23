from pymonad.Reader import curry
import functools
def pipe(*functions):
    def pipe2(f, g):
        return lambda x: g(f(x))
    return functools.reduce(pipe2, functions, lambda x: x)

cmap = curry(lambda f,xs: map(f,xs))
cfilter = curry(lambda f,xs: filter(f,xs))
