"""
This function is not relevant for this repository,
but is a nice method for easily timing functions 
by simply decorating the method that we want to time.

Author:
	Vemund S. Schøyen
	vemund@live.com
	03.05.2021
"""
from timeit import default_timer as timer
from functools import wraps
import inspect


def ctimeit(function):
    """
    A custom decorator for timing a function

    Usage:
            Simply add: @ctimeit before a function definition

    Example:
            @ctimeit
            def g(x):
                    return x
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        # time function
        start = timer()
        output = function(*args, **kwargs)
        end = timer()

        # adds function name and argument info to print
        info = ""
        argspec = inspect.getargspec(function)
        arg_strs = argspec[0]
        for i in range(len(args)):
            try:
                info += arg_strs.pop(0) + "={}, ".format(args[i])
            except IndexError:
                info += "arg[{}]".format(i) + "={}, ".format(args[i])

        for i, arg_str in enumerate(arg_strs):
            info += arg_str + "={}, ".format(argspec[-1][i])

        for arg_str in kwargs.keys():
            info += arg_str + "={}, ".format(kwargs[arg_str])

        info = "{}(" + info[:-2] + ") used <{}> seconds"
        info = info.format(function.__name__, (end - start))
        print(info)

        return output

    return wrapper
