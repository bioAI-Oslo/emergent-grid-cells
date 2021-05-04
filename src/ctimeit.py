from timeit import default_timer as timer
from functools import wraps
from inspect import getframeinfo, stack, getargspec


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
        argspec = getargspec(function)
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

        caller = getframeinfo(stack()[1][0])
        info = "File \"{}\", line {}, function: {}(" + info[:-2] + ") used <{}> seconds"
        info = info.format(caller.filename, caller.lineno, function.__name__, (end - start))
        print(info)

        return output

    return wrapper
