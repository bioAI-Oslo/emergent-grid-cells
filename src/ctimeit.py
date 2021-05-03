from timeit import default_timer as timer
from functools import wraps

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
	def wrapper(*args,**kwargs):
		start = timer()
		output = function(*args,**kwargs)
		end = timer()
		print(*args[1:])
		print("Call to function <{}> took = {} s".format(
			function.__name__, (end - start)))
		return output

	return wrapper