import numpy as np 

class Brain:
	def __init__self(self, env, npcs, sigma):
		"""
		Args:
			env: ABCEnvironment class instance
			npcs: Number of place cells
			sigma: Array-like or scalar. Tuning curve of place cells
		"""
		self.pcs = env.sample_uniform(npcs) # sample place cell
		self.npcs = npcs
		self.sigma = sigma

	def d_pcc(self, pos, pc):
		"""
		(Shortest?) distance to a place cell center
		"""
		ni, wall = env.crash_point(pos, pc - pos)
		atcf = euclidean(pos,pc) # as the crows fly
		if atcf <= euclidean(pos,ni):
			# no obstruction
			return atcf

		# else
		tmp_dist = 0
		tmp_dist += d_pcc(new_pos, pc) # recursive
		return tmp_dist

	@abstractmethod
	def norm_reponse(distance):
		"""
		Returns activity where place cell tuning curves are 
		modelled as a gaussian func
		"""
		return activity

	def laplace_response(self,distance):
		"""
		Returns activity where place cell tuning curves are 
		modelled as a difference of gaussian func
		"""
		return activity