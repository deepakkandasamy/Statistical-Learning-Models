from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
class AdaBoost:

	def __init__(self, training_set):
		self.training_set = training_set
		self.N = len(self.training_set)
		self.weights = np.ones(self.N)/self.N
		self.RULES = []
		self.ALPHA = []
		
	def fit(self,rounds=50):
		for i in xrange(rounds):
			min_error=1;
			min_index=-1;
			polarity=-1;
			line=0;
			for j in xrange(len(self.training_set[0][0])):
				l = [x[0][j] for x in (self.training_set)]
				l.sort()
				for k in xrange(len(l)-1):
					bound=l[k]+(l[k+1]-l[k])/2
					#print k,bound
					errors_left=((np.array([t[1]!=(2*(t[0][j]>=bound)-1) for t in self.training_set]))*self.weights).sum();
					if(min_error>errors_left):
						rule=lambda x: 2*(x[j]>=bound)-1
						min_index=j
						polarity=0
						line=bound
						min_error=errors_left
					errors_right=((np.array([t[1]!=(2*(t[0][j]<bound)-1) for t in self.training_set]))*self.weights).sum();
					if(min_error>errors_right):
						rule=lambda x: 2*(x[j]<bound)-1
						min_index=j
						polarity=1
						line=bound
						min_error=errors_right
			#print min_index,polarity,line
			self.set_rule(min_index,polarity,line)	
					
					
				
				
	def func(self,x,min_index,polarity,bound):
		if(polarity):
			return 2*(x[min_index]<bound)-1
		else:
			return 2*(x[min_index]>=bound)-1

	def set_rule(self, min_index,polarity,bound,test=False):
		errors = np.array([t[1]!=self.func(t[0],min_index,polarity,bound) for t in self.training_set])
		e = (errors*self.weights).sum()
		print errors
		if test: return e
		alpha = 0.5*np.log((1-e)/e)
		print 'e=%.2f a=%.2f'%(e, alpha)
		w = np.zeros(self.N)
		for i in range(self.N):
			if errors[i] == 1: w[i] = self.weights[i] * np.exp(alpha)
			else: w[i] = self.weights[i] * np.exp(-alpha)
		self.weights = w / w.sum()
		self.RULES.append((min_index,polarity,bound))
		self.ALPHA.append(alpha)
	#def fit(self,rounds=10):
		

	def evaluate(self):
		NR = len(self.RULES)
		for (x,l) in self.training_set:
			hx = [self.ALPHA[i]*self.func(x,self.RULES[i][0],self.RULES[i][1],self.RULES[i][2]) for i in range(NR)]
			print x, np.sign(l) == np.sign(sum(hx))

if __name__ == '__main__':
	examples = []
	examples.append(((1,  2  ), 1))
	examples.append(((1,  4  ), 1))
	examples.append(((2.5,5.5), 1))
	examples.append(((3.5,6.5), 1))
	examples.append(((4,  5.4), 1))
	examples.append(((2,  1  ),-1))
	examples.append(((2,  4  ),-1))
	examples.append(((3.5,3.5),-1))
	examples.append(((5,  2  ),-1))
	examples.append(((5,  5.5),-1))
	
	m = AdaBoost(examples)
	
	m.fit(rounds=3)
	'''
	m.set_rule(lambda x: 2*(x[0] < 1.5)-1)
	m.set_rule(lambda x: 2*(x[0] < 4.5)-1)
	m.set_rule(lambda x: 2*(x[1] > 5)-1)
	'''
	#m.fit(rounds=5)
	m.evaluate()
	for i in range(len(examples)):
		if(examples[i][1]==1):
			plt.plot(examples[i][0][0],examples[i][0][1],'ro')
		elif(examples[i][1]==-1):
			plt.plot(examples[i][0][0],examples[i][0][1],'go')

	plt.show()

