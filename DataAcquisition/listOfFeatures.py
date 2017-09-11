class HECFeatures:
	def __init__(self):
		self.feature_names = []
		print('Collecting list of features.')
		with open('featuresUsed.txt') as fp:
			for line in fp:
				self.feature_names.append(line[:-1])
	
	def returnFeatureNames(self):
		return self.feature_names

