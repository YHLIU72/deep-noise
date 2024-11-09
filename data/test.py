import noisedata

NoiseData = noisedata.NoiseData(dir='../../data')
print(NoiseData.__len__())
print(NoiseData.__getitem__(1888))

# NoiseData = noisedata.NoiseData(dir='../../data',use_type=True)
# print(NoiseData.le)
# print(NoiseData.__len__())
# print(NoiseData.__getitem__(1))