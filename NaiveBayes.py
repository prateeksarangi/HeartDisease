import numpy as np
import pandas as pd

df = pd.read_csv('heart.csv')

columns = df.columns.values
attrs = columns[0:len(columns) - 1]

pdict_final = {}
acc_final = 0

target_col = columns[-1]
classes = df[target_col].unique()

print('\nAttributes:\n')
print(attrs)

print('\nOutput Classes:\n')
print(classes)

for i in range(0, 100):
	xdf = df.sample(frac=1).reset_index(drop=True)

	vdf = xdf.tail((int)(len(xdf) * 0.3))

	tdf = xdf.head((int)(len(xdf) * 0.7))

	# print("\nTraining Data:\n")
	# print(df.head(len(df)))

	pdict = {}

	for oclass in classes:
		pdict[oclass] = {}
		pdict[oclass]['self'] = len(tdf[tdf[target_col].isin([oclass])]) / len(tdf[target_col])
		for attr in attrs:
			pdict[oclass][attr]={}
			attr_vals = tdf[tdf[target_col].isin([oclass])][attr].values
			pdict[oclass][attr]['mean'] = attr_vals.mean()
			pdict[oclass][attr]['std'] = attr_vals.std()
			pdict[oclass][attr]['var'] = attr_vals.var()

	# print("\nData Influence Dictionary:\n")
	# print(pdict)

	pred = []

	for index, row in vdf.iterrows():
		pout = {}
		for oclass in classes:
			p = pdict[oclass]['self']
			for attr in attrs:
				attr_val = row[attr]
				v1 = 1 / (np.sqrt(2 * np.pi) * pdict[oclass][attr]['std'])
				v2 = np.exp(-(((attr_val - pdict[oclass][attr]['mean']) ** 2) / (2 * pdict[oclass][attr]['var'])))
				p *= v1 * v2
			pout[oclass] = p
		pred.append(max(pout, key=pout.get))

	vdf['Prediction'] = pred

	# print("\nTest Data with Predictions:\n")
	# print(vdf.head(len(vdf)))

	accuracy = len(vdf[vdf['Prediction'] == vdf[target_col]]) / len(vdf)

	print('Iteration:', i, ', Accuracy:', (accuracy * 100), '%')

	if accuracy > acc_final:
		acc_final = accuracy
		pdict_final = pdict

print('\nBest Accuracy:', (acc_final * 100), '%')
print("\nBest Data Influence Dictionary:\n")
print(pdict)

# tdf = pd.read_csv('predict_c.csv', keep_default_na=False)

# print("\nTest Data:\n")
# print(tdf.head(len(tdf)))

# for index, row in tdf.iterrows():
# 	pout = {}
# 	for oclass in classes:
# 		p = pdict[oclass]['self']
# 		for attr in attrs:
# 			attr_val = row[attr]
# 			v1 = 1 / (np.sqrt(2 * np.pi) * pdict[oclass][attr]['std'])
# 			v2 = np.exp(-(((attr_val - pdict[oclass][attr]['mean']) ** 2) / (2 * pdict[oclass][attr]['var'])))
# 			p *= v1 * v2
# 		pout[oclass] = p
# 	tdf.at[index, target_col] = max(pout, key=pout.get)

# print("\nTest Data with Predictions:\n")
# print(tdf.head(len(tdf)))