import pandas as pd

output = pd.read_csv('./data/output.csv')
status = []
for str in output['status_group']:
	str = str.strip('b')
	str = str.strip('\'')
	status.append(str)

pred=pd.DataFrame({'id': output["id"], "status_group": status})
pred.to_csv("test_label.csv",index=False)
