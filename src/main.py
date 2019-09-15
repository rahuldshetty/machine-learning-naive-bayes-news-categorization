from nb import *
from nlp import *
import pandas as pd 

df = pd.read_csv('bbc-text.csv')

x = list(df['text'])
y = list(df['category'])

x = process_doc(x)

LIMIT = 95
size = int(len(x) * LIMIT / 100 ) 
x_train = x[:size]
y_train = y[:size]
x_test = x[size:]
y_test = y[size:]

# training
nb = NB()
nb.fit(x_train,y_train)
nb.save()

# finding accuracy
res = nb.get_pred_classes(x_test)
count_correct = 0
for i in range(len(res)):
    output = res[i]
    act = y_test[i]
    if output == act:
        count_correct +=1 

acc = 100 * (count_correct/len(res))

print("Accuracy:",acc)



# random sample to run
samples = ['a very close game','bollywood shines light on world','economy shrinks down in german','microsoft launches windows 10 for new computer system','batsman failed in cricket','people elected barack as the new president']
pre_samples = [preprocess(x) for x in samples]
res = nb.get_pred_classes(pre_samples)
for i in range(len(samples)):
    print("-------")
    print("Text:",samples[i])
    print("Predicted:",res[i])