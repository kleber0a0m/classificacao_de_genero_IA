## Classificação de gênero usando Inteligência Artificial

Essa aplicação é capaz de informar seu gênero (Masculino ou Feminino) a partir de 7 perguntas, são elas:
 - Cabelo longo?
 - Largura da testa(cm)
 - Altura da testa(cm)
 - Nariz longo?
 - Nariz comprido?
 - Lábios finos?
 - Distância entre nariz e lábios?

### Teste você mesmo:
[Clique aqui](http://kleberalbinomoreira.com.br/classificacao_de_genero_IA.html "http://kleberalbinomoreira.com.br/classificacao_de_genero_IA.html")
### Processo de desenvolvimento:
Essa aplicação utiliza aprendizagem de máquina, e o modelo foi treinado utilizando um Dataset de domino púlico disponivel no  [kaggle](https://www.kaggle.com/datasets/elakiricoder/gender-classification-dataset "kaggle").

Segue um trecho do Dataset:

|long_hair|forehead_width_cm|forehead_height_cm|nose_wide|nose_long|lips_thin|distance_nose_to_lip_long|gender|
|--|--|--|--|--|--|--|
|1|11.8|6.1|1|0|1|1|Male|
|0|14|5.4|0|0|1|0|Female|
|0|11.8|6.3|1|1|1|1|Male|
|0|14.4|6.1|0|1|1|1|Male|
|1|13.5|5.9|0|0|0|0|Female|
|1|13|6.8|1|1|1|1|Male|

Para o modelo de treinamendo da inteligencia artificial foi utilizado a biblioteca de codigo aberto [scikit-learn](https://scikit-learn.org/ "scikit-learn")
Segue o codigo ultilizado para criação do modelo pkl utilizando o Dataset:
```python
from sklearn import tree,datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
url = "gender_classification_v7.csv"
data = pd.read_csv(url)

print('---------------data.info()-------------------')
data.info()
print('---------------data.head-------------------')
data.head
x = data.drop("gender",axis=1) #features
y = data['gender'] #labels
print('---------------print(x)-------------------')
print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

predictions=clf.predict(x_test)
print('----------------------------------')
print("Preditos: ",predictions[:20])
print("Real:     ",y_test[:20].values)

print("Acurácia: ",accuracy_score(y_test,predictions))

print(clf.predict([[1,1,1,1,1,1,1]]))
print(clf.predict([[9,9,9,9,9,9,9]]))
print(clf.predict([[3,3,3,3,3,3,3]]))
print(clf.predict([[900,900,900,900,900,900,900]]))
a = int(input('long_hair: '))
b = int(input('forehead_width_cm: '))
c = int(input('forehead_height_cm: '))
d = int(input('nose_wide: '))
e = int(input('nose_long: '))
f = int(input('lips_thin: '))
g = int(input('distance_nose_to_lip_long: '))
pred = clf.predict([[a,b,c,d,e,f,g]])
print(pred)
#print(lookup_iris_name[pred[0]])

import pickle
pickle.dump(clf, open('model.pkl','wb'))


```
Parte do Dataset foi utilizada para o treinamento da IA e parte foi usada para testar so modelo, foi obtido 95.46% de acurácia.

O model.pkl exportado anteriomente foi colocado em produção utilizando [flask](https://flask.palletsprojects.com/ "flask") e está hospedado gratuitamento no [heroku](https://www.heroku.com/ "heroku")




