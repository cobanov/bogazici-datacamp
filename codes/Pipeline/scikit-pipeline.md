# Sci-kit Learn Pipeline Rehberi

Pipeline kavramı projelerinizde kullanabileceğiniz en verimli ve kullanışlı kavramlardır. Genellikle bir proje bitirdiğinizde ve canlıya almak istediğinizde kullanabilirsiniz, tabi ayrıca tüm prosesi pipeline'a alarak,  farklı zamanlarda aynı tür veriye preprocess kısmında yapabileceğiniz hataları da ortadan kaldırmış olursunuz. Ayrıca GridSearch yaparken pipeline'ı kullanmanız aşırı derecede önerilir.

Yapılan iş en genel tabirle işlemlerinizi bir zincir gibi çalıştırmaktır. Özellik seçimi, normalleştirme ve sınıflandırma gibi verilerin işlenmesinde genellikle sabit bir dizi adım olduğu için bu yararlıdır. 

### Pipeline Nasıl Oluşturulur

Genel olarak pipeline oluşturmak için 2 yöntem vardır. Bunlardan birincisi `Pipeline` ve ikincisi `make_pipeline` görece olarak ikinci seçeneği seçmek daha basit ve hızlı bir yöntemdir. İsimlendirmeler otomatik olarak yapılır


```python
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

pipe = Pipeline([('s_scaler', StandardScaler()), ('clf', DecisionTreeClassifier())])     # 1. Seçenek
pipe2 = make_pipeline(StandardScaler(), DecisionTreeClassifier())                        # 2. Seçenecek
```

### Basit layout


```python
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pipe_example  = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
print(pipe_example)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe_example.fit(X_train, y_train)

print(f" \nAccuracy : {accuracy_score(pipe.predict(X_test), y_test)}")
```

    Pipeline(memory=None,
             steps=[('standardscaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('decisiontreeclassifier',
                     DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                            criterion='gini', max_depth=None,
                                            max_features=None, max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            presort='deprecated', random_state=0,
                                            splitter='best'))],
             verbose=False)
     
    Accuracy : 0.9736842105263158
    

### Parametreler

`DecisionTreeClassifier`'ın `min_sample_split` argümanını, pipe içerisinde verdiğimiz ismin yanına `__` ve parametrenin ismini girerek, bir sözlük olarak verebiliriz.

_GridSearchCV'ye estimator değil pipe'ı verdiğimize dikkat edin!_


```python
from sklearn.model_selection import GridSearchCV

param_grid = dict(decisiontreeclassifier__min_samples_split=[2, 5, 10])
grid_search = GridSearchCV(pipe_example, param_grid=param_grid)
```


```python
grid_search.fit(X_train, y_train)
```




    GridSearchCV(cv=None, error_score=nan,
                 estimator=Pipeline(memory=None,
                                    steps=[('standardscaler',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('decisiontreeclassifier',
                                            DecisionTreeClassifier(ccp_alpha=0.0,
                                                                   class_weight=None,
                                                                   criterion='gini',
                                                                   max_depth=None,
                                                                   max_features=None,
                                                                   max_leaf_nodes=None,
                                                                   min_impurity_decrease=0.0,
                                                                   min_impurity_split=None,
                                                                   min_samples_leaf=1,
                                                                   min_samples_split=2,
                                                                   min_weight_fraction_leaf=0.0,
                                                                   presort='deprecated',
                                                                   random_state=0,
                                                                   splitter='best'))],
                                    verbose=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'decisiontreeclassifier__min_samples_split': [2, 5,
                                                                           10]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
grid_search.best_params_
```




    {'decisiontreeclassifier__min_samples_split': 5}




```python
preds_cv = grid_search.predict(X_test)
print(f" \nAccuracy : {accuracy_score(preds_cv, y_test)}")

```

     
    Accuracy : 0.9736842105263158
    
