# Отчет по лабораторной работе 1

### Название: Asteroseismology of ~16000 Kepler red giants

### Ссылка: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/236/42#/browse

### Пояснения

В прошлой работе 0 были допущены некоторые ошибки. В разделе Pre-processing'а мной были исправлены недочеты. Первый из них - удаление значений звезд, у которых были пропущены значения параметров - 'A', 'e_A', 'Width', 'e_Width'. В этой же работе я заменял значения пустых параметров на среднее значения звезд 1 фазы, так как в 0 лабораторной были выделено, что звезды с пропущенными значенями были 1 фазы.

```
arr = ['A', 'e_A', 'Width', 'e_Width']
df_1 = df[df['Phase'] == 1]
    for i in arr:    
        df[i] = df[i].fillna(df[i].mean())
```
        
Второй недочет - нормализация значений, это было сделано для наилучшей работы классификаторов.
```
df=(df-df.mean())/df.std()
```
### Работа состоит: 

- Проектировка классификаторов

Линейная регрессия 
```
class my_LinearRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.iters = 1500
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0
        
        for i in range(self.iters):
            pred = np.dot(X, self.w) + self.b
            dw = 1/samples * np.dot(X.T, (pred - y))
            db = 1/samples * np.sum(pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def predict(self, X):
        pred = np.dot(X, self.w) + self.b
        y_pred = np.where(pred <=  1.5, 1, 2)
            
        return y_pred
 ```
 
 KNN
 ```
 class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_val=5):
        self.n_val = n_val
        self.train = None
        self.validation = None

    def fit(self, X, y):
        self.train = X
        self.validation = y
        
    def predict(self, X):
        prediction = []
        j = 1
        n = len(X)
        for y_test in X.to_numpy():
            distance = []
            for i in range(len(self.train)):
                distance.append(np.linalg.norm(np.array(self.train.iloc[i]) - y_test))
                
            distance_data = pd.DataFrame(data = distance, columns = ['dist'], index = self.validation.index)
            neighbours = distance_data.sort_values(by='dist', axis=0)[:self.n_val]
            
            labels = self.validation.loc[neighbours.index]
            vote = mode(labels).mode[0]
            
            prediction.append(vote)
            j+=1
            
        return prediction
 ```
 
 SVM
 ```
 class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=0.01, lamb=0.01, iters=1000):
        self.lr = lr
        self.lamb = lamb
        self.iters = iters
        self.w = None
        self.b = None
        self.n = 0
        self.m = 0
        
    def fit(self, X, y):
        self.n = y.max()
        self.m = y.min()
        y_ = np.where(y <= (self.n+self.m)/2, -1, 1)
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0
        
        for i in range(self.iters):
            for j, x_j in enumerate(X.to_numpy()):
                cond = y_[j] * (np.dot(x_j, self.w) - self.b) >= 1
                if cond:
                    self.w -= self.lr * (2 * self.lamb * self.w)
                else:
                    self.w -= self.lr * (2 * self.lamb * self.w - np.dot(x_j, y_[j]))
                    self.b -= self.lr * y_[j]
                    
    def predict(self, X):
        linear = np.sign(np.dot(X, self.w) - self.b)
        pred = np.where(linear <= 0, self.m, self.n)
        return pred
 ``` 
 
 Naive Bayes
 ```
 class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes = None 
        self.n = 0
        self.mean = None
        self.var = None
        self.pri = None
        
    def fit(self, X, y):
        samples, features = X.shape
        self.classes = np.unique(y)
        self.n = len(self.classes)
        self.mean = np.zeros((self.n, features), dtype=np.float64)
        self.var = np.zeros((self.n, features), dtype=np.float64)
        self.pri = np.zeros(self.n, dtype=np.float64)
        
        for j in self.classes:
            X_ = X[j==y]
            self.mean[j-1] = X_.mean(axis=0)
            self.var[j-1] = X_.var(axis=0)
            self.pri[j-1] = X_.shape[0]/float(self.n)
            
    def func(self, idx, x):
        mean = self.mean[idx]
        var = self.var[idx]
        num = np.exp(- (x-mean)**2 / (2*var))
        denum = np.sqrt(2 * np.pi * var)
        k = num / denum
                
        return k
            
    def _predict(self, x):
        post = []
    
        for k in range(len(self.classes)):
            pri = np.log(self.pri[k])
            class_cond = np.sum(np.log(self.func(k, x)))
            poster = pri + class_cond
            post.append(poster)
            
        return self.classes[np.argmax(post)]
    
    def predict(self, X):
        pred = [self._predict(x) for x in X.to_numpy()]
        return pred
 ``` 
- Составление Pipline'ов

Функция pre-proccesing'а
 ```
def trans(data):
    df = data.copy()
    arr = []
    for column in df.columns:
        if df.dtypes[column] == 'object':
            arr.append(column)
    for i in range(1, 20):    
        df[df == i*' '] = np.nan
    for i in arr:
        df[i] = pd.to_numeric(df[i])
    df_1 = df[df['Phase'] == 1]
    for i in arr:    
        df[i] = df[i].fillna(df[i].mean())
    
    par = ["NoCorM", "e_NoCorM", "NoCorR", "e_NoCorR",  "RGBcorM", "e_RGBcorM", "RGBcorR", "e_RGBcorR", "Phase"]
    df = df.drop(par, axis = 1)
    df=(df-df.mean())/df.std()
    
    return df
    
class SpInpTransformer():
    def __init__(self,f):
        self.func = f

    def transform(self, input_df, **transform_params):
        return self.func(input_df)
  
    def fit(self, X, y = None, **fit_params):
        return self
 ```
 
 Pipeline
  ```
 my_clf = Pipeline(
    [
        ("trans", SpInpTransformer(trans)),
        ("model", my_LinearRegression())
    ])
 ```

- GridSearchCV

На этом этапе были определены наилучшие параметры дляклассификаторов

- Pickle

Наилучшие классификаторы удалось сохраните в формате pickle. Их можно загрузить в репозитории

- Вывод

В данной части лабораторной работы мной были изучены и составлены 4 предложенных классификатора. Но мои классификаторы работают немного хуже, чем коробочные, а также дольше из-за оптимизации. С помощью их удалось добиться хороших результатов. Все результаты больше 80. Также был изучен pipeline, который позволяет создавать последовательные стадии использования классификатора (pre-processing, classification ...). Во время выполнения работы также использовал метод GridSearchCV, для определения лучших параметров и использовании кросс-валидации. В результате чего, удалось добиться еще более лучших результатов даже 95 %. Наилушие классификаторы были сохранены в файл формата pickle. 
