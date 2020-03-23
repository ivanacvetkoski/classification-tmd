import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB, BernoulliNB
from sklearn.metrics import classification_report
import sklearn.preprocessing as prep
import numpy as np
import sklearn.metrics as met
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Ucitavamo podatke i skracujemo imena atributima
df=pd.read_csv('dataset.csv')
df.columns = df.columns.str.replace('android.sensor.','').str.replace('#','_')

print('Informacije o bazi')
print(df.info())
print('Prvih 5 istanci')
print(df.head())

df=df.dropna()
df=df.drop_duplicates()

#Resavamo se outliers-a
df_x=df.drop(df.columns[-1],axis=1)
df_y=df[df.columns[-1]]
Q1 = df_x.quantile(0.25)
Q3 = df_x.quantile(0.75)
IQR = Q3 - Q1
v=1.5

df_no_out_x=df_x[~((df_x < (Q1 - v*IQR)) | (df_x > (Q3 + v*IQR))).any(axis=1)]
df_no_out_y=df_y[~((df_x < (Q1 - v*IQR)) | (df_x > (Q3 + v*IQR))).any(axis=1)]

new_idxs=pd.RangeIndex(len(df_no_out_y.index))
df_no_out_x.index=new_idxs
df_no_out_y.index=new_idxs

df_no_out=pd.concat([df_no_out_x,df_no_out_y], axis=1)

#Preko korelacije smanjujemo dimenziju baze
#ovaj deo je preuzet sa interneta

cormatrix=df.corr()
#fig, ax = plt.subplots(figsize=(16, 8))
#sns.heatmap(cormatrix, annot=True ,square=True)
#plt.show()

#Atributi sa korelacijom manjom od p
p = 0.5
var = []
for i in cormatrix.columns:
    for j in cormatrix.columns:
        if(i!=j):
            if np.abs(cormatrix[i][j]) > p:
                var.append([i,j])

upper = cormatrix.where(np.triu(np.ones(cormatrix.shape), k=1).astype(np.bool))

#Nadji indekse sa korelacijom vecom od c
c=0.5
to_drop = [column for column in upper.columns if any(abs(upper[column]) > c)]
#Izbaci atribute
data_less_cor=df.drop(to_drop, axis=1)

cormatrix_less_cor=data_less_cor.corr()
fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(cormatrix_less_cor, annot=True ,square=True)
plt.show()
#do ovog dela je preuzeto

#Poredicemo preciznost razlicitih algoritama nad podacima sa outliers-ima koje nismo ni normalizovani ni standardizovali,
# sa outliers-ima koje jesmo normalizovani i standardizovali,
# bez outliers-a koje jesmo normalizovani i standardizovali,
# sa smanjenim brojem atributa sa outliers-ima i bez njih

features=df.columns[:37].tolist()
x_without_prep=df[features]
y_without_prep=df['target']

x=df[features]
y=df['target']
x=pd.DataFrame(prep.MinMaxScaler().fit_transform(x))
x=pd.DataFrame(prep.StandardScaler().fit_transform(x))
x.columns = features

x_no_out=df_no_out[features]
y_no_out=df_no_out['target']
x_no_out=pd.DataFrame(prep.MinMaxScaler().fit_transform(x_no_out))
x_no_out=pd.DataFrame(prep.StandardScaler().fit_transform(x_no_out))
x_no_out.columns = features


features_cor=data_less_cor.columns[:12].tolist()
x_cor=data_less_cor[features_cor]
y_cor=data_less_cor['target']
x_cor=pd.DataFrame(prep.MinMaxScaler().fit_transform(x_cor))
x_cor=pd.DataFrame(prep.StandardScaler().fit_transform(x_cor))
x_cor.columns = features_cor

#Ovaj deo se ponavlja samo sada uklanjamo outliers-e manjoj dimenziji
df_x2=data_less_cor.drop(data_less_cor.columns[-1],axis=1)
df_y2=data_less_cor[data_less_cor.columns[-1]]
Q1_2 = df_x2.quantile(0.25)
Q3_2 = df_x2.quantile(0.75)
IQR_2 = Q3_2 - Q1_2
v=1.5

data_less_cor_no_out_x=df_x2[~((df_x2 < (Q1_2 - v*IQR_2)) | (df_x2 > (Q3_2 + v*IQR_2))).any(axis=1)]
data_less_cor_no_out_y=df_y2[~((df_x2 < (Q1_2 - v*IQR_2)) | (df_x2 > (Q3_2 + v*IQR_2))).any(axis=1)]

new_idxs2=pd.RangeIndex(len(data_less_cor_no_out_y.index))
data_less_cor_no_out_x.index=new_idxs2
data_less_cor_no_out_y.index=new_idxs2

data_less_corr_no_out=pd.concat([data_less_cor_no_out_x,data_less_cor_no_out_y], axis=1)

x_no_out2=data_less_corr_no_out[features_cor]
y_no_out2=data_less_corr_no_out['target']
x_no_out2=pd.DataFrame(prep.MinMaxScaler().fit_transform(x_no_out2))
x_no_out2=pd.DataFrame(prep.StandardScaler().fit_transform(x_no_out2))
x_no_out2.columns = features_cor


x_without_prep_train, x_without_prep_test, y_without_prep_train, y_without_prep_test= train_test_split(x_without_prep, y_without_prep, train_size=0.7, stratify=y_without_prep)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)
x_train_no_out, x_test_no_out, y_train_no_out, y_test_no_out = train_test_split(x_no_out, y_no_out, train_size=0.7, stratify=y_no_out)
x_train_cor, x_test_cor, y_train_cor, y_test_cor = train_test_split(x_cor, y_cor, train_size=0.7, stratify=y_cor)
x_train_no_out2, x_test_no_out2, y_train_no_out2, y_test_no_out2 = train_test_split(x_no_out2, y_no_out2, train_size=0.7, stratify=y_no_out2)

#Unakrsna validacija koju cemo koristiti za KNN
def cross_validation(algorithm, parameters, scores, x_train, y_train, x_test, y_test):

    for score in scores:
        print("Mera ", score)

        clf = GridSearchCV(algorithm, parameters, cv=10, scoring='%s_macro' % score)
        clf.fit(x_train, y_train)

        print("Najbolji parametri:")
        print(clf.best_params_)

        print("Izvestaj za test skup:")
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))


        y_pred = clf.predict(x_test)
        cnf_matrix = met.confusion_matrix(y_test, y_pred)
        print("Matrica konfuzije", cnf_matrix, sep="\n")

        print("\n")
        acc = met.accuracy_score(y_test, y_pred)
        print("Preciznost", acc)
        print("\n")

#KNN
def knn(x_train_, y_train_,  x_test_, y_test_):
    param = [{'n_neighbors': range(3, 15),
              'p': [1, 2],
              'weights': ['uniform', 'distance'],
              }]

    scores = ['precision', 'f1']
    cross_validation(KNeighborsClassifier(), param, scores, x_train_, y_train_,  x_test_, y_test_)


#Naivni Bajes
def g_nayve_bayes(x_train, y_train, x_test, y_test):
    clf_gnb = GaussianNB()
    clf_gnb.fit(x_train, y_train)

    y_pred = clf_gnb.predict(x_test)

    cnf_matrix = met.confusion_matrix(y_test, y_pred)
    print("Matrica konfuzije", cnf_matrix, sep="\n")
    print("\n")

    acc = met.accuracy_score(y_test, y_pred)
    print("Preciznost", acc)
    print("\n")

    class_report = met.classification_report(y_test, y_pred, target_names=df["target"].unique())
    print("Izvestaj klasifikacije", class_report, sep="\n")


def b_nayve_bayes(x_train, y_train, x_test, y_test):
    clf_gnb = BernoulliNB()
    clf_gnb.fit(x_train, y_train)

    y_pred = clf_gnb.predict(x_test)

    cnf_matrix = met.confusion_matrix(y_test, y_pred)
    print("Matrica konfuzije", cnf_matrix, sep="\n")
    print("\n")

    acc = met.accuracy_score(y_test, y_pred)
    print("Preciznost", acc)
    print("\n")

    class_report = met.classification_report(y_test, y_pred, target_names=df["target"].unique())
    print("Izvestaj klasifikacije", class_report, sep="\n")

print()
print('KNN nad nesredjenim podacima') #nisu normalizovani ni standardizovani
print()
knn(x_without_prep_train, y_without_prep_train, x_without_prep_test, y_without_prep_test)

print('KNN nad normalizovanim i standardizovanim podacima')
print()
knn(x_train, y_train, x_test, y_test)

print('KNN nad normalizovanim i standardizovanim podacima bez outlier-a')
print()
knn(x_train_no_out, y_train_no_out, x_test_no_out, y_test_no_out)

print('KNN nad normalizovanim i standardizovanim podacima kojima je smanjena dimenzija')
print()
knn(x_train_cor, y_train_cor, x_test_cor, y_test_cor)


print('KNN nad normalizovanim i standardizovanim podacima kojima je smanjena dimenzija i iz kojih su izbaceni outliers-i')
print()
knn(x_train_no_out2, y_train_no_out2, x_test_no_out2, y_test_no_out2)


print()

print('Gaussian Nayve Bayes nad nesredjenim podacima') #nisu normalizovani ni standardizovani
print()
g_nayve_bayes(x_without_prep_train, y_without_prep_train, x_without_prep_test, y_without_prep_test)

print('Gaussian Nayve Bayes nad normalizovanim i standardizovanim podacima')
print()
g_nayve_bayes(x_train, y_train, x_test, y_test)

print('Gaussian Nayve Bayes nad normalizovanim i standardizovanim podacima bez outlier-a')
print()
g_nayve_bayes(x_train_no_out, y_train_no_out, x_test_no_out, y_test_no_out)

print('Gaussian Nayve Bayes nad normalizovanim i standardizovanim podacima kojima je smanjena dimenzija')
print()
g_nayve_bayes(x_train_cor, y_train_cor, x_test_cor, y_test_cor)


print('Gaussian Nayve Bayes nad normalizovanim i standardizovanim podacima kojima je smanjena dimenzija i iz kojih su izbaceni outliers-i')
print()
g_nayve_bayes(x_train_no_out2, y_train_no_out2, x_test_no_out2, y_test_no_out2)

print()
print('Bernoulli Nayve Bayes nad nesredjenim podacima') #nisu normalizovani ni standardizovani
print()
b_nayve_bayes(x_without_prep_train, y_without_prep_train, x_without_prep_test, y_without_prep_test)

print('Bernoulli Nayve Bayes nad normalizovanim i standardizovanim podacima')
print()
b_nayve_bayes(x_train, y_train, x_test, y_test)

print('Bernoulli Nayve Bayes nad normalizovanim i standardizovanim podacima bez outlier-a')
print()
b_nayve_bayes(x_train_no_out, y_train_no_out, x_test_no_out, y_test_no_out)

print('Bernoulli Nayve Bayes nad normalizovanim i standardizovanim podacima kojima je smanjena dimenzija')
print()
b_nayve_bayes(x_train_cor, y_train_cor, x_test_cor, y_test_cor)


print('Bernoulli Nayve Bayes nad normalizovanim i standardizovanim podacima kojima je smanjena dimenzija i iz kojih su izbaceni outliers-i')
print()
b_nayve_bayes(x_train_no_out2, y_train_no_out2, x_test_no_out2, y_test_no_out2)
