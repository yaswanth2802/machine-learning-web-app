import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


st.title("""
         EXPLORE DIFFERENT CLASSIFIERS
         """)
dataset_name=st.sidebar.selectbox("select Data Set",("Iris","Breast Cancer","Wine","Digits","DIABETES","BOSTON","LINNERUD"))
classifier_name=st.sidebar.selectbox("Select Classifier",("KNN","SVM","RANDOM FOREST","DECISION TREE","GAUSSION NB","MLP","ADABoost","QUADRATIC DISCRIMINANT"))

def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    elif dataset_name=="Wine":
        data=datasets.load_wine()
    elif dataset_name:
        data=datasets.load_digits()
    x=data.data
    y=data.target
    return x,y
x,y=get_dataset(dataset_name)
st.write("SHAPE OF DATASET",x.shape)


def add_parameter(clf_name):
    p=dict()
    if clf_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        p["K"]=K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01,10.00)
        p["C"] =C
    elif clf_name == "RANDOM FOREST":
        M_D= st.sidebar.slider("M_D",2,15)
        N_E=st.sidebar.slider("N_E",1,100)
        p["M_D"] =M_D
        p["N_E"]=N_E
    elif clf_name == "DECISION TREE":
        M_DD = st.sidebar.slider("M_DD", 2, 15)
        p["M_DD"] = M_DD
    elif clf_name == "MLP":
        p["A"] = 1
    return p


p=add_parameter(classifier_name)

def get_classifier(clf_name,p):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=p["K"])
    elif clf_name=="SVM":
        clf=SVC(C=p["C"])
    elif clf_name=="RANDOM FOREST":
        clf=RandomForestClassifier(max_depth=p["M_D"],n_estimators=p["N_E"],random_state=900)
    elif clf_name=="DECISION TREE":
        clf=DecisionTreeClassifier(max_depth=p["M_DD"])
    elif clf_name=="GAUSSION NB":
        clf=GaussianNB()
    elif clf_name=="MLP":
        clf=MLPClassifier(alpha=p["A"], max_iter=1000)
    elif clf_name=="ADABoost":
        clf=AdaBoostClassifier()
    elif clf_name == "QUADRATIC DISCRIMINANT":
        clf = QuadraticDiscriminantAnalysis()
    return clf


clf=get_classifier(classifier_name,p)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=900)
clf.fit(x_train,y_train)
y_p = clf.predict(x_test)
a=accuracy_score(y_test,y_p)
st.write(f"CLASSIFIER={classifier_name}")
st.write(f"ACCURACY={a}")

pca=PCA(2)
x_project=pca.fit_transform(x)


x1=x_project[:,0]
x2=x_project[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("principle component 1")
plt.ylabel("principle component 2")
plt.colorbar()

st.pyplot(bbox_inches='tight')
st.set_option('deprecation.showPyplotGlobalUse', False)