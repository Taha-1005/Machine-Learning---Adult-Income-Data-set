import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score,classification_report
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import io

st.set_page_config(layout="wide")

#-----------------------------------------------------------------

st.markdown("""
<style>
div.stButton > button:first-child {
background-color: #00cc00;
color:black;
font-size:15px;
height:2.7em;
width:20em;
border-radius:10px 10px 10px 10px;}
</style>
    """,
    unsafe_allow_html=True
)


st.markdown(
     """
     <style>
     .reportview-container {
         background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTI8DRxgEP4PMaChfWJQKulfwMWdF486bB0SF0ZHXkgS5z4gc2Jd7EGKC8-gjjKWNxEUlQ&usqp=CAU")
     }
     </style>
     """,
    unsafe_allow_html=True
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


#-------------------------------------------------------------

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set_style("darkgrid")
#plt.style.use("dark_background")

st.title("ML algorithms and their analysis ")
st.sidebar.title("ML algorithms ")
st.subheader("This application shows graphs for the given algorithms ")
st.sidebar.markdown("This application is a ML algo dashboard ")
show = st.sidebar.checkbox("SHOW")

st.sidebar.subheader("ACCURACY OF MODELS")

st.sidebar.title("ALGORITHMS")
button1 = st.sidebar.button("Decision Tree Classifier")
button2 = st.sidebar.button("Bagging Classifier")
button3 = st.sidebar.button("Random Forest Classifier")
button4 = st.sidebar.button("Cat Boost Classifier")
button5 = st.sidebar.button("Ada classifier")
button6 = st.sidebar.button("Knn Classifier")





option = st.selectbox(
    'Select any of the following.',
    ('Home','Show data', 'Data scrubbing', 'Data visulization','Label Encoding'))


if show:

    classifier = ['Decision Tree','Bagging','Random Forest','Cat Boost','Ada Boost','KNN']
    training = [0.856293,0.855321,0.897255,0.881714,0.838101,0.827667]
    testing = [0.837460,0.830323,0.858081,0.871695,0.841690,0.801110]

    accuracy = {'Classifiers' : classifier, 'Training' : training, 'Testing '  : testing}

    acc = pd.DataFrame(accuracy)
    st.subheader("ACCURACY TABLE")
    st.dataframe(acc)

    st.subheader("Training Data")
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(training)), training, tick_label=classifier, width=0.4)
    plt.xlabel("Accuracy")
    plt.ylabel("Classifier")
    st.pyplot()

    st.subheader("Testing Data")
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(testing)), testing, tick_label=classifier, width=0.4)
    plt.xlabel("Accuracy")
    plt.ylabel("Classifier")
    st.pyplot()




def to_csv(filename):
    
    df = pd.read_csv(filename,header=None)

    df.columns = ['age','workclass','fnlwgt','education','education_num','marital_satus',
                  'occupation','relationship','race','sex','capital_gain','capital_loss',
                  'hours_per_week','native_country','class']
    
    df = df.replace( '[\?,)]',np.nan, regex=True )
    return df

    
def create_table(df):

    fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=df.transpose().values.tolist(),
               fill_color='lavender',
               align='left'))
    ])

    
    fig.update_layout(width=1050, height=600,margin=dict(l=0,r=0,b=0,t=0))
    
    return fig




df_train = to_csv("adult.data")
df_test =  to_csv("adult.test")

fig_train = create_table(df_train)
fig_test = create_table(df_test)

df_train_final = df_train.dropna().drop_duplicates()
df_test_final = df_test.dropna().drop_duplicates()


if option == 'Home':
    st.write()


if option == 'Show data':
    st.subheader('Training set')
    st.plotly_chart(fig_train)
    st.write("The shape of train set(before cleaning) : ",str(df_train.shape))
    
    st.subheader('Testing set')
    st.plotly_chart(fig_test)
    st.write("The shape of test set(before cleaning)  : ",str(df_test.shape))

    

if option == 'Data scrubbing':
    
    genre = st.radio("Select",('Extract NAN', 'Display duplicates'))

    if genre == 'Extract NAN':
        st.subheader('Removing unknown values from Training set')
        df_train_nan = df_train.dropna()
        st.plotly_chart(create_table(df_train_nan))
        st.write("The shape of train set(after removing NAN) : ",str(df_train_nan.shape))


        st.subheader('Removing unknown values from Test set')
        df_test_nan  = df_test.dropna()
        st.plotly_chart(create_table(df_test_nan))
        st.write("The shape of test set(after removing NAN)  : ",str(df_test_nan.shape))
        
    if  genre == 'Display duplicates':

        st.subheader('Displaying duplicates Training set')
        duplicate_train = df_train[df_train.duplicated()]
        st.write("Duplicates of training set : ",str(duplicate_train.shape))
        st.plotly_chart(create_table(duplicate_train))
        

        st.subheader('Displaying duplicates Test set')
        duplicate_test = df_test[df_test.duplicated()]
        st.write("Duplicates of testing set : ",str(duplicate_test.shape))
        st.plotly_chart(create_table(duplicate_test))


if option == 'Data visulization':

        radio = st.radio("",('Gender', 'Education','Occupation','Race','Features'))

        def precent_plot(df,col,ax):
                ax = sns.countplot(data=df, x=col,order=df[col].value_counts().index)
                for p in ax.patches:
                    percentage = '{:.1f}%'.format(100 * p.get_height()/len(df))
                    x = p.get_x() + p.get_width()*0.4
                    y = p.get_y()  + p.get_height()
                    ax.annotate(percentage, (x, y))

        if radio == 'Gender':
                
            fig = plt.figure()
            ax_1 = fig.add_subplot(111)
            st.subheader("TRAINING SET")
            plt.title('Gender Number Representation')
            precent_plot(df_train_final,'sex',ax_1)
            st.pyplot(fig)

            fig2 = plt.figure()
            sns.boxenplot(x='sex',y='hours_per_week',data=df_train_final)
            plt.title("Work Hours per Week",fontsize=16);
            st.pyplot(fig2)

        if radio == 'Education':
            fig = plt.figure(figsize=(14,8))

            ax_1 = fig.add_subplot(211)
            sns.countplot(x='education',hue='sex',palette='tab10',order=df_train_final['education'].value_counts().index,data=df_train_final,ax=ax_1)
            plt.xticks(rotation=45)

            fig.tight_layout(pad=3.0)

            ax_2 = fig.add_subplot(212)
            sns.countplot(x='education',hue='class',palette='tab10',order=df_train_final['education'].value_counts().index,data=df_train_final,ax=ax_2)
            plt.xticks(rotation=45)

            st.pyplot(fig)

        if radio == "Occupation":
            fig = plt.figure(figsize=(14,8))

            ax_1 = fig.add_subplot(211)
            sns.countplot(x='occupation',hue='sex',palette='tab10',order=df_train_final['occupation'].value_counts().index,data=df_train_final,ax=ax_1)
            plt.xticks(rotation=45)

            fig.tight_layout(pad=3.0)

            ax_2 = fig.add_subplot(212)
            sns.countplot(x='occupation',hue='class',palette='tab10',order=df_train_final['occupation'].value_counts().index,data=df_train_final,ax=ax_2)
            plt.xticks(rotation=45)

            st.pyplot(fig)

        if radio == 'Features':
            quantitative = ['age', 'fnlwgt', 'education_num','capital_gain', 'capital_loss', 'hours_per_week']
            fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(14,10))
            fig.suptitle('Features Distribution',fontsize=16,fontweight='bold')
            for i, ax in enumerate(axes.flat):
                sns.histplot(df_train[quantitative].iloc[:,i],kde=True,ax=ax)

            st.pyplot(fig)


        if radio == 'Race':
            fig = plt.figure(figsize=(8,4))
            ax_1 = fig.add_subplot(121)
            plt.xticks(rotation=90)
            plt.title('Race Number Representation',fontsize=16)
            precent_plot(df_train_final,'race',ax_1)

            st.pyplot(fig)


d = defaultdict(LabelEncoder)
def encode(col):
    if col.dtype == 'object':
        return d[str(col)].fit_transform(col)
    else:
        return col 
    
df_train_encode = df_train_final.apply(encode)
df_test_encode = df_test_final.apply(encode)


if option == 'Label Encoding':

    st.subheader('Training set')
    st.plotly_chart(create_table(df_train_encode))
    
    
    st.subheader('Testing set')
    st.plotly_chart(create_table(df_test_encode))
    



smote = SMOTE()
X_train, y_train = smote.fit_resample(df_train_encode.drop(['class','fnlwgt'],axis=1),df_train_encode['class'])
X_test, y_test = smote.fit_resample(df_test_encode.drop(['class','fnlwgt'],axis=1),df_test_encode['class'])
# print(y_balanced.value_counts())
            


#------------------------------------------------
if button1:

    st.title("Decision Tree")
    st.markdown("Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.")

    tree = DecisionTreeClassifier(criterion='entropy',max_depth=12,random_state=42)
    tree.fit(X_train,y_train)

    st.subheader("Accuracy")
    st.write('Train accuracy : '+str(accuracy_score(y_train,tree.predict(X_train))))
    st.write('Test accuracy :  '+str(accuracy_score(y_test,tree.predict(X_test))))

    st.subheader("Classification Report")
    data = classification_report(y_test,tree.predict(X_test),output_dict=True)
    st.dataframe(pd.DataFrame.from_dict(data))
    #st.plotly_chart(create_table(pd.DataFrame.from_dict(data)))


    st.subheader("Confusion Matrix") 
    plot_confusion_matrix(tree,X_test,y_test)
    st.pyplot()



 
if button2:

    st.title("Bagging Classifier")
    st.markdown("Bagging (Bootstrap Aggregation) is used when our goal is to reduce the variance of a decision tree. Here idea is to create several subsets of data from training sample chosen randomly with replacement. Now, each collection of subset data is used to train their decision trees.")
    bagg = BaggingClassifier(max_samples=0.05)
    bagg.fit(X_train,y_train)

    st.subheader("Accuracy")
    st.write('Train accuracy:' +str(accuracy_score(y_train,bagg.predict(X_train))))
    st.write('Test accuracy:'  +str(accuracy_score(y_test,bagg.predict(X_test))))

    st.subheader("Classification Report")
    data = classification_report(y_test,bagg.predict(X_test),output_dict=True)
    st.dataframe(pd.DataFrame.from_dict(data))
    #st.plotly_chart(create_table(pd.DataFrame.from_dict(data)))


    st.subheader("Confusion Matrix") 
    plot_confusion_matrix(bagg,X_test,y_test)
    st.pyplot()

   
if button3:

    st.title("Random Forest")
    st.markdown("A random forest is a machine learning technique that's used to solve regression and classification problems. It utilizes ensemble learning, which is a technique that combines many classifiers to provide solutions to complex problems. A random forest algorithm consists of many decision trees.")
    model = RandomForestClassifier(n_estimators=100,max_depth=12,random_state=42)
    model.fit(X_train,y_train)

    st.subheader("Accuracy")
    st.write('Train accuracy:'+str(accuracy_score(y_train,model.predict(X_train))))
    st.write('Test accuracy:'  +str(accuracy_score(y_test,model.predict(X_test))))

    st.subheader("Classification Report")
    data = classification_report(y_test,model.predict(X_test),output_dict=True)
    st.dataframe(pd.DataFrame.from_dict(data))
    #st.plotly_chart(create_table(pd.DataFrame.from_dict(data,index=['precision','recall','f1-score','support'])))

    st.subheader("Confusion Matrix") 
    plot_confusion_matrix(model,X_test,y_test)
    st.pyplot()

    

if button4:

    st.title("Cat Boost")
    st.markdown("CatBoost is based on gradient boosted decision trees. During training, a set of decision trees is built consecutively. Each successive tree is built with reduced loss compared to the previous trees. The number of trees is controlled by the starting parameters.")
    clf = CatBoostClassifier(
        iterations=400,
        max_depth=3,
        random_seed=42,
        learning_rate=0.5,
        custom_loss=['AUC', 'Accuracy'])

    clf.fit(X_train, y_train,eval_set=(X_test, y_test),verbose=False)

    st.subheader("Accuracy")
    st.write('Train accuracy:'+ str(accuracy_score(y_train,clf.predict(X_train))))
    st.write('Test accuracy:'  +str(accuracy_score(y_test,clf.predict(X_test))))

    st.subheader("Classification Report")
    data = classification_report(y_test,clf.predict(X_test),output_dict=True)
    st.dataframe(pd.DataFrame.from_dict(data))
    # st.plotly_chart(create_table(pd.DataFrame.from_dict(data)))

    st.subheader("Confusion Matrix") 
    plot_confusion_matrix(clf,X_test,y_test)
    st.pyplot()

    

if button5:

    st.title("Ada Boost")
    st.markdown("AdaBoost algorithm, short for Adaptive Boosting, is a Boosting technique used as an Ensemble Method in Machine Learning. It is called Adaptive Boosting as the weights are re-assigned to each instance, with higher weights assigned to incorrectly classified instances")
    abc = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=15)
    abc.fit(X_train.values, y_train)
    
    st.subheader("Accuracy")
    st.write('Train accuracy:'+ str(accuracy_score(y_train,abc.predict(X_train))))
    st.write('Test accuracy:'  +str(accuracy_score(y_test,abc.predict(X_test))))


    st.subheader("Classification Report")
    data = classification_report(y_test,abc.predict(X_test),output_dict=True)
    st.dataframe(pd.DataFrame.from_dict(data))
    #st.plotly_chart(create_table(pd.DataFrame.from_dict(data)))


    st.subheader("Confusion Matrix") 
    plot_confusion_matrix(abc,X_test,y_test)
    st.pyplot()

    

if button6:
    st.title("KNN")
    st.markdown("What is Knn and how it works? KNN works by finding the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression).")

    pipe = Pipeline([
        ('sc', StandardScaler()),     
        ('knn', KNeighborsClassifier())])

    params = {
        'knn__n_neighbors': [23, 25, 27] 
    }
    grid = GridSearchCV(estimator=pipe,           
                      param_grid=params, 
                      cv=5,
                      return_train_score=False) # Turn on cv train scores
                      
    grid_search = grid.fit(X_train.values, y_train)

    neighbors = grid_search.best_params_['knn__n_neighbors']
    print(grid_search.best_params_['knn__n_neighbors'])

     
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(X_train,y_train)
    y_test_hat=knn.predict(X_test) 
    test_accuracy= accuracy_score(y_test,y_test_hat)

    st.subheader("Accuracy")
    st.write('Training accuracy : '+str(grid_search.best_score_))
    st.write("Testing accuracy : "+str(test_accuracy))


    st.subheader("Classification Report")
    data = classification_report(y_test,knn.predict(X_test),output_dict=True)
    st.dataframe(pd.DataFrame.from_dict(data))
    #st.plotly_chart(create_table(pd.DataFrame.from_dict(data)))

    st.subheader("Confusion Matrix") 
    plot_confusion_matrix(knn,X_test,y_test)
    st.pyplot()


   





    

