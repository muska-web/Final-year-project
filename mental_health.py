# suicide prediction program
# import time
'''
suicide prediction program is working 
            but 
will take time so dont quit in middle
'''

# import modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import time

# sklearn module for tuning
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import preprocessing # For DAta encoding and Labeling
from sklearn.model_selection import train_test_split # for splitting the dataset
from sklearn.ensemble import ExtraTreesClassifier # for feature importancs calculation
from sklearn.metrics import accuracy_score 

# sklearn modules for model creation
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

start = time.time()

# data loading, checking, cleaning and encoding

'''
- Data Cleaning nd Encoding
- Corrlation Matrix
- Splitting the data into training and testing
- Feature importance
'''

# data loading

# enter the location of your input file
# input_location = "input.csv"
input_location = (input("Enter your input file location: "))


# check if the file exists
while not os.path.isfile(input_location):
    print("File does not exist")
    exit()
# Check input and read file
if(input_location.endswith(".csv")):
    data = pd.read_csv(input_location)
elif(input_location.endswith(".xlsx")):
    data = pd.read_excel(input_location)
# elif(input_location.endswith(".json")):
#     data = pd.read_json(input_location)
else:
    print("ERROR: File format not supported!")
    exit()


# check data
variable = ['family_size', 'annual_income', 'eating_habits',
            'addiction_friend', 'addiction', 'medical_history',
            'depressed', 'anxiety', 'happy_currently', 'suicidal_thoughts']
check = all(item in list(data) for item in variable)
if check is True:
    print("Data is loaded")
else:
    print("Dataset doesnot contain: ", variable)
    exit()
print("Data Loaded and Checked")
print("Loded data is:\n")
print(data)

# data Cleaning
## drop unnecessary columns
if '_id' in data:
    data = data.drop(['_id'], axis=1)
elif 'Timestamp' in data:
    data = data.drop(['Timestamp'], axis=1)
print("Data Cleaned")

# data encoding
labelDictionary = {}
for feature in data:
    le = preprocessing.LabelEncoder()
    le.fit(data[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    data[feature] = le.transform(data[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDictionary[labelKey] = labelValue
print("Data Encoded")


# correlation matrix
print(data)
corr = data.corr(method='pearson')
print("\n")
print("Correlation Matrix:\n")
print(corr)
print("\n")
f, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(corr, vmax=.8, square=True, annot=True)
# plt.show()
plt.savefig('output_graph/CorrelationMatrix.png')


#Splitting X and y into training and testing sets
#Splitting the data
independent_vars = ['family_size', 'annual_income', 'eating_habits', 
                    'addiction_friend', 'addiction', 'medical_history', 
                    'depressed', 'anxiety', 'happy_currently']
X = data[independent_vars] 
y = data['suicidal_thoughts']
#Splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#visualising the feature importance
#using ExtraTreesClassifier to acertain important features
frst = ExtraTreesClassifier(random_state=0)
frst.fit(X,y)
imp = frst.feature_importances_
stan_dev = np.std([tree.feature_importances_ for tree in frst.estimators_], axis = 0)

#creating an nparray with decreasing order of importance of features
indices = np.argsort(imp)[::-1]

#appending column names to labels list
labels = []
for f in range(X.shape[1]):
    labels.append(X.columns[f])

#ploting feature importance bar graph
plt.figure(figsize=(12,8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), imp[indices],
    color="g", yerr=stan_dev[indices], align="center")
plt.xticks(range(X.shape[1]), labels, rotation='vertical')
plt.xlim([-1, X.shape[1]])
# plt.show()
plt.savefig('output_graph/FeatureImportance.png')

#Dictionary to store accuracy results of different algorithms
accuracyDict = {}

#Dictionary to store time log of different funcitons
timelog = {}

#creating a confunsion matrix
def evalModel(model, X_test, y_test, y_pred_class):
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    #creating a confunsion matrix
    conmat = metrics.confusion_matrix(y_test, y_pred_class)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    plt.figure()
    sns.heatmap(conmat, annot=True, cbar=False)
    plt.title("Confusion " + str(model))
    plt.xlabel("predicted")
    plt.ylabel("Actual")
    # plt.show()
    plt.savefig("output_graph/Confusion_" + str(model).partition("(")[0] + ".png")
    return acc_score
'''
- Tuning
'''
def get_csv_output(model, y_test, y_pred_class):
    name = str(model).partition('(')
    y_pred = pd.Series(y_pred_class, name='predictions')
    print(y_pred)
    print()
    print(type(y_test))
    measure = y_test.reset_index(drop=True)
    print(measure)
    # measure = pd.DataFrame(measure)
    # output_data = measure.join(y_pred)
    output_data = pd.concat([measure, y_pred], axis=1)
    csv = pd.DataFrame(output_data)
    # path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/output_result/"
    path = "output_result/"
    file_name = path + name[0] +'.csv'
    csv.to_csv(file_name, header=True)


    # Tuning with GridSearchCV
def GridSearch(X_train, X_test, y_train, y_test, accuracyDict, timelog):
    start = time.time()
    log_reg_mod(X_train, X_test, y_train, y_test, accuracyDict)
    tuneKNN(X_train, X_test, y_train, y_test, accuracyDict)
    tuneDT(X_train, X_test, y_train, y_test, accuracyDict)
    tuneRF(X_train, X_test, y_train, y_test, accuracyDict)
    tuneBoosting(X_train, X_test, y_train, y_test, accuracyDict)
    tuneBagging(X_train, X_test, y_train, y_test, accuracyDict)
    tuneStacking(X_train, X_test, y_train, y_test, accuracyDict)
    print("The accuracy of the models are: ")
    print(accuracyDict)
    end = time.time()
    timelog['GridSearch models'] = end - start

# tuning the logistic regression model with Gridsearchcv
def log_reg_mod(X_train, X_test, y_train, y_test, accuracyDict):
    global lr
    print("\nTuning the Logistic Regression Model with GridSearchCV\n")
    param_grid = {'C':[0.1,1,10,100,1000],
                  'solver':['newton-cg','lbfgs','sag'],
                  'multi_class':['ovr','multinomial'],
                  'max_iter':[100,200,300,400,500]}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, n_jobs=-1,  cv=5)
    grid_search.fit(X_train,y_train)
    # print("Best param_grids: ", grid_search.best_params_)
    # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # print("Best estimator: ", grid_search.best_estimator_)
    lr = grid_search.best_estimator_
    y_pred_class = lr.predict(X_test)
    accuracy = evalModel(lr, X_test, y_test, y_pred_class)
    print(accuracy)
    accuracyDict['Log_Reg_mod_GSCV'] = accuracy * 100
    print("this is X_test:")
    print(X_test)
    print("this is y_test:")
    print(y_test)
    print("this is y_pred:")
    print(y_pred_class)
    get_csv_output(lr, y_test, y_pred_class)

# tuning the KNN model with GridSearchCV
def tuneKNN(X_train, X_test, y_train, y_test, accuracyDict):
    global knn
    print("\nTuning KNN model with GridSearchCV\n")
    param_grid = {'n_neighbors':[3,5,7,9,11,13,15],
                  'weights':['uniform','distance'],
                  'algorithm':['auto','ball_tree','kd_tree','brute'],
                  'leaf_size':[10,20,30,40,50,60,70,80]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1,  cv=5)
    grid_search.fit(X_train,y_train)
    # print("Best param_grids: ", grid_search.best_params_)
    # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # print("Best estimator: ", grid_search.best_estimator_)
    knn = grid_search.best_estimator_
    y_pred_class = knn.predict(X_test)
    accuracy = evalModel(knn,X_test, y_test, y_pred_class)
    accuracyDict['KNN_GSCV'] = accuracy * 100
    get_csv_output(knn, y_test, y_pred_class)

# tuning the Decision Tree model with GridSearchCV
def tuneDT(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Decision Tree model with GridSearchCV\n")
    param_grid = {'criterion':['gini','entropy'],'max_depth':[3,5,7,9,11,13,15],
                  'min_samples_split':[2,3,4,5,6,7,8],'random_state':[0]}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, n_jobs=-1,  cv=5)
    grid_search.fit(X_train,y_train)
    # print("Best param_grids: ", grid_search.best_params_)
    # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # print("Best estimator: ", grid_search.best_estimator_)
    dt = grid_search.best_estimator_
    y_pred_class = dt.predict(X_test)
    accuracy = evalModel(dt,X_test, y_test, y_pred_class)
    accuracyDict['DecisionTree_GSCV'] = accuracy * 100
    get_csv_output(dt, y_test, y_pred_class)

# tuning the Random Forest model with GridSearchCV
def tuneRF(X_train, X_test, y_train, y_test, accuracyDict):
    global rf
    print("\nTuning Random Forest model with GridSearchCV\n")
    param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],'max_depth':[3,5,7,9,11,13,15],
            'min_samples_split':[2,3,4,5,6,7,8],'criterion':['gini','entropy'],'random_state':[0]}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1,  cv=5)
    grid_search.fit(X_train,y_train)
    # print("Best param_grids: ", grid_search.best_params_)
    # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # print("Best estimator: ", grid_search.best_estimator_)
    rf = grid_search.best_estimator_
    y_pred_class = rf.predict(X_test)
    accuracy = evalModel(rf,X_test, y_test, y_pred_class)
    accuracyDict['RandomForest_GSCV'] = accuracy * 100
    get_csv_output(rf, y_test, y_pred_class)

# tuning boosting model with GridSearchCV
def tuneBoosting(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Boosting model with GridSearchCV\n")
    param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
                  'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],'random_state':[0]}
    grid_search = GridSearchCV(AdaBoostClassifier(), param_grid, n_jobs=-1,  cv=5)
    grid_search.fit(X_train,y_train)
    # print("Best param_grids: ", grid_search.best_params_)
    # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # print("Best estimator: ", grid_search.best_estimator_)
    ada = grid_search.best_estimator_
    y_pred_class = ada.predict(X_test)
    accuracy = evalModel(ada,X_test, y_test, y_pred_class)
    accuracyDict['AdaBoost_GSCV'] = accuracy * 100
    get_csv_output(ada, y_test, y_pred_class)

# tuning bagging model with GridSearchCV
def tuneBagging(X_train, X_test, y_train, y_test, accuracyDict):
    print("\nTuning Bagging model with GridSearchCV\n")
    param_grid = {'n_estimators':[10,20,30,40,50,60,70,80,90,100],
                  'max_samples':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                  'bootstrap':[True,False],'bootstrap_features':[True,False],
                  'random_state':[0]}
    grid_search = GridSearchCV(BaggingClassifier(), param_grid, n_jobs=-1,  cv=5)
    grid_search.fit(X_train,y_train)
    # print("Best param_grids: ", grid_search.best_params_)
    # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # print("Best estimator: ", grid_search.best_estimator_)
    bag = grid_search.best_estimator_
    y_pred_class = bag.predict(X_test)
    accuracy = evalModel(bag,X_test, y_test, y_pred_class)
    accuracyDict['Bagging_GSCV'] = accuracy * 100
    get_csv_output(bag, y_test, y_pred_class)
    
# tuning stacking model with GridSearchCV
def tuneStacking(X_train, X_test, y_train, y_test, accuracyDict):
    classifiers=[('rf',rf),('lr', lr),('knn', knn)]
    print("\nTuning Stacking model with GridSearchCV\n")
    param_grid = {'stack_method': ['predict_proba', 'decision_function', 'predict']}
    grid_search = GridSearchCV(StackingClassifier(estimators = classifiers), param_grid, n_jobs=-1,  cv=5)
    grid_search.fit(X_train,y_train)
    # print("Best param_grids: ", grid_search.best_params_)
    # print("Best cross-validation score: ", grid_search.best_score_*100, "%")
    # print("Best estimator: ", grid_search.best_estimator_)
    stack = grid_search.best_estimator_
    y_pred_class = stack.predict(X_test)
    accuracy = evalModel(stack,X_test,y_test, y_pred_class)
    accuracyDict['Stacking_GSCV'] = accuracy * 100
    get_csv_output(stack, y_test, y_pred_class)

# save accuracyDict accuracy Bar Graph to file
def get_accuracy(accuracyDict):
    s = pd.Series(accuracyDict)
    s = s.sort_values(ascending=False)
    plt.figure(figsize=(12,8))
    ax = s.plot(kind='bar') 
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.xlabel('Method')
    plt.ylabel('Percentage')
    plt.title('Success of methods')
    # plt.show()
    plt.savefig('output_graph/AccuracyBarGraph.png')

    # acc_score = metrics.accuracy_score(y_test, y_pred_class)
    # print("Accuracy: ", acc_score)
    # print("NULL Accuracy: ", y_test.value_counts())
    # print("Percentage of ones: ", y_test.mean())
    # print("Percentage of zeros: ", 1 - y_test.mean())
GridSearch(X_train, X_test, y_train, y_test,accuracyDict, timelog)
get_accuracy(accuracyDict)

end = time.time()
print("Time Taken by Grid Search: ", timelog['GridSearch models'], "Seconds")
print("Total Time taken: ", end - start,"seconds")
