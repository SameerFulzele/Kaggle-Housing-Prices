import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import warnings; warnings.simplefilter('ignore')
import statsmodels.api as sm 
import gc
import re


train =  pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def Encoding(dataframe, ordinal_ratings, nominal):
    df= dataframe.copy()   
    
    for i in ordinal_ratings:
        encoder= ce.OrdinalEncoder(cols=[str(i)], return_df=True, mapping=[{'col':str(i), 'mapping':{'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}}])
        df[str(i)]= encoder.fit_transform(df[str(i)])
                
    for i in nominal:
        encoder=ce.OneHotEncoder(cols=str(i),handle_unknown='return_nan',return_df=True,use_cat_names=True)
        con= encoder.fit_transform(df[str(i)])
        df= pd.concat([df,con], axis=1)
        
#     for i in ordinal:
#         encoder= ce.OrdinalEncoder(cols=[str(i)], return_df=True)
#         df[str(i)]= encoder.fit_transform(df[str(i)])
    
    df = df.drop(nominal, axis = 1) 
    gc.collect()
    return df 

######################################################################################################################

selected_columns= train[['SalePrice','ExterQual','MSSubClass','LotFrontage','BedroomAbvGr', 
                         '1stFlrSF','OverallCond','OverallQual','KitchenQual','BsmtCond','HeatingQC',
                         'CentralAir','YearBuilt','YearRemodAdd','LotArea', 'LotShape','LandContour','BldgType']]

df_train = selected_columns.copy()
ordinal_ratings = [ 'BsmtCond' ,'HeatingQC','KitchenQual','ExterQual']
nominal = [ 'CentralAir','BldgType' ,'LotShape','LandContour']
df_train = Encoding(df_train,ordinal_ratings,nominal)
df_train['LotFrontage']=df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())
X = df_train.drop('SalePrice' , axis=1)
Y = df_train['SalePrice']



######################################################################################################################
algos = {
    'Linear_Regression' : {'model': LinearRegression,
                            'parameters': {'normalize': [True,False]
                                              }
                          },

                 'Lasso': {'model': Lasso,
                           'parameters': {'alpha': [0.01,0.1,0.5,1,2,5,10,20],
                                          'selection': ['random', 'cyclic']
                                         }
                          },

                 'Ridge': {'model' : Ridge, 
                            'parameters':{'alpha': [0.01,0.1,0.5,1,2,5,10,20]
                                         }
                          } ,
         
        
         'Decision_Tree': {'model': DecisionTreeRegressor, 
                            'parameters': {'criterion' : ['mse','friedman_mse','mae'],
                                           'splitter': ['best','random']
                                          }
                          }
         
         
        }


######################################################################################################################


class Model():

    seed = 100 
    def __init__(self,dataframe,X,Y,algo_dictionary, problem_type):
        self.df = dataframe
        self.algos = algo_dictionary
        self.problem_type = problem_type
        self.X = X
        self.Y = Y
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size=0.2,random_state= self.seed)


    def Gridsearch(self,name,update_best_features = None):


        for key,value in self.algos[name].items():

            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)
            gs =  GridSearchCV(value['model'](), value['parameters'], cv=cv, return_train_score=False)
            gs.fit(self.X,self.Y)

            if update_best_features is True:
                value['parameters'] = gs.best_params_


        print({'model': key,'best_score': gs.best_score_,'best_params': gs.best_params_})


    def Gridsearch_all(self,update_best_features = None):


        table = []
        for key,value in self.algos.items():

            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state= self.seed)
            gs =  GridSearchCV(value['model'](), value['parameters'], cv=cv, return_train_score=False)
            gs.fit(self.X,self.Y)
            table.append({'model': key,'best_score': gs.best_score_,'best_params': gs.best_params_})

            if update_best_features is True:
                value['parameters'] = gs.best_params_


        table = pd.DataFrame(table)
        print(table)


    def train(self,algorithm_name):


        parameters = self.algos[algorithm_name]['parameters']
        self.algos[algorithm_name]['model'] = self.algos[algorithm_name]['model'](**parameters)
        self.self.algos[algorithm_name]['model'].fit(self.X_train,self.Y_train)


        # for key,value in algos[name].items():
        # 	parameters = value['parameters']
        # 	value['model'] = value['model'](**parameters)
        # 	value['model'].fit(x_train, y_train)


        predictions = self.algos[algorithm_name]['model'].predict(self.X_test)
        if self.problem_type == 'Classification':
            print('Classification report after predicting on testing data for', algorithm_name)
            print(classification_report(self.Y_test, predictions))
            print('TN,FN')
            print('FP,TP')
            print(confusion_matrix(self.Y_test, predictions))


    def train_all(self):
        for key,value in self.algos.items():
            parameters = value['parameters']
            value['model'] = value['model'](**parameters)
            value['model'].fit(self.X_train, self.Y_train)

            predictions = value['model'].predict(self.X_test)
            if self.problem_type == 'Classification':
                print('Classification report after predicting on testing data for', key)
                print(classification_report(self.Y_test, predictions))
                print('TN,FN')
                print('FP,TP')
                print(confusion_matrix(self.Y_test, predictions))
                print('\n')

            if self.problem_type == 'Regression':
                pass


    def predict(self,x_input,algorithm_name):
        predictions = self.algos[algorithm_name]['model'].predict(x_input)
        return predictions


    def feature_importances(self,x,y):
        pass


    def permutation_importances(self,x,y):
        pass


    def remove_algo(self, name):
        del self.algos[str(name)]


    def add_algo(self,dict):
        self.algos.update(dict)


    def reset_seed(self , new_seed):
        self.seed = new_seed


    def report(self):
        if self.problem_type == 'Regression':
            pass

        elif self.problem_type == 'Classification':
            pass

        else:
            print('Please define problem_type')


m1 = Model(df_train,X,Y,algos,'Regression')
m1.Gridsearch_all()
