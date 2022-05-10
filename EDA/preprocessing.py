import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from scipy.special import boxcox1p
from from_root import from_root
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore") 

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

import pickle



class Preprocessing:
    def __init__(self):
        pass


    def dataloading(self):

        self.features = pd.read_csv(os.path.join(from_root(),"dataset","features.csv"))
        self.train = pd.read_csv(os.path.join(from_root(),"dataset",'train.csv'))
        self.stores = pd.read_csv(os.path.join(from_root(),"dataset",'stores.csv'))
        self.test = pd.read_csv(os.path.join(from_root(),"dataset",'test.csv'))


        feat_sto = self.features.merge(self.stores, how='inner', on='Store')
        feat_sto.Date = pd.to_datetime(feat_sto.Date)
        self.train.Date = pd.to_datetime(self.train.Date)
        self.test.Date = pd.to_datetime(self.test.Date) 

        feat_sto['Week'] = feat_sto.Date.dt.week 
        feat_sto['Year'] = feat_sto.Date.dt.year
        
        train_detail = self.train.merge(feat_sto, 
                           how='inner',
                           on=['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
        test_detail = self.test.merge(feat_sto, 
                           how='inner',
                           on=['Store','Date','IsHoliday']).sort_values(by=['Store',
                                                                            'Dept',
                                                                            'Date']).reset_index(drop=True)
        del self.features, self.train, self.stores, self.test

        null_columns = (train_detail.isnull().sum(axis = 0)/len(train_detail)).sort_values(ascending=False).index
        null_data = pd.concat([
            train_detail.isnull().sum(axis = 0),
            (train_detail.isnull().sum(axis = 0)/len(train_detail)).sort_values(ascending=False),
            train_detail.loc[:, train_detail.columns.isin(list(null_columns))].dtypes], axis=1)
        null_data = null_data.rename(columns={0: '# null', 
                                            1: '% null', 
                                            2: 'type'}).sort_values(ascending=False, by = '% null')
        null_data = null_data[null_data["# null"]!=0]

        weekly_sales_2010 = train_detail[train_detail.Year==2010]['Weekly_Sales'].groupby(train_detail['Week']).mean()
        weekly_sales_2011 = train_detail[train_detail.Year==2011]['Weekly_Sales'].groupby(train_detail['Week']).mean()
        weekly_sales_2012 = train_detail[train_detail.Year==2012]['Weekly_Sales'].groupby(train_detail['Week']).mean()

        plt.figure(figsize=(20,10))
        sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)
        sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)
        sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)
        plt.grid()
        plt.xticks(np.arange(1, 53, step=1))
        plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
        plt.title('Average Weekly Sales - Per Year', fontsize=18)
        plt.ylabel('Sales', fontsize=16)
        plt.xlabel('Week', fontsize=16)
        plt.show()

        train_detail.loc[(train_detail.Year==2010) & (train_detail.Week==13), 'IsHoliday'] = True
        train_detail.loc[(train_detail.Year==2011) & (train_detail.Week==16), 'IsHoliday'] = True
        train_detail.loc[(train_detail.Year==2012) & (train_detail.Week==14), 'IsHoliday'] = True
        test_detail.loc[(test_detail.Year==2013) & (test_detail.Week==13), 'IsHoliday'] = True

        weekly_sales_mean = train_detail['Weekly_Sales'].groupby(train_detail['Date']).mean()
        weekly_sales_median = train_detail['Weekly_Sales'].groupby(train_detail['Date']).median()

        plt.figure(figsize=(20,8))
        sns.lineplot(weekly_sales_mean.index, weekly_sales_mean.values)
        sns.lineplot(weekly_sales_median.index, weekly_sales_median.values)

        plt.grid()
        plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
        plt.title('Weekly Sales - Mean and Median', fontsize=18)
        plt.ylabel('Sales', fontsize=16)
        plt.xlabel('Date', fontsize=16)
        plt.show()

        weekly_sales = train_detail['Weekly_Sales'].groupby(train_detail['Store']).mean()
        plt.figure(figsize=(20,10))
        sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
        plt.grid()
        plt.title('Average Sales - per Store', fontsize=18)
        plt.ylabel('Sales', fontsize=16)
        plt.xlabel('Store', fontsize=16)
        plt.show()

        weekly_sales = train_detail['Weekly_Sales'].groupby(train_detail['Dept']).mean()
        plt.figure(figsize=(20,10))
        sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
        plt.grid()
        plt.title('Average Sales - per Dept', fontsize=18)
        plt.ylabel('Sales', fontsize=16)
        plt.xlabel('Dept', fontsize=16)
        plt.show()
        sns.set(style="white")

        corr = train_detail.corr()
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        f, ax = plt.subplots(figsize=(20, 20))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        plt.title('Correlation Matrix', fontsize=18)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        plt.show()

        return train_detail,test_detail

    def drop1(self,train,test):

        self.train_detail=train
        self.test_detail=test
        
        train_detail = self.train_detail.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
        test_detail = self.test_detail.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])

        train_detail.Type = self.train_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))
        test_detail.Type = self.test_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))   

        train_detail = self.train_detail.drop(columns=['Temperature'])
        test_detail = self.test_detail.drop(columns=['Temperature'])

        train_detail = self.train_detail.drop(columns=['Unemployment'])
        test_detail = self.test_detail.drop(columns=['Unemployment'])


        train_detail = self.train_detail.drop(columns=['CPI'])
        test_detail = self.test_detail.drop(columns=['CPI'])
        

        X_train = train_detail[['Store','Dept','IsHoliday','Size','Week','Type','Year']]
        Y_train = train_detail['Weekly_Sales']

        df=pd.get_dummies(X_train, prefix=['Type'])

        
        return df,Y_train

      

     
     

