import pandas as pd
import numpy as np

class DataProcessor:

    def __init__(self, beginTime, endTime, data, name, max_iter=1000):
        self.beginTime = beginTime
        self.endTime = endTime
        self.data = data
        self.name = name
        self.max_iter = max_iter

        self.cleanData()

    
    def cleanData(self):
        data = self.data

        # if self.name == '2015-07.csv':        dit wordt nu gedaan door TransformData()
        #    data = data.drop(data.index[0])

        if self.name == '2015-07.csv' or self.name == '2024-02.csv':
            data['sasdate'] = pd.to_datetime(data['sasdate'])
            data.set_index('sasdate', inplace=True)

            data_cleaned = data.dropna()
            data_cleaned.reset_index(inplace=True)

            self.data = data_cleaned
            self.TransformData()

        return self.data


    def CreateDataSet(self, dependentVariable, beginTime=None, endTime=None):

        beginTime = beginTime if beginTime is not None else self.beginTime
        endTime = endTime if endTime is not None else self.endTime

        data_cleaned_train = self.data[(self.data['sasdate'] < endTime) & (self.data['sasdate'] >= beginTime)]

        x_train = data_cleaned_train.drop(columns=[dependentVariable, 'sasdate'])
        y_train = data_cleaned_train[dependentVariable]

        extraMonth = endTime + pd.DateOffset(months=1)

        data_cleaned_test = self.data[(self.data['sasdate'] < extraMonth) & (self.data['sasdate'] >= endTime)]
        x_test = data_cleaned_test.drop(columns=[dependentVariable, 'sasdate'])
        y_test = data_cleaned_test[dependentVariable].values[0]

        return [x_train, y_train, x_test, y_test]

    def TransformData(self): # missing values are not taken into account
        data = self.data
        
        for column in data 
            if column != 'sasdate': # skip 'sasdate' column
                transformation = data[column].iloc[0] # check transformation code
                data[column].drop(data.index[0]) # drop transformation code from the column
                if transformation == 2: 
                    # first differences
                    data[column] = data[column].diff()
                elif transformation == 3: 
                    # second differences
                    data[column] = (data[column].diff()).diff()
                elif transformation == 4: 
                    # logarithm
                    data[column] = np.log(data[column])
                elif transformation == 5: 
                    # log first differences
                    data[column] = (np.log(data[column])).diff()
                elif transformation == 6: 
                    # log second differences
                    data[column] = ((np.log(data[column])).diff()).diff()
                elif transformation == 7: # i'm not sure about this , chatgpt also talked about pct_change()
                    # Compute the first part of the transformation: ((xt/xt-1) - 1)
                    part1 = (data[column] / data[column].shift(1)) - 1
                    # Compute the second part of the transformation: ((xt-1/xt-2) - 1)
                    part2 = (data[column].shift(1) / data[column].shift(2)) - 1
                    # Combine the two parts to get the final transformation
                    data[column] = part1 - part2
                else: 
                    # no transformation
                    data[column] = data[column]
                
        self.data = data 

        return self.data



        
        
    
