import pandas as pd
pd.set_option("display.precision", 2)
import numpy as np
from IPython.display import display
import General_f as Gf



def uniqueness(data, clean=False):
    '''evenness/uniformity/Diversity  -> https://www.statology.org/shannon-diversity-index/
    https://ecopy.readthedocs.io/en/latest/diversity.html (gini-simpson)
    Check how much the variable changes.
    Return a  diccionary with the diversity of each variable of the dataframe.
    
    If clean is True: Return a list of variables with no changes and a df without this variables.'''

    df = data.copy()
    delete = np.NaN
    
    diversity = {}
    for x in df.columns:
        diversity[x] = round((df[x].nunique(dropna=True)/len(df)),2)
    
    if clean:
        delete = []
        [[df.drop(x, axis=1, inplace=True), delete.append(x)] for x in df.columns if (df[x].nunique(dropna=True) <=1)]

        #return round(sum(diversity.values())/len(diversity),2), delete, df

    
    return round(sum(diversity.values())/len(diversity),2), delete, df#, diversity
    
def redundancy(data, clean=False):
    '''Checks for duplicate rows in a dataframe
    Return the propotyions of duplicated
    duplicity: A   measure   of   unwanted   duplicity   existing within  or  across  systems  for  a  particular  field, record, or data set

    If clean is True: Return a list of the index of duplicated rows and a df without the duplicated rows''' 

    df = data.copy()
    duplicated = np.NaN

    if clean:
        #df = data.copy()
        duplicated = list(df[df.duplicated()].index)
        df.drop_duplicates(inplace=True)

        #return round((1 - data.duplicated().sum()/len(data)),2), duplicated, df    
    
    return round((1 - df.duplicated().sum()/len(df)),2), duplicated, df


def consistency(data, clean=False):
    '''Checks the Consistency of each variable according to data type recognized for pandas and the format which the values have
    Return the proportion of variables with total Consistency
    Consistency: The  extent  to  which data is  presented  in  the same  format  and  compatible  with  previous  data
    Consistency: Refer  to  the  violation  of  semantic  rules defined over the set of data

    If clean is True: Return a list of the variable's names with 100% of legibility rows and a df without the unlegibility variables'''

    dff = pd.DataFrame()
    types = []
    cont = 0
    for col in data.columns:
        #typeDetected = checkColType(df[col])
        res = Gf.checkColType(data[col])
        types.append(res)
        if res[1] < 100:
            cont += 1

    dff['Columns'] = data.columns
    dff['DtypeDetec'] = types
    
    df = data.copy()
    noLeg = np.NaN
    
    if clean:
        cols = []
        for i, index in zip(dff['DtypeDetec'], dff.index):
            if i[1] == 100:
                col = dff.loc[index,'Columns']
                if i[0] == 'Number':
                    df[col] = pd.to_numeric(df[col].apply(Gf.numeric, args=[',','.']), errors='ignore')
                if i[0] == 'Date':
                    df[col] = pd.to_datetime(df[col])
                if i[0] == 'Time':
                    df[col] = pd.to_datetime(df[col]).dt.time

                cols.append(col)
        df = df[cols]
        noLeg = [x for x in list(data.columns) if x not in cols]  
        
        #return round(1-(cont/len(data.columns)),2), noLeg, df
        
    return round(1-(cont/len(df.columns)),2), noLeg, df

def completeness(data, clean=False):
    '''Checks if the values in the dataset are complete
    Return the percentage of completeness od the dataset

    If clean is True: Return a df without the columns and rows wich have less than 20% of completeness'''

    df = data.copy()

    if clean:
        #df = data.copy()
        cols = list(df.columns[(df.isnull().sum(axis=0)/len(df)) > 0.2])
        rows = list(df.index[(df.isnull().sum(axis=1)/len(df.columns)) > 0.2])

        df.drop(cols, axis=1, inplace=True)
        df.drop(rows, axis=0, inplace=True)
        
        #return round(1 - (data.isnull().sum(axis=0)/len(data)).mean(),2), df
    
    return round(1 - (df.isnull().sum(axis=0)/len(df)).mean(), 2), df

def volume(data, clean=False):
    ''' To  extent  to  which  the  quantity  of available data is appropriate
    Return the percentage of the quantity of data required is available in the dataset
    Among of data must be 10 times according to the rule-of-thumb approach
    
    "The rule of thumb regarding machine learning is that you need at least ten times 
    as many rows (data points) as there are features (columns) in your dataset. This 
    means that if your dataset has 10 columns (i.e., features), you should have at 
    least 100 rows for optimal results."
    https://graphite-note.com/how-much-data-is-needed-for-machine-learning
    
    If clean is True: Return a df without the columns wich have less than 50% of no NaN values'''

    df = data.copy()

    if clean:
        df = df.dropna(axis=1,thresh=len(df)/2)

    #cols = len((df.dropna().reset_index(drop=True)).columns)
    df = df.dropna().reset_index(drop=True)
    cols = len(df.columns)
    expected = cols*10

    if clean:
        return round(min(len(df)/expected, 1), 2), df
    else:
        return round(min(len(df)/expected, 1), 2), data

def precision(data):
    '''Calculates the precision of the dataset according to de dispersion of the data
    Return the precision where 0 is the worse value and 1 the best one'''
    
    df = data.copy()

    res = 1 - abs((df.std(numeric_only=True)/df.mean(numeric_only=True)))
    #print(res)
    res[res<0] = 0
    #print(res)

    return round(res.mean(), 2), df


def uncertainty(data, ref=False):
    ''' Calculates the uncertainty of the data according to the number of diferent values
    Return '''

    
    error = False
    df = data.copy()
    df.reset_index(drop=True, inplace=True)
    if ref:
        if isinstance(ref, tuple):
            res = []
            for c in df.columns:
                try:
                    res.append(df[c].between(min(ref),max(ref)).sum()/len(df))
                except:
                    None
            
            uncer = np.mean(res)

        elif isinstance(ref, pd.Series):
            try:
                if len(df) == len(ref):
                    res = []
                    for c in df.columns:
                        #res = pd.concat([df[c], ref], axis=1)

                        np.sqrt(((df[c] - ref).pow(2).sum())/(2*(df[c] + ref).count()*((df[c] + ref).mean())**2))
                        uncer = round(1-uncer,2)
                        uncer = max(0,uncer)

                        
                        #rmse = mean_squared_error(df[c], ref)
                        rmse = np.sqrt(((df[c] - ref).pow(2).sum())/len(df[c]-ref))
                        rmse = round(rmse,2)
                        
                        print('The uncertainty value is:',uncer)
                        print('The RMSE is:', rmse)
                    
                        res[res[df[c].name].notna()].plot()

                        return uncer
                
                else:
                    error = True
                    #print('Uncertainty: Something has gone wrong with the data. Check if the dimensions are the same and both are dataset or data series')
                    #return

            except:
                error = True
                #print('Uncertainty: Something has gone wrong with the data. Check if the dimensions are the same and both are dataset or data series')
                #return
            else:
                error = True
                #print('Uncertainty: Something has gone wrong with the data. Check if the dimensions are the same and both are dataset or data series')
                #return

        if error:
                print('Uncertainty: Something has gone wrong with the data. Check if the dimensions are the same and both are dataset or data series')
                return


    else:
        '''Tomado de: https://www.webassign.net/bohphysvl1/uncertainty.pdf

        number = 30  # Value
        des = 15 # absolute_uncer, tambien llamado standar uncertainty

        relative_uncer = (1 - (des/number)) 

        absolute_uncer = int((relative_uncer/100)*number) # Where 6 is the measured value which wants to know the absolute uncertainty

        relative_uncer, absolute_uncer'''

        res = 1 - (df.std(numeric_only=True)/df.nunique())
        res[res<0] = 0
    
        return round(res.mean(), 2), df




def outliers(data, clean=False):
    '''Claculate the percentage of possible outliers according to an threshold calculated by Q3 + 1.5*(Q3-Q1), 
    where Q1 and Q3 are the quartiles 0.25 and 0.75 of the data
    Return the percentage of outliers in the dataset according to the number of possible outliers for each variable
     
    If clean is True: Return a df without the values identified like possible outliers. Values are changed for NaN values.'''
    
    df = data.copy()
    outliers = pd.DataFrame(columns=['Column','Outliers'])
    for c in data.columns:
        try:
            Q1, Q3 = data[c].quantile([0.25, 0.75])
            threshold = Q3 + 1.5*(Q3-Q1)
            res = (data[c] > threshold).sum()/data[c].count()
            

            if res > 0.0:             
                outliers = pd.concat([outliers, pd.DataFrame({'Column': c,
                                                                'Outliers': res}, index=[0])], axis=0, ignore_index=True)

                if clean:
                    df[c] = df[c].mask(df[c] > threshold)
        except:
            None

    return round(1 - outliers['Outliers'].mean(), 2), df


def accuracy(data, ref):
    '''Calculates the accuracy of a feature according to another reference feature'''

    df = data.copy()

    try: 
        if len(df) == len(ref):
            res = pd.concat([df, ref], axis=1)
            res['cero'] = 0

            res['acc'] = round(1-(abs(df - ref)/ref),2)
            res['acc'] = res[['cero','acc']].max(axis=1, skipna=False)
            res.drop('cero', axis=1, inplace=True)

            mean = round(res['acc'].mean(),2)

            print('The accuracy value is:',mean)

            if not res.empty:
                #print('\n',res.to_markdown())
                display(res)

                return res, df
            
            return mean, df      

        else:
            print('Something has gone wrong with the data. Check if the dimensions are the same and both are dataset or data series')
            return

    except:
        print('Something has gone wrong with the data. Check if the dimensions are the same and both are dataset or data series')
        return


def concordance(data, ref):
    '''Calculates the concordance between 2 features, according to the number of different results for each
    value. If each different value has only one result, indicates a 100% of concordance'''

    df = data.copy
    try:
        if len(df) == len(ref):
            res = pd.concat([df, ref], axis=1)
            cols = res.columns
            res = res.groupby([cols[0]])[cols[1]].describe()[['min','max']]
            concor = len(res.loc[res['min'] == res['max']])/len(res)

            concor = round(concor,2)

            print('The concordance value is:',concor)
            
            return concor, df
        
        else:
            print('Something has gone wrong with the data. Check if the dimensions are the same and both are dataset or data series')
            return
    
    except:
        print('Something has gone wrong with the data. Check if the dimensions are the same and both are dataset or data series')
        return