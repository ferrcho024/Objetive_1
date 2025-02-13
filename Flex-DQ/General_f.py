import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wx
from IPython.display import display
from scipy.stats import anderson, shapiro
import pingouin as pg
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib import gridspec



def get_path():
    """Opens a file dialog box and asks the user to select a file, the path to the file is saved as string and returned.
    """
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Select the dataset file in *.csv format', wildcard='*.csv', style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    
    return path

def typedNoCorrect():
    '''Opens a dialog box asking if you want to correct the information typed'''
    
    app = wx.App(None)
    dlg = wx.MessageDialog(None,
                               'Typed was no correct'
                               '\nDo you want to correct it?','No correct input',
                               wx.YES_NO | wx.ICON_QUESTION)
    result = dlg.ShowModal()
    if result == wx.ID_YES:
        ans = True
    else:
        ans = False
    
    dlg.Destroy()

    return ans

def inputBox():
    '''Open a dialog box and asks the names of the columns where the data to be aveluated is
    Return the names written in a list'''

    app = wx.App(None)
    inputBox = wx.TextEntryDialog(None, 'Type the names of the variables you want to analyze' 
                                        '\nseparated by commas and without spaces'
                                        '\ni.e.: columnA,columnB,columnC,...'
                                        '\nor type * if you want to select all available columns', 'Input box')
    if inputBox.ShowModal() == wx.ID_OK:
        text = inputBox.GetValue()
        if text == '':
            inputBox.Destroy()
            return
        elif text == '*':
            list = ['*']
            inputBox.Destroy()
        else:
            list = text.split(',')
            inputBox.Destroy()

    else:
        return
    
    return list

def selectColumns(columnList):
    '''Opens a dialog box and asks for the names of the columns where the data to be evaluated are located, 
    if no columns were written, a correction question is asked
    Return the names written in a list'''

    while True:
        columns = inputBox()
        if not columns:
            if not typedNoCorrect():
                print('No column selected')
                print('Check the names of the available columns', columnList)
                return
        elif columns[0] == '*':
            columns = columnList
            break
        else:
            break
    
    return columns

def dataType():
    '''Evaluates all data types recognized by python'''
    for i in 'aAbBcCdDeEfFgGhHiIjJkKlLmMnNoOpPqQrRsStTuUvVwWxXyYzZ':
        try:
            print(i, 'es de tipo', np.dtype(i))

        except TypeError:
            pass

def checkDfType(df):

    if not 'DataFrame' in str(type(df)):
        print('No valid DataFrame entered')
        return False
    return True

def numeric(a, old,new):
    ''' Remplace a string to number format
    If there are more than 1 old string, only remplace the last one, the other are removed'''

    pos = str(a).count(old)
    cont = 0
    word = ''
    for l in str(a):
        if l == old:
            cont += 1
            if cont == pos:
                l = new
            else:
                l = ''
        word = word+l
    
    return word

def checkColType(values):

    values = values.dropna()
    
    if len(values) <  1:
        return (None,0)
    
    dropped = []
    dtype = {'Date':0,'Time':0,'Number':0, 'String':0}
    for d in values:
        if (str(d).count('/')) == 2:
            dtype['Date'] += 1
        if (str(d).count('-')) == 2:
            dtype['Date'] += 1
        elif (str(d).count(':')) == 2:
            dtype['Time'] += 1
        else:
            try:
                float(numeric(d,',','.'))
                dtype['Number'] += 1
            except:
                dropped.append(d)
                None
                
    dtype['String'] = values.count() - sum(dtype.values())
    #print(dtype, values.count())

    dataType = max(dtype, key=dtype.get)

    return (dataType, int(((dtype[dataType])/values.count())*100))#, dropped

def check_df(df=None):
    ''' Function to check dataset. Import dataset and Identify column names, data type of each feature and lenght of data
    Return the imported df as the firts variable and a dicctionary with name and data type of each column as a second variable.'''
    
    if not checkDfType(df):
        patch = get_path()
        try: 
            df = pd.read_csv(patch,sep=',',low_memory=False)

        except ValueError:
            print('No file selected')
            return df, df
    
    df.columns = df.columns.str.replace(' ', '')
    #dic = df.dtypes.to_dict()
    #Default dtypes are in numpy format, uncomment if you want types in string format
    #dic = {x: str(dic[x]).split("'")[0] for x in dic}

    # Prints information about the data type of the columns and the number of values in each column of the data set.
    buffer = io.StringIO()
    df.info(buf=buffer,max_cols=10000)
    lines = buffer.getvalue().splitlines()
    dff = (pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split())
        .drop(['#','Count'],axis=1)
        .rename(columns={'Column':'Columns','Non-Null':'Non-NaN Count'})
        )
    
    types = []
    for col in df.columns:
        #typeDetected = checkColType(df[col])
        types.append(checkColType(df[col]))

    dff['DtypeDetec'] = types
    #print(dff.to_string(index=False))
    display(dff)
    #print(dff.to_markdown())
    print('Total length:',len(df),'rows')
    #dff.style.hide_index()
    dff.style.hide(axis="index")

    return df, dff

def checkColumnsVar(df):

    global columns
    
    columnList = list(df)

    try:
        if columns:
            app = wx.App(None)
            dlg = wx.MessageDialog(None,
                                'Do you want to continue with the same columns?'
                                '\n'+str(columns)+'',
                                'Confirm same columns',
                                wx.YES_NO | wx.ICON_QUESTION)
            result = dlg.ShowModal()
            if result == wx.ID_YES:
                columns = columns
            else:
                columns = selectColumns(columnList)
            
            dlg.Destroy()
    
    except NameError:
        columns = selectColumns(columnList)

    if not columns:
        del columns
        return
    
    return columns


def basicStatistics(df):
    '''Function to calculate basic statistics for selected variables in a dataset. Calculate mean, median, standar desviation, variance and normality test, and show the min and max value.
    Returns a diccionary with the calculated values'''

    if not checkDfType(df):
        return

    columns = checkColumnsVar(df)

    if not columns:
        return
        
    dic = {}
    res = pd.DataFrame(columns=['Column', 'Unique-values','Min', 'Max', 'Mean','Median','Stand_desv','Variance'])
    nameError = []

    if len(df) < 1000:
        test = 'shapiro'
    else:
        test = 'jarque_bera'

    for c in columns:
        try:
            if df[c].dtype.char in 'bBdefghHiIlLpPqQ':
                
                data = np.array(df[c][df[c].notna()])
                #stats = pg.normality(data, method=test, alpha=0.05)
                normality = 'NaN'# str(stats.normal[0])

                # if test == 'Shap-wilk':
                #     if shapiro(data).pvalue >= 0.05:
                #         normality = 'YES'
                #     else:
                #         normality = 'NO'

                # else:
                #     An_res = anderson(data, dist='norm')
                #     if An_res.critical_values[2] >= An_res.statistic:
                #         normality = 'YES'
                #     else:
                #         normality = 'NO'


                dic[c] = {'Unique-values':df[c].nunique(dropna=True),
                        'Min':np.round(df[c].min(),2), 
                        'Max':np.round(df[c].max(),2),
                        'Mean':np.round(df[c].mean(),2), 
                        'Median':np.round(df[c].median(),2), 
                        'Stand_desv': np.round(df[c].std(),2), 
                        'Variance': np.round(df[c].var(),2),
                        'Norm('+test+')': normality}
                
                res = res.append({'Column':c,
                                    'Unique-values':df[c].nunique(dropna=True),
                                    'Min':np.round(df[c].min(),2), 
                                    'Max':np.round(df[c].max(),2),
                                    'Mean':np.round(df[c].mean(),2),
                                    'Median':np.round(df[c].median(),2),
                                    'Stand_desv':np.round(df[c].std(),2),
                                    'Variance':np.round(df[c].var(),2),
                                    'Norm('+test+')': normality}, ignore_index = True)
                
            else:
                print('\nData type of', c, 'is not calculable')
                nameError.append(c)
                dic[c] = {'Unique-values':df[c].nunique(dropna=True),
                        'Min':np.NaN, 
                        'Max':np.NaN,
                        'Mean':np.NaN, 
                        'Median':np.NaN, 
                        'Stand_desv': np.NaN, 
                        'Variance': np.NaN,
                        'Norm('+test+')': np.NaN}
                
                res = res.append({'Column':c,
                                    'Unique-values':df[c].nunique(dropna=True),
                                    'Min':np.NaN, 
                                    'Max':np.NaN,
                                    'Mean':np.NaN,
                                    'Median':np.NaN,
                                    'Stand_desv':np.NaN,
                                    'Variance':np.NaN,
                                    'Norm('+test+')': np.NaN}, ignore_index = True)


        except (KeyError, ValueError):
            nameError.append(c)
        
    res.set_index('Column', inplace=True)

    if not res.empty:
        #print('\n',res)
        display(res)

    if nameError:
        print('\n',nameError, 'is not a column of Dataset or you must check the data type')
        print('Check the names of the available columns and its type of data using the check_df function')
    
    columns = list(set(columns) - set(nameError))

    if len(columns) > 1:
        try:
            cormat = df[columns].corr()
            sns.heatmap(cormat, annot=True, fmt=".2f",)
        
        except KeyError:
            None
    else:
        print('\nNo correlation graph is generated because no more than 1 column is selected')

    return dic

def statsGraph(df):

    columns = checkColumnsVar(df)
   
    ncol = 2
    nrow = len(columns)

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3,nrow*3))
    fig.suptitle('Density plots                                   Box-plot')
    axes = axes.ravel()

    for col, ax in zip(columns, range(nrow)):
        sns.histplot(data=df[col], kde=True, ax=axes[ax*2])
        sns.boxplot(x=df[col], orient='h', showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"10"}, ax=axes[(ax*2)+1])

    fig.tight_layout()
    plt.show()



def paralelGraph(media, res, db1, db2, variable):

    fig = make_subplots(
        rows=1, cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.03,
        specs=[[{"type": "table"}, {"type": "scatter"}]]
    )

    fig.add_trace(go.Table(
        header=dict(values=[res.index.name] + list(res.columns)),
        cells=dict(values=[res.index, res[db1], res[db2]])),
        row=1, col=1
    )

    fig.add_trace(
    go.Scatter(x=media.index, y=media[db1], name=db1, mode='lines+markers',
                    marker={'color':'#3366cc','size': 12}), 
    row=1, col=2
    )

    fig.add_trace(
    go.Scatter(x=media.index, y=media[db2], name=db2, mode='lines+markers',
                    marker={'color':'#FF9900','size': 12}), 
    row=1, col=2
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        title='Mean of ' + variable + ' by '+ media.index.name +' for ' + db1 + ' vs '+ db2, 
        xaxis_title=media.index.name
    )

    fig.show()


def scatterMatploit(media, res, db1, variable, cont):

    fig = plt.figure(figsize=(12, 6), dpi=100)
    fig.suptitle('Mean of ' + variable + ' by '+ media.index.name +' for ' + db1, fontsize=16)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 3])

    cell_text = []
    for row in range(len(res)):
        cell_text.append(res.iloc[row])

    #ax0 = plt.subplot(3,1,1)
    ax0 = plt.subplot(gs[0])
    ax0.table(cellText=cell_text, 
                rowLabels=res.index,
                rowColours=['lightgray']*10,
                colLabels=res.columns,
                colColours=['lightgray']*5, 
                loc='center')
    ax0.axis('off')

    cell_text = []
    for row in range(len(cont)):
        cell_text.append(cont.iloc[row])

    #ax2 = plt.subplot(3,1,2)
    ax2 = plt.subplot(gs[1])
    ax2.table(cellText=cell_text, 
                rowLabels=cont.index,
                rowColours=['lightgray']*10,
                colLabels=cont.columns,
                colColours=['lightgray']*5, 
                loc='center')
    ax2.axis('off')

    # plot
    #ax1 = plt.subplot(3,1,3)
    ax1 = plt.subplot(gs[2])
    ax1.plot(media.T, 'o-')
    #ax1.bar(media.index, media[db2], 0.3, align='edge')
    #ax1.set_xlabel(media.index.name)
    ax1.legend(media.index)

    plt.show()

def barMatploit(media, res, db, variable, cont):

    fig = plt.figure(figsize=(12, 5), dpi=100)
    fig.suptitle('Mean of ' + variable + ' by '+ media.index.name +' for ' + db, fontsize=16)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax2 = plt.subplot()
    ax2.bar(cont.index, cont.iloc[:,0], -0.35, align='edge')
    ax2.bar(cont.index, cont.iloc[:,1], 0.35, align='edge')

    cell_text = []
    for row in range(len(res)):
        cell_text.append(res.iloc[row])

    ax0 = plt.subplot(gs[0])
    ax0.table(cellText=cell_text, 
                rowLabels=res.index,
                rowColours=['lightgray']*10,
                colLabels=res.columns,
                colColours=['lightgray']*5, 
                loc='center')
    ax0.axis('off')

    if media.index.name == 'Age':
        rot = 0
    if media.index.name == 'Education':
        rot = 20

    # plot
    ax1 = plt.subplot(gs[1])
    ax1.bar(media.index, media.iloc[:,0], -0.35, align='edge')
    ax1.bar(media.index, media.iloc[:,1], 0.35, align='edge')
    ax1.set_xticklabels(media.index, rotation=rot)
    ax1.legend(media.columns, loc='lower right')

    n = 0
    for p, q in zip(ax2.patches, ax1.patches):
        if n < len(media):
            ax1.annotate(str(round(p.get_height(),2)), (q.get_x() + q.get_width(), q.get_height() * 1.005),color='black')
            n += 1
        else:
            ax1.annotate(str(round(p.get_height(),2)), (q.get_x(), q.get_height() * 1.005),color='black')

    plt.show()

class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, id=-1, title="Data Quality Dimensions")
        panel = wx.Panel(self)
        vertical_sizer = wx.BoxSizer(wx.VERTICAL)
              
        self.txt1_head = wx.StaticText(panel, label='Choose a value between 0 and 5 for each dimension')
        self.txt2_head = wx.StaticText(panel, label='with 5 being the best value. Start by assigning 5 to the')
        self.txt3_head = wx.StaticText(panel, label='most important dimension and assign values to the other')
        self.txt4_head = wx.StaticText(panel, label='dimensions based on this maximum value.')
        vertical_sizer.Add(self.txt1_head, 0, wx.ALL | wx.RIGHT, 5)
        vertical_sizer.Add(self.txt2_head, 0, wx.ALL | wx.RIGHT, 5)
        vertical_sizer.Add(self.txt3_head, 0, wx.ALL | wx.RIGHT, 5)
        vertical_sizer.Add(self.txt4_head, 0, wx.ALL | wx.RIGHT, 5)

        self.chosen = ['Consistency', 
                       'Diversity', 
                       'Completeness', 
                       'Duplicity', 
                       'Volume', 
                       'Precision', 
                       'Outliers',
                       'Uncertainty']
        
        self.dims = {dim:wx.TextCtrl(panel) for dim in self.chosen}

        for dim in self.chosen:
            self.txt = wx.StaticText(panel, label=dim)
            horizontal_sizer = wx.BoxSizer(wx.HORIZONTAL)
            horizontal_sizer.Add(self.txt, 0, wx.ALL | wx.RIGHT, 5)
            horizontal_sizer.Add(self.dims[dim], 0, wx.ALL | wx.RIGHT, 5)
            vertical_sizer.Add(horizontal_sizer, flag=wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=10)

        my_btn = wx.Button(panel, label='Ok')       
        vertical_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 5)
        panel.SetSizer(vertical_sizer)
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)

        self.Show()
    
    def on_press(self, event):
        ban = False
        # while ban:
        try:
            scores = []
            for dim in self.chosen:
                scores.append(int(self.dims[dim].GetValue()))

            if any(value > 5 for value in scores):
                self.ShowMessage('Some values are out of the range [0-5]')

            elif any(value < 0 for value in scores):
                 self.ShowMessage('Some values are out of the range [0-5]')
                
            else:
                self.dimensions = {dim:val for dim,val in zip(self.chosen, scores)}

                del self.dims, self.chosen
                ban = True                  

        except:
            self.ShowMessage('Some of the values are not an integer')

        if ban:
            self.Close()
    
    def ShowMessage(self, txt):
        wx.MessageBox(txt, 'Info', 
            wx.OK | wx.ICON_INFORMATION)


class MyDataBase(wx.Frame):
            
    def __init__(self, parent= None, title='DataBase Type'): 
      super(MyDataBase, self).__init__(parent, title = title,size = (300,260)) 
         
      self.InitUI()
		
    def InitUI(self):    
        pnl = wx.Panel(self)
        vertical_sizer = wx.BoxSizer(wx.VERTICAL)

        self.txt1_head = wx.StaticText(pnl, label='What type of database do you want to analyze?')
        vertical_sizer.Add(self.txt1_head, 0, wx.ALL | wx.RIGHT, 5)
            
        self.choice = '1'
        self.ban = False
        self.rb1 = wx.RadioButton(pnl,11, label = 'Time Series databases', pos = (10,30), name='1', style = wx.RB_GROUP) 
        self.rb2 = wx.RadioButton(pnl,22, label = 'Survey Results', pos = (10,60), name='2') 
        self.rb3 = wx.RadioButton(pnl,33, label = 'On-demand Monitoring', pos = (10,90), name='3') 
        self.Bind(wx.EVT_RADIOBUTTON, self.OnRadiogroup)

        my_btn = wx.Button(pnl, label='Ok')       
        vertical_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 85)
        pnl.SetSizer(vertical_sizer)
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)
                    
        self.Centre() 
        self.Show(True)    
	
    def OnRadiogroup(self, e): 
      rb = e.GetEventObject()
      self.choice =  rb.GetName() 
    
    def on_press(self, event):
        if self.choice == '1':
            self.dimensions = ['Timeliness', 'Conformity', 'Uniqueness', 'Accuracy', 'Completeness']
        elif self.choice == '2':
            self.dimensions = ['Concordance', 'Conformity', 'Uniqueness']
        else:
            self.dimensions = ['Redundancy']
        self.ban = True

        self.Close()


class MyDims(wx.Frame):
            
    def __init__(self, parent= None, title='Questions'): 
      super(MyDims, self).__init__(parent, title = title,size = (600,650)) 
         
      self.InitUI()
		
    def InitUI(self):    
        pnl = wx.Panel(self)
        vertical_sizer = wx.BoxSizer(wx.VERTICAL)

        self.txt1_head = wx.StaticText(pnl, label='Response the questions')
        vertical_sizer.Add(self.txt1_head, 0, wx.ALL | wx.RIGHT, 5)

        ans = ['YES', 'NO']
        self.answers = ['YES']*9
        self.ban = False 
        self.rbox1 = wx.RadioBox(pnl, label = 'Do you have reference data that allows other data in the dataset to be validated?', pos = (10,30), choices = ans,
            majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 
        #self.rbox1.Bind(wx.EVT_RADIOBOX,self.onRadioBox_Q1)

        self.rbox2 = wx.RadioBox(pnl, label = 'Does the data set include data obtained at the same time from diff erent sources for the same variable?', pos = (10,90), choices = ans,
            majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 
        #self.rbox2.Bind(wx.EVT_RADIOBOX,self.onRadioBox_Q2)

        self.rbox3 = wx.RadioBox(pnl, label = 'Has the data been modifi ed or preprocessed in any way?', pos = (10,150), choices = ans,
            majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 
        #self.rbox3.Bind(wx.EVT_RADIOBOX,self.onRadioBox_Q3)

        self.rbox4 = wx.RadioBox(pnl, label = 'Do the data refl ect the variable’s behavior over time?', pos = (10,210), choices = ans,
            majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 
        
        self.rbox5 = wx.RadioBox(pnl, label = 'Are repeated records common in dataset?', pos = (10,270), choices = ans,
            majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 
        
        self.rbox6 = wx.RadioBox(pnl, label = 'Are there any categorical variables in the dataset?', pos = (10,330), choices = ans,
            majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 
        
        self.rbox7 = wx.RadioBox(pnl, label = 'Does the dataset use any abbreviations or symbols to represent other values?', pos = (10,390), choices = ans,
            majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 
        
        self.rbox8 = wx.RadioBox(pnl, label = 'Is the data expected to exhibit clustering behavior or to be closely spaced?', pos = (10,450), choices = ans,
            majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 
        
        self.rbox9 = wx.RadioBox(pnl, label = 'Does the intended use of the dataset require the presence of certain percentage of all records?', pos = (10,510), choices = ans,
            majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 

        my_btn = wx.Button(pnl, label='Ok')       
        vertical_sizer.Add(my_btn, 1, wx.FR_DOWN | wx.UP, 550)
        pnl.SetSizer(vertical_sizer)
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)
                    
        self.Centre(True) 
        self.Show(True)

    
    def on_press(self, event):
        self.answers[0] = self.rbox1.GetStringSelection()
        self.answers[1] = self.rbox2.GetStringSelection()
        self.answers[2] = self.rbox3.GetStringSelection()
        self.answers[3] = self.rbox4.GetStringSelection()
        self.answers[4] = self.rbox5.GetStringSelection()
        self.answers[5] = self.rbox6.GetStringSelection()
        self.answers[6] = self.rbox7.GetStringSelection()
        self.answers[7] = self.rbox8.GetStringSelection()
        self.answers[8] = self.rbox9.GetStringSelection()

        self.ban = True

        self.Close()