import pandas as pd
import numpy as np

def data_to_supervised_1(data, n):
    df = pd.DataFrame(data, columns = ['var1(t)', 'var2(t)', 'var3(t)','var4(t)','samp(t)'])
    df1 = df.iloc[:,:-1]
    df2 = df.iloc[:,-1]
    new = df1
    if n == 0:
        return df
    else:
        for i in range(1,n+1):
            shift_front = df1.shift(i)
            shift_behind = df1.shift(i-1)
            diff = shift_behind - shift_front
            if i>1:
                diff.columns = ['var1(t-'+str(i-1)+')-var1(t-'+str(i)+')',\
                                'var2(t-'+str(i-1)+')-var2(t-'+str(i)+')',\
                                'var3(t-'+str(i-1)+')-var3(t-'+str(i)+')',\
                                'var4(t-'+str(i-1)+')-var4(t-'+str(i)+')']
            else:
                diff.columns = ['var1(t)-var1(t-' + str(i) + ')', \
                                'var2(t)-var2(t-' + str(i) + ')', \
                                'var3(t)-var3(t-' + str(i) + ')', \
                                'var4(t)-var4(t-' + str(i) + ')']
            new = pd.concat([diff, new], axis=1)
        new = pd.concat([new, df2], axis=1)
        return new[i:]

def data_to_supervised_2(path, n):
    data = pd.read_csv(path, header=0, index_col=0)
    data.columns = ['var1(t)', 'var2(t)', 'var3(t)','var4(t)']
    new = data
    for i in range(1,n+1):
        shift_front = data.shift(i)
        shift_behind = data.shift(i-1)
        diff = shift_behind - shift_front
        diff.columns = ['var1(t-'+str(i-1)+')-var1(t-'+str(i)+')',\
                        'var2(t-'+str(i-1)+')-var2(t-'+str(i)+')',\
                        'var3(t-'+str(i-1)+')-var3(t-'+str(i)+')',\
                        'var4(t-'+str(i-1)+')-var4(t-'+str(i)+')']
        new = pd.concat([diff, new], axis=1)
    return new[i:]

def data_to_supervised_3(data, n):
    df = pd.DataFrame(data, columns = ['var1(t)', 'var2(t)', 'var3(t)'])
    df1 = df.iloc[:,:-1]
    df2 = df.iloc[:,-1]
    new = df1
    if n == 0:
        return df
    else:
        for i in range(1,n+1):
            shift_front = df1.shift(i)
            shift_behind = df1.shift(i-1)
            diff = shift_behind - shift_front
            if i>1:
                diff.columns = ['var1(t-'+str(i-1)+')-var1(t-'+str(i)+')',\
                                'var2(t-'+str(i-1)+')-var2(t-'+str(i)+')']

            else:
                diff.columns = ['var1(t)-var1(t-' + str(i) + ')', \
                                'var2(t)-var2(t-' + str(i) + ')']

            new = pd.concat([diff, new], axis=1)
        new = pd.concat([new, df2], axis=1)
        return new[i:]

