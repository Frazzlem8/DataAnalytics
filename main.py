import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
from statsmodels.stats import weightstats as stests
import glob

def Clean_Merge():
    #This code merges the files csv files into one
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics/SingleFileCrimes')

    extension = 'csv'

    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv.to_csv("AllCrimes2014-2015.csv", index=False, encoding='utf-8-sig')

# Clean_Merge()


def mean():
    #code for calculating the mean
    import os
    os.chdir ('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv('AllCrimes2014-2015.csv')
    print("Mean: ")
    print(np.mean(df))

#mean()

def median():
    # code for calculating the median
    os.chdir ('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv('AllCrimes2014-2015.csv')
    print("median: ")
    print(df.median())

#median()

def mode():
    # code for calculating the mode
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv('AllCrimes2014-2015.csv')

    print("Mode: ")
    print("Number of Homicide convictions")
    print(df['Homicide'].mode())
    print("Offences Against The Person Convictions")
    print(df['Offences Against The Person'].mode())
    print("Number of Sexual Offences Convictions")
    print(df['Sexual Offences'].mode())
    print("Number of Burglary Convictions")
    print(df['Burglary'].mode())
    print("Number of Robbery Convictions")
    print(df['Robbery'].mode())
    print("Number of Theft And Handling Convictions")
    print(df['Theft And Handling'].mode())
    print("Number of Fraud And Forgery Convictions")
    print(df['Fraud And Forgery '].mode())
    print("Number of Criminal Damage Convictions")
    print(df['Criminal Damage'].mode())
    print("Number of Drugs Offences Convictions")
    print(df['Drugs Offences'].mode())
    print("Number of Public Order Offences Convictions")
    print(df['Public Order Offences'].mode())
    print("Number of All Other Offences (excluding Motoring) Convictions")
    print(df['All Other Offences (excluding Motoring)'].mode())
    print("Number of Motoring Offences Convictions")
    print(df['Motoring Offences '].mode())

#mode()


def correlation_matrix():
    #This code displays the correlation within a matrix
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv('AllCrimes2014-2015.csv')

    df.head(2)

    corr = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="Greens", annot=True)
    c1 = corr.abs().unstack()
    c1.sort_values(ascending=False)
    corr[corr < 1].unstack().transpose() \
        .sort_values(ascending=False) \
        .drop_duplicates()

    plt.show()

#correlation_matrix()

def everything():
    # This code contains everything including standard deviation, variance etc
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv(r'AllCrimes2014-2015.csv')


    mean1 = df['Criminal Damage'].mean()
    mean2 = df['Offences Against The Person'].mean()
    sum1 = df['Criminal Damage'].sum()
    sum2 = df['Offences Against The Person'].sum()
    max1 = df['Criminal Damage'].max()
    min1 = df['Criminal Damage'].min()
    count1 = df['Criminal Damage'].count()
    median1 = df['Criminal Damage'].median()
    std1 = df['Criminal Damage'].std()
    std2 = df['Offences Against The Person'].std()
    var1 = df['Criminal Damage'].var()
    var2 = df['Offences Against The Person'].var()

    # groupby_sum1 = df.groupby(['counties']).sum()
    # groupby_count1 = df.groupby(['counties']).count()


    print('Mean Criminal Damage : ' + str(mean1))
    print('Mean Offences Against The Person : ' + str(mean2))
    print('Sum of Criminal Damage : ' + str(sum1))
    print('Sum of Offences Against The Person : ' + str(sum2))
    print('Max Criminal Damage : ' + str(max1))
    print('Min Criminal Damage : ' + str(min1))
    print('Count of Criminal Damage : ' + str(count1))
    print('Median Criminal Damage : ' + str(median1))
    print('Standard deviation of Criminal Damage : ' + str(std1))
    print('Standard deviation of Offences Against The Person : ' + str(std2))
    print('Variance of Criminal Damage : ' + str(var1))
    print('Variance of Offences Against The Person : ' + str(var2))
    # print block 2
    #print('Sum of values, grouped by the county: ' + str(groupby_sum1))
    #print('Count of values, grouped by the county: ' + str(groupby_count1))



#everything()


def Random_Dataset():
    #This code is used to retrieve the data the variances of criminal damage and offences ATP
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv(r'AllCrimes2014-2015.csv')

    var1 = df['Criminal Damage'].var()
    var2 = df['Offences Against The Person'].var()

    print('Variance of Criminal Damage : ' + str(var1))
    print('Variance of Offences Against The Person : ' + str(var2))

#Random_Dataset()

def scatterplot_corrolation():

    # This code generates a scatterplot portraying correlation between two variables x and y
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    data = pd.read_csv(r'AllCrimes2014-2015.csv')

    x = data['Offences Against The Person']
    y = data['Criminal Damage']

    print(np.corrcoef(x, y))

    plt.scatter(x, y)
    plt.title('A plot to show the correlation between Offences Against The Person and Criminal Damage')
    plt.xlabel('Offences Against The Person ')
    plt.ylabel('Criminal Damage')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='yellow')
    plt.show()

#scatterplot_corrolation()


def ztestsingle():
    #This code is for z testing, testing to either reject or accept the null hypothesesis
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv(r'Random.csv')

    from statsmodels.stats import weightstats as stests

    ztest, pval1 = stests.ztest(df['Offences Against The Person'], x2=df['Criminal Damage'], value=0,
                                alternative='two-sided')
    print(float(pval1))
    if pval1 < 0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")


#ztestsingle()

def heatmap():
    #code for a heatmap
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    data = pd.read_csv(r'AllCrimes2014-2015.csv')
    data.columns = ["V" + str(i) for i in range(1, len(data.columns) + 1)]

    sns.heatmap(data.corr(), cmap="RdYlGn", linewidths=0.30)
    plt.show()



#heatmap()

def meanformultiplecolumns():
    #This code is for calculating the mean for multiple columns
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    data = pd.read_csv(r'AllCrimes2014-2015.csv')

    cols = ['Criminal Damage', 'Offences Against The Person']
    mean1 = data[cols].mean()

    print(mean1)

#meanformultiplecolumns()



def ztable_2variables():
    #This code displays a bell curve represented by the z-tests p values
    x = []
    y = []
    i = -4
    while i <4:
        i += 0.01
        x.append(round(i, 2))

    # Criminal Damage
    CD = 251
    # offences against the person
    OATP = 1392
    # Constant
    c = 1 / ((2 * OATP) ** 0.5)
    for a in x:
        # exponent
        expo = (-a ** 2) / 2
        # distrobution formula
        distro = c * (CD ** expo)
        y.append(distro)

    plt.figure('Bell Curve')
    plt.title('Standard Normal Distribution')
    plt.ylabel('Probability Density')
    plt.xlabel('Z-Score')
    plt.plot(x, y)
    plt.show()



#ztable_2variables()

os.chdir('C:/Users/Fraser/Documents/DataAnalytics')

with open('AllCrimes2014-2015.csv','r') as f:
    g=f.readlines()
    # Each line is split based on commas, and the list of floats are formed
    criminal_damage = [int(x.split(',')[2]) for x in g[1:]]
    offences_atp = [int(x.split(',')[8]) for x in g[1:]]


def covariance(x, y):
    # Finding the mean of the series x and y
    mean_x = sum(x)/int(len(x))
    mean_y = sum(y)/int(len(y))
    # Subtracting mean from the individual elements
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    numerator = sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))])
    denominator = len(x)-1
    cov = numerator/denominator
    return cov

with open('AllCrimes2014-2015.csv', 'r') as f:
    ...
    cov_func = covariance(criminal_damage, offences_atp)
    print("Covariance between criminal damage and offences atp:", cov_func)


#covariance()


def random_row_generator():
    #This code runs a random row generator that grabs a random 1 rows which can then be analysed
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv('AllCrimes2014-2015.csv')
    show = df.sample(n=10)
    with open("new_file2.csv", "w") as sink:
        sink.write("\n".join(show))
    print(show)

#random_row_generator()


def Multiple_Linear_Regression():
    # Code for running multiple linear regression model

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    data = pd.read_csv('AllCrimes2014-2015.csv', index_col=0)

    feature_cols = ['Criminal Damage', 'Offences Against The Person', 'Burglary']

    X = data[feature_cols]
    Y = data.Robbery

    lr_model = LinearRegression()

    lr_model.fit(X, Y)

    X_test_new = pd.DataFrame({'Criminal Damage': [10], 'Offences Against The Person': [10], 'Burglary': [10]})

    prediction = lr_model.predict(X_test_new)

    print(prediction)

#Multiple_Linear_Regression()


def linear_regression():
    #two linear regression model
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv('AllCrimes2014-2015.csv')

    x = df['Offences Against The Person']
    y = df['Criminal Damage']

    linreg = LinearRegression()
    x = x.values.reshape(-1, 1)
    linreg.fit(x,y)
    #labeling graph and prediction function
    y_pred = linreg.predict(x)
    plt.scatter(x,y)
    plt.plot(x, y_pred, color='red')
    plt.title('Linear Regression')
    plt.ylabel('Criminal Damage')
    plt.xlabel('Offences ATP')
    print(linreg.coef_)
    print(linreg.intercept_)
    plt.show()

    feature_cols = ['Offences Against The Person']

    X = df[feature_cols]
    Y = df['Criminal Damage']

    lr_model = LinearRegression()

    lr_model.fit(X, Y)

    X_test_new = pd.DataFrame({'Offences Against The Person': [50]})

    prediction = lr_model.predict(X_test_new)

    print(prediction)

#linear_regression()

import numpy as np
import matplotlib.pyplot as plt


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x = Offences ATP')
    plt.ylabel('y = Criminal Damage')

    # function to show plot
    plt.show()


def main():
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    df = pd.read_csv('AllCrimes2014-2015.csv')
    # observations / data
    x = df['Offences Against The Person']
    y = df['Criminal Damage']

    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)


#main()




def clustering():
    import pandas as pd
    import numpy as np
    import random as rd
    import matplotlib.pyplot as plt
    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    data = pd.read_csv('AllCrimes2014-2015.csv')
    data.head()

    X = data[["Offences Against The Person", "Criminal Damage"]]
    # Visualise data points
    plt.scatter(X["Criminal Damage"], X["Offences Against The Person"], c='black')
    plt.xlabel('Crimes')
    plt.ylabel('Criminal Damage + Offences ATP')

    K = 3

    # Select random observation as centroids
    Centroids = (X.sample(n=K))
    plt.scatter(X["Criminal Damage"], X["Offences Against The Person"], c='black')
    plt.scatter(Centroids["Criminal Damage"], Centroids["Offences Against The Person"], c='red')
    plt.xlabel('Crimes')
    plt.ylabel('Criminal Damage + Offences ATP')

    diff = 1
    j = 0

    while (diff != 0):
        XD = X
        i = 1
        for index1, row_c in Centroids.iterrows():
            ED = []
            for index2, row_d in XD.iterrows():
                d1 = (row_c["Criminal Damage"] - row_d["Criminal Damage"]) ** 2
                d2 = (row_c["Offences Against The Person"] - row_d["Offences Against The Person"]) ** 2
                d = np.sqrt(d1 + d2)
                ED.append(d)
            X[i] = ED
            i = i + 1

        C = []
        for index, row in X.iterrows():
            min_dist = row[1]
            pos = 1
            for i in range(K):
                if row[i + 1] < min_dist:
                    min_dist = row[i + 1]
                    pos = i + 1
            C.append(pos)
        X["Cluster"] = C
        Centroids_new = X.groupby(["Cluster"]).mean()[["Offences Against The Person", "Criminal Damage"]]
        if j == 0:
            diff = 1
            j = j + 1
        else:
            diff = (Centroids_new['Offences Against The Person'] - Centroids['Offences Against The Person']).sum() + (
                        Centroids_new['Criminal Damage'] - Centroids['Criminal Damage']).sum()
            print(diff.sum())
        Centroids = X.groupby(["Cluster"]).mean()[["Offences Against The Person", "Criminal Damage"]]

    color = ['blue', 'green', 'cyan']
    for k in range(K):
        data = X[X["Cluster"] == k + 1]
        plt.scatter(data["Criminal Damage"], data["Offences Against The Person"], c=color[k])
    plt.scatter(Centroids["Criminal Damage"], Centroids["Offences Against The Person"], c='red')
    plt.xlabel('Crimes')
    plt.ylabel('Criminal Damage + Offences ATP')
    plt.show()

#clustering()

def apriori():
    from mlxtend.frequent_patterns import apriori, association_rules
    import os

    os.chdir('C:/Users/Fraser/Documents/DataAnalytics')
    data = pd.read_csv('new_file.csv')

    # Building the model
    frq_items = apriori(data, min_support=0.1, use_colnames=True)

    # Collecting the rules
    rules = association_rules(frq_items, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

    # Filtering the rules
    filtered = rules[(rules['confidence'] > 0.60) & (rules['lift'] > 1)]
    filtered = filtered[filtered.consequents == {'Criminal Damage'}]

    # Cleaning the results
    print(filtered[['antecedents', 'consequents', 'lift']])

#apriori()