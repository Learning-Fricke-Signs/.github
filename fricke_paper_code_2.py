#!/usr/bin/env python
# coding: utf-8

# ## This code is for revisions to paper "Learning Fricke signs from Maass Form Coefficients".
# 
# Extra data analysis to answer questions from reviewers:
#  - Are some levels easier to predict?  
#  - Covariance analysis between two classes

# In[24]:


#| code-fold: true
#| code-summary: "Package Imports"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

# Plotting
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'colab+jupyterlab+pdf+iframe'
## note: added iframe to make it work on my personal machine rather than in colab

# ML packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# Sage Math
from sage.all import primes_first_n,is_prime, primes, factor


# In[25]:


# Process data to make each Fourier coefficient have its own column
ALL_n_COLS = [str(n) for n in range(1,1001)]

def build_maass_an_df(DF):
    DF_new = pd.DataFrame()
    for rlf_label in DF.columns:
        if rlf_label == 'dirichlet_coefficients': continue
        DF_new[rlf_label] = DF[rlf_label].copy()
    DF_new[ALL_n_COLS] = [np.fromstring(a.strip('[]'), dtype=float, sep=',') for a in DF['dirichlet_coefficients']]
    return DF_new


# In[26]:

## Dataset available from https://zenodo.org/records/15490636/
file_name = 'Maassforms.txt'
DF_new = pd.read_table(file_name,delimiter=':',header=None)
columns = ['label','N','R','s','w','dirichlet_coefficients']
DF_new.columns = columns
DF_new['cond']=(DF_new['N']*DF_new['R']**2)/(4*np.pi**2)
DF_all = build_maass_an_df(DF_new)
DF_all.head()


# In[ ]:





# In[27]:


# Graph the distributions of the Fricke sign by level
# 0=unknown
#Note: the levels in the dataset are all squarefree numbers from 1 to 105.

for flag in ["prime","composite"]:
    if flag=="prime":
        mask=(DF_all['N']>1)&([is_prime(int(n)) for n in DF_all['N']])
    if flag=="composite":
        mask=(DF_all['N']>0)&([ not is_prime(int(n)) for n in DF_all['N']])
    DF_plot=DF_all[mask]
    df_w1=DF_plot[DF_plot["w"]==1]
    df_wn1=DF_plot[DF_plot["w"]==-1]
    df_w0=DF_plot[DF_plot["w"]==0]
    level_w1=df_w1.N.values
    level_w1.sort()
    level_wn1=df_wn1.N.values
    level_wn1.sort()
    level_w0=df_w0.N.values
    level_w0.sort()
    plt.hist([level_w1,level_wn1,level_w0],density=True, stacked=True,bins=105, label=['1','-1','0'])
    plt.legend(title='Fricke sign')
    plt.ylabel("frequency")
    plt.xlabel(f"level ({flag})")
    plt.title(f"Distribution of Fricke signs for {flag} levels")
    fig_name=f'histogram_{flag}_levels_fricke_signs.png'
    plt.savefig(fig_name)
    plt.show()


# In[28]:


## Examining the distribution of known and unknown Fricke signs


# In[29]:


level_nums=[]
unknown_percents=[]
for level in set(DF_all.N.values):
    df_sub=DF_all[DF_all.N==level]
    if level>1:
        level_nums.append(level)
        unknown_percents.append(list(df_sub.w.values).count(0)/len(df_sub.w.values))


# In[30]:


level_subset1=[]
level_subset2=[]
unk_1=[]
unk_2=[]
for i,level in enumerate(level_nums):
    if level in primes(105):
        label1="prime levels"
#     if level%2==0:
#         label1="evens"
        level_subset1.append(level)
        unk_1.append(unknown_percents[i])
    else:
        label2="composite levels"
#         label2="odds"
        level_subset2.append(level)
        unk_2.append(unknown_percents[i])


# In[31]:


plt.scatter(level_subset1,unk_1,color='blue',label=label1)
plt.scatter(level_subset2,unk_2,color='green',label=label2)
plt.legend()
plt.xlabel("level")
plt.ylabel("percent")
plt.title("percent of unknown Fricke signs by level")
#plt.savefig("Fricke_unk_level")
plt.show()


# In[ ]:





# ## LDA analysis for entire dataset

# In[32]:


ALL_n_COLS=[str(i) for i in range(1,1001)]

groups = 'w' # What we want to predict
random_state=42

# Get a copy of the data
DF_sample=DF_all.copy()

# # Normalize by (-1)^s
DF_sample[ALL_n_COLS] = DF_sample[ALL_n_COLS].apply(lambda x: x*((-1)**(DF_sample['s'])))

# Removing samples where Fricke sign = 0 (unknown)
mask = DF_sample['w']!=0  
DF_sample = DF_sample[mask]

# # Get features and labels
x = DF_sample[ALL_n_COLS].values # Features are a_n values
y = DF_sample[groups].values # Labels are Fricke sign groups='w'

# Get test, train data sets - stratify sample on y=Fricke Sign
x_train_val, x_test, y_train_val, y_test, = train_test_split(x, y, 
                                                            test_size=float(0.2),
                                                            random_state=random_state, 
                                                            stratify=y)

# Break training data into train and validate sets - stratify sample on y=Fricke Sign
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, 
                                                test_size=float(0.2), 
                                                random_state=random_state, 
                                                stratify=y_train_val)

# Do the LDA
print(f'Performing LDA on {len(x_train)} training observations.')
lda = LinearDiscriminantAnalysis(solver="svd",n_components=1)
lda.fit(x_train,y_train)
print(f'LDA Accuracy on {len(x_val)} validation observations')
print(lda.score(x_val,y_val))


# #Do Quadratic discriminant analysis (this didn't perform well)
# print(f'Performing QDA on {len(x)} training observations.')
# qda = QuadraticDiscriminantAnalysis()
# qda.fit(x,y)
# # Score the model on the validation set
# print(f'QDA Accuracy on {len(xv)} validation observations')
# print(qda.score(xv,yv))
## This is not as good.  60%.


# In[ ]:





# ## LDA analysis for various subsets of Maass forms by level

# In[33]:


# Make list of smooth numbers
smooth_N=[]
smooth_bd=11
for n in set(DF_all.N.values):
    if n==1:
        continue
    else:    
        max_prime=list(factor(n))[-1][0]
        if max_prime<=smooth_bd:
            smooth_N.append(n)


# In[34]:


# Get a copy of the data
DF_sample=DF_all.copy()

# # Normalize by (-1)^s
DF_sample[ALL_n_COLS] = DF_sample[ALL_n_COLS].apply(lambda x: x*((-1)**(DF_sample['s'])))

# Removing samples where Fricke sign = 0 (unknown)
mask = DF_sample['w']!=0  
DF_sample = DF_sample[mask]

## Remove N=1, since w=1 always on this dataset. 
mask = DF_sample['N']!=1
DF_sample =DF_sample[mask]

mask_list=[DF_sample.N>1,
           DF_sample.N%2==0,DF_sample.N%2!=0,
           DF_sample.N%3==0,DF_sample.N%3!=0,DF_sample.N%3==1,DF_sample.N%3==2,
           DF_sample.N%5==0,DF_sample.N%5!=0,
           DF_sample.N<=50,DF_sample.N>50,
          [is_prime(int(n)) for n in DF_sample['N']],[not is_prime(int(n)) for n in DF_sample['N']],
           [n in smooth_N for n in DF_sample['N']],[not n in smooth_N for n in DF_sample['N']],
           DF_sample.N<=10,DF_sample.N>10,
          ]

name_list=['No Level 1    ',
 'Even Levels    ',
 'Odd Levels    ',
 'Multiple Of 3    ',
 'Not Multiple Of 3    ',
 '1 Mod 3    ',
 '2 Mod 3    ',
 'Multiple Of 5    ',
 'Not Multiple Of 5    ',
 'Level < 50    ',
 'Level > 50    ',
 'Prime Level    ',
 'Composite Level    ',
 'Not 11-Smooth    ',
 '11-Smooth    ',
  'Level < 10    ',
 'Level > 10    ',
    ]

for i,mask in enumerate(mask_list):
    print(name_list[i])
    DF_subsample = DF_sample[mask]

    # # Get features and labels
    x = DF_subsample[ALL_n_COLS].values # Features are a_n values
    y = DF_subsample[groups].values # Labels are Fricke sign groups='w'

    # Get test, train data sets - stratify sample on y=Fricke Sign
    x_train_val, x_test, y_train_val, y_test, = train_test_split(x, y, 
                                                                test_size=float(0.2),
                                                                random_state=random_state, 
                                                                stratify=y)

    # Break training data into train and validate sets - stratify sample on y=Fricke Sign
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, 
                                                    test_size=float(0.2), 
                                                    random_state=random_state, 
                                                    stratify=y_train_val)

    # Do the LDA
    print(f'Performing LDA on {len(x_train)} training observations.')
    lda = LinearDiscriminantAnalysis(solver="svd",n_components=1)
    lda.fit(x_train,y_train)
    print(f'LDA Accuracy on {len(x_val)} validation observations')
    print(lda.score(x_val,y_val))
    print(" ")


# In[ ]:





# In[ ]:





# In[ ]:





# # Covariance Analysis
#  - Analyze the covariance between the set of Maass forms with Fricke sign equal to 1 and the set of Maass forms with Fricke sign equal to -1.
#  - Apply normalization first to match how we applied LDA.
#  - Construct covariance matrices, $1000 \times 1000$ matrices
#  - Compute eigenvalues and determinants of matrices
#  - Use box\_m test to analyze covariance

# In[35]:


# For covariance matrix
from sklearn.covariance import empirical_covariance
# For box_m test
import pingouin as pg


# In[36]:


## covariance matrix is variance and all covariance of x_train with fricke =1 
## and similarly for fricke =-1


# In[37]:


## Separate data into two subsets, w=1, w=-1
DF_sample=DF_all.copy()
# # Normalize by (-1)^s
DF_sample[ALL_n_COLS] = DF_sample[ALL_n_COLS].apply(lambda x: x*((-1)**(DF_sample['s'])))

mask1 = DF_sample['w']==1
df_1 = DF_sample[mask1]
data = df_1[ALL_n_COLS].values
covariance_matrix_1 = empirical_covariance(data)

mask2 = DF_sample['w']==-1
df_2 = DF_sample[mask2]
data = df_2[ALL_n_COLS].values
covariance_matrix_2 = empirical_covariance(data)


# ## Compare covariance matrices

# In[38]:


a=0
b=5

# values with maximum difference value between covariance matrices
# a=697
# b=703

print("fricke sign is 1")
print(covariance_matrix_1[a:b,a:b])
print("")
print("fricke sign is -1")
print(covariance_matrix_2[a:b,a:b])
print(" ")
print("difference of matrices")
print((covariance_matrix_1-covariance_matrix_2)[a:b,a:b])

print(" ")
print("maximum value of differences in matrices, over full range")
print((covariance_matrix_1-covariance_matrix_2).max())


# ### Determinants and Eigenvalues

# In[39]:


print("fricke sign is 1")
cov1=matrix(covariance_matrix_1)
print("determinant:", cov1.determinant())
print("first 10 eigenvalues")
print(cov1.eigenvalues()[0:10])
print(" ")
print("fricke sign is -1")
cov2=matrix(covariance_matrix_2)
print("determinant:", cov2.determinant())
print("first 10 eigenvalues")
print(cov2.eigenvalues()[0:10])


# In[40]:


M=covariance_matrix_1-covariance_matrix_2
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[i,j]>.22:
            print(i,j,M[i,j])


# In[ ]:





# ## Examining distributions over the two different data sets 

# In[41]:


## Construct distributions of $a_n$ values with all levels and with just levels co-prime to n.

n=7 #for which a_n
col_n=str(n)

mask1_1=[gcd(n,N)==1 for N in df_1['N']]
mask2_1=[gcd(n,N)==1 for N in df_2['N']]
mask1_2=[gcd(n,N)!=1 for N in df_1['N']]
mask2_2=[gcd(n,N)!=1 for N in df_2['N']]
test_1=df_1[mask1_1]
test_2=df_2[mask2_1]
test_3=df_1[mask1_2]
test_4=df_2[mask2_2]

low_bd=round(df_1[col_n].min())
up_bd=round(df_1[col_n].max())
names=[f"for levels co-prime to {n}", f"for all levels",f'for levels not co-prime to {n}']

for i,df_list in enumerate([ [test_1,test_2], [df_1,df_2], [test_3,test_4]]):
    name=names[i]
    df_list[0][col_n].hist(density=True, bins=np.arange(low_bd,up_bd,.1),alpha=.7,label='1')
    df_list[1][col_n].hist(density=True,alpha=.5,bins=np.arange(low_bd,up_bd,.1),label='-1')
    plt.legend(title="Fricke sign")
    fig_name1=f'dist_{n}_fricke_levels_{i}'
    title_name=f'distribution of $a({n})$ '+name
    plt.title(title_name)
    if is_prime(n) and i==0:
        z=np.linspace(-2,2,100)
        plt.plot(z,1/(2*pi)*(4-z^2)^0.5)
    plt.xlabel(f'$a({n})$')
    plt.ylabel("frequency")
    plt.savefig(fig_name1)
    plt.show()


# # Box's M test  and $a_p$ distributions 
# 

# In[42]:


# Get a copy of the data
DF_sample=DF_all.copy()

# Removing samples where Fricke sign = 0 (unknown)
mask = DF_sample['w']!=0  
DF_sample = DF_sample[mask]

# # Normalize by (-1)^s
DF_sample[ALL_n_COLS] = DF_sample[ALL_n_COLS].apply(lambda x: x*((-1)^(DF_sample['s'])))


# In[43]:


true_n=[]
false_n=[]
for n in range(2,1000):
    n=str(n)
    info=pg.box_m(DF_sample, dvs=[n], group = 'w')
    if info['equal_cov'].values[0]==True:
        true_n.append(n)
    else:
        false_n.append(n)
print("For which a_n are the covariances equal from box_m test")
print("True")
print(len(true_n))
print(true_n[0:15])
print("False")
print(len(false_n))
print(false_n[0:15])


# In[44]:


n_list=[['5'],['6'],['2','3','5','7'],['160']]
for n in n_list:
    print(f"for n = {n}")
    print(pg.box_m(DF_sample, dvs=n, group = 'w'))
    


# In[ ]:





# ## Removing levels $N$ when $\gcd(n,N)>1$

# In[45]:


df_test=DF_sample.copy()
true_n_all=[]
false_n_all=[]
true_n_coprime=[]
false_n_coprime=[]

for n in tqdm.tqdm(range(2,1001)):
    mask=[gcd(n,N)==1 for N in df_test['N']]
    df_test_level=df_test[mask]
    ncol=str(n)
    #print(n)
    #print(df_test_level.shape)
    #print(set(df_test_level.N))
    #print(pg.box_m(df_test_level,dvs=[str(n)], group = 'w'))
    info1=pg.box_m(df_test, dvs=[ncol], group = 'w')
    info2=pg.box_m(df_test_level, dvs=[ncol], group = 'w')
    if info1['equal_cov'].values[0]==True:
        true_n_all.append(n)
    else:
        false_n_all.append(n)
    if info2['equal_cov'].values[0]==True:
        true_n_coprime.append(n)
    else:
        false_n_coprime.append(n)    
            
print("For which a_n are the covariances equal from box_m test")
print("True")
print(len(true_n_all))
print(true_n_all[0:15])
print("False")
print(len(false_n_all))
print(false_n_all[0:15])


print("For which a_n are the covariances equal from box_m test")
print("True")
print(len(true_n_coprime))
print(true_n_coprime[0:15])
print("False")
print(len(false_n_coprime))
print(false_n_coprime[0:15])


# In[46]:


# for n in false_n:
#     print(int(n),factor(int(n)))


# In[ ]:




