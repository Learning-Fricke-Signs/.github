#!/usr/bin/env python
# coding: utf-8

# # Notebook for Fricke sign paper, II
#  - For paper "Learning Fricke signs from Maass Form Coefficients"
#  - Exploring $a_p$ verus $a_n$, subsets of coefficients

# In[1]:


#| code-fold: true
#| code-summary: "Package Imports"
import numpy as np
import pandas as pd
import random
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
from sklearn.model_selection import train_test_split

# Sage Math
from sage.all import primes_first_n,is_prime, primes


# In[2]:


# Process data to make each Fourier coefficient have its own column
ALL_n_COLS = [str(n) for n in range(1,1001)]

def build_maass_an_df(DF):
    DF_new = pd.DataFrame()
    for rlf_label in DF.columns:
        if rlf_label == 'dirichlet_coefficients': continue
        DF_new[rlf_label] = DF[rlf_label].copy()
    DF_new[ALL_n_COLS] = [np.fromstring(a.strip('[]'), dtype=float, sep=',') for a in DF['dirichlet_coefficients']]
    return DF_new


# In[3]:

## Dataset available from https://zenodo.org/records/15490636/
file_name = 'Maassforms.txt'
df_maass = pd.read_table(file_name,delimiter=':',header=None)
columns = ['label','N','R','s','w','dirichlet_coefficients']
df_maass.columns = columns
df_maass['cond']=(df_maass['N']*df_maass['R']**2)/(4*np.pi**2)
df_maass = build_maass_an_df(df_maass)
df_maass.head()


# In[4]:


## Remove Maass forms with unknown Fricke sign
mask=df_maass['w']!=0
df_known_all=df_maass[mask]
print(df_known_all.shape)
df_known_all.head()


# In[5]:


## Normalize fourier coefficients of the Maass forms: multiply by (-1)^s for symmetry s
df_known_all[ALL_n_COLS] = df_known_all[ALL_n_COLS].apply(lambda x: x*((-1)^(df_known_all['s'])))


# In[6]:


df_known_all.head()


# In[ ]:





# # Creating subsets of a_n to test LDA on

# In[7]:


def lda_maass(x,y,random_state):
    # x = feature set, y= label set
    
    # Get test, train data sets - stratify sample on y=Fricke Sign
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, 
                                               test_size=float(0.2),
                                               random_state=random_state, 
                                               stratify=y)

    # Break training data into train and validate sets - stratify sample on y=Fricke Sign
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                      test_size=float(0.2), 
                                                      random_state=random_state, 
                                                      stratify=y_train_val)
        
    #print(f'Performing LDA on {len(x_train)} training observations.')
    # Do the LDA
    lda = LinearDiscriminantAnalysis(solver="svd",n_components=1)
    lda.fit(x_train,y_train)

    # Score the model on the validation set
    #print(f'LDA Accuracy on {len(x_val)} validation observations')
    #print(lda.score(x_val,y_val))
    
    return lda.score(x_val,y_val)


# In[8]:


## LDA on entire dataset
cols=ALL_n_COLS
x = df_known_all[cols].values # Features are a_n values
y = df_known_all['w'].values # Labels are Fricke sign groups='w'
print("LDA accuracy on full dataset")
print("num a_n:",len(cols))
print("accuracy: ",lda_maass(x,y,42))


# In[9]:


## creating a set of smooth numbers
smooth_nums=[str(2)]
smooth_bd=45
for n in range(3,1000):
    n=int(n)
    if max(prime_divisors(n))<smooth_bd:
        smooth_nums.append(n)
print("number of smooth a_n",len(smooth_nums))

## creating subset of [1,1000] for all numbers that have small prime factors
small_prime_factors=[]
prime_bd=6
for n in range(2,1000):
    #n=int(n)
    for p in primes(2,prime_bd):
        if n%p==0:
            if n not in small_prime_factors:
                small_prime_factors.append(n)
    
print("number of an with small prime factors",len(small_prime_factors))


# In[10]:


groups = 'w' # What we want to predict
random_state=42

col1=[str(n) for n in range(1,1001) if (n in smooth_nums)]
name1="a_n for smooth n"

col2=[str(n) for n in range(1,1001) if n not in smooth_nums]
name2="a_n for not smooth n"

col3=[str(n)for n in range(1,1001) if (n in small_prime_factors)]
name3="a_n for n with small prime factors"

col4=[str(n) for n in range(1,1001) if n not in  small_prime_factors]
name4="a_n for n with no small prime factors"

col5=[str(n) for n in range(1,1001) if is_prime(n)]
name5="a_n for prime n"

col6=[str(n)for n in range(1,1001) if (n in small_prime_factors or is_prime(n))]
name6="a_n for n with small prime factors or prime n"

# col6=[str(n) for n in range(1,1001) if not is_prime(n)]
# name6="a_n for n with composite factors"

col7=[str(n) for n in range(1,1001) if is_prime(n) or is_prime_power(n)]
name7="a_n for prime powers n"

col8=[str(n) for n in range(1,1001) if n%2==0]
name8="a_n for even n"

col9=[str(n) for n in range(1,1001) if is_odd(n)]
name9="a_n for odd n"

col10=[str(n) for n in range(1,1001) if (is_prime(n) or len(factor(n))<=2) and n<1001]
name10="a_n for n with 1 or 2 prime factors"

col11=[str(n) for n in range(1,1001) if n<500]
name11="a_n for n<500"

subset_col_list=[col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11]
names_list=[name1,name2,name3,name4,name5,name6,name7,name8,name9,name10,name11]
   
for cols,name in zip(subset_col_list,names_list):
    df_subset=df_known_all.copy()
    print(name)
    df_subset=df_subset[['label','N','R','s','w','cond']+cols]
    print("num a_n:",len(cols))
    
    # # Get features and labels
    x = df_subset[cols].values # Features are a_n values
    y = df_subset[groups].values # Labels are Fricke sign groups='w'
    print("accuracy: ",lda_maass(x,y,42))
    print("  ")


# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


## eliminating one prime at a time
bd=1000
score_list=[]

for i in tqdm.tqdm(range(1,bd)):
    cols=[str(n) for n in primes(bd) if n!=nth_prime(i)]
    #print(nth_prime(i))
    df_sample=df_known_all.copy()

    # # Get features and labels
    x = df_sample[cols].values # Features are a_n values
    y = df_sample[groups].values # Labels are Fricke sign groups='w'
    
    score_list.append(lda_maass(x,y,42))

print("range of values for dropping a single prime from the list") 
print(f"minimum score: {min(score_list)} at prime {nth_prime(np.argmin(score_list)+1)}")
print(f"maximum score: {max(score_list)} at prime {nth_prime(np.argmax(score_list)+1)}")


# In[ ]:





# In[12]:


# Create random subsets of 2,1000 of size 168
# to compare random subsets which are the same size as the size of the primes

bd=100
score_list_random=[]
cols_list=[]

for i in tqdm.tqdm(range(bd)):
    cols=[str(n) for n in random.sample(range(2,1000), 168)]
    cols_list.append(cols) ##random sets; so might want these
    df_sample=df_known_all.copy()

    # # Get features and labels
    x = df_sample[cols].values # Features are a_n values
    y = df_sample[groups].values # Labels are Fricke sign groups='w'
    
    score_list_random.append(lda_maass(x,y,42))
   
print(min(score_list_random),max(score_list_random))


# In[13]:


##random subsest of 168 elements: 61.4 to 68.2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




