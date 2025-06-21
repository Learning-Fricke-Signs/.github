#!/usr/bin/env python
# coding: utf-8

# # Notebook for Fricke sign paper, I
#  - For paper "Learning Fricke signs from Maass Form Coefficients"
#  - Murmuration type patterns: average $a_p$ by Fricke sign, exploring by symmetry
#  - Normalization technique
#  - LDA
#  - LDA analysis, primes that divide the level, and level explorations

# ## Getting and processing the data

# In[1]:


import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
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
from sage.all import primes_first_n,is_prime

PRIME_COLUMNS=[str(p) for p in primes(1000)]

# Process data to make each Fourier coefficient have its own column
ALL_n_COLS = [str(n) for n in range(1,1001)]

def build_maass_an_df(DF):
    DF_new = pd.DataFrame()
    for rlf_label in DF.columns:
        if rlf_label == 'dirichlet_coefficients': continue
        DF_new[rlf_label] = DF[rlf_label].copy()
    DF_new[ALL_n_COLS] = [np.fromstring(a.strip('[]'), dtype=float, sep=',') for a in DF['dirichlet_coefficients']]
    return DF_new


# ## Data
# 
# * Data contains 35416 rigorously computed Maass forms from the LMFDB
# * all have motivic weight 0, trivial character
# * all have squarefree level at most 105 (with bounds on the spectral parameter that depend on the level)
# 
# #### Variables
# 
# * label is the Maass form label from LMFDB
# * N is the level
# * R is the spectral paramter
# * s is the symmetry (0=even,1=odd)
# * w is the Fricke sign (+/-1 when known, 0=unknown). We don't know the Fricke sign in about 1/3 of the cases.
# * dirichlet_coefficients is a list of a_n for n in range(1000). Processing converts data to make each a_n its own column.
# 

# In[22]:

## Dataset available from https://zenodo.org/records/15490636/
file_name = 'Maassforms.txt'
DF_maass= pd.read_table(file_name,delimiter=':',header=None)
columns = ['label','N','R','s','w','dirichlet_coefficients']
DF_maass.columns = columns
#adding analytic conductor to data
DF_maass['cond']=(DF_maass['N']*DF_maass['R']**2)/(4*np.pi**2)
DF_all = build_maass_an_df(DF_maass)
DF_all.head()


# In[3]:


DF_prime=DF_all[['label','N','R','s','w','cond']+PRIME_COLUMNS]
print(len(PRIME_COLUMNS))
DF_prime.head()


# In[ ]:





# In[ ]:





# # Part 1: Murmurations

# ## Explore Murmurations
# 
# Here we plot the average $a_p$ values - separated by Fricke sign. We remove cases where the Fricke sign is unknown, given in the data as $w=0$. We consider three experiments:
# 
# 1. Plot all data where we know the Fricke sign.
# 2. Plot only $s=0$ data where we know the Fricke sign. (Even)
# 3. Plot only $s=1$ data where we know the Fricke sign. (Odd)
# 
# 
# ## Plot average $a_p$ values for Maass Forms Grouped on Fricke Sign 

# In[4]:


# Remove unknown Fricke signs
DF_known = DF_prime.copy()
mask = DF_known['w']!=0
DF_known = DF_known[mask]

# Use gropus to separate by value of Fricke sign
groups = 'w'
my_columns = PRIME_COLUMNS.copy()
my_columns += [groups]  

plot_formatting='''autosize=False,
                  width=1000,
                  height=600'''


# In[5]:


# Separate by None, even or odd symmetry
# Note the sign change for average a_p for even and odd Maass forms

### Pick symmetry
sym_s=None
#sym_s=0
#sym_s=1

DF_plot = DF_known.copy()

if sym_s is not None:
    mask=DF_plot['s']==sym_s
    DF_plot = DF_plot[mask]  
    
# Group the data by Fricke sign and average
DF_ave = DF_plot[my_columns].groupby(groups).mean()
new_cols = list(DF_ave.index)

# Transpose the data for plotting
dfT=DF_ave.transpose()
dfT['p'] = PRIME_COLUMNS

# Make nicer for plotting
dfT=dfT[[1,-1,'p']]
new_cols=dfT.columns
if sym_s==0:
    title_plot=r'$p \text{ vs average } a_p \text{ for even Maass Forms}$'
if sym_s==1:
    title_plot=r'$p \text{ vs average } a_p \text{ for odd Maass Forms}$'
if sym_s is None:
    title_plot=r'$p \text{ vs average } a_p \text{ for all Maass Forms}$'

# Create Scatter Plot
fig = px.scatter(dfT,x='p',y=new_cols,color_discrete_map={'1':'blue','-1':'red'})
fig.update_layout(title=title_plot,
                  title_x=0.5,
                  template='plotly_white',
                  xaxis_title="p",
                  yaxis_title=r"$\text{average } a_p$",
                  legend_title='Fricke Sign',
                  autosize=False,
                  width=1000,
                  height=600)

fig.write_image('MaassForms_p_vs_Ap_allsym'+'.png')

fig.show()


# In[6]:


# Get the data for s=0
mask = DF_known['s']==0
DF_even = DF_known[mask]
DF_ave_even = DF_even[my_columns].groupby(groups).mean()
new_cols = list(DF_ave_even.index)
dfT_even=DF_ave_even.transpose()
dfT_even.columns = new_cols
dfT_even['p'] = PRIME_COLUMNS

# Get the data for s = 1
mask = DF_known['s']==1
DF_odd = DF_known[mask]
DF_ave_odd = DF_odd[my_columns].groupby(groups).mean()
new_cols = list(DF_ave_odd.index)
dfT_odd=DF_ave_odd.transpose()
dfT_odd.columns = new_cols
dfT_odd['p'] = PRIME_COLUMNS

# Rename the columns
dfT_even.rename(columns={-1:r'$w=-1,\sigma=0$',1:r'$w=1,\sigma=0$'},inplace=True)
dfT_odd.rename(columns={-1:r'$w=-1,\sigma=1$',1:r'$w=1,\sigma=1$'},inplace=True)

# Merge the data
dfT = pd.merge(dfT_odd,dfT_even,on='p',how='inner')
new_cols = [n for n in dfT.columns if n!='p']

fig = px.scatter(dfT,x='p',y=new_cols,
                color_discrete_map={r'$w=-1,\sigma=0$':'lightpink', 
                                    r'$w=1,\sigma=0$':'lightskyblue', 
                                    r'$w=-1,\sigma=1$':'blue', 
                                    r'$w=1,\sigma=1$':'red'})
fig.update_layout(#title=f'p vs Average $a_p$ values for Maass Forms type - Symmetry Plots Combined',
                  title=r'$p \text{ vs average } a_p \text{ for Maass Forms separated by Symmetry and Fricke sign}$',
                  title_x=0.5,
                template='plotly_white',
                  xaxis_title="$p$",
                  yaxis_title=r"$\text{average } a_p$",
                  legend_title='Fricke Sign, Symmetry',
                legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 20)),
                  autosize=False,
                  width=1000,
                  height=600)

fig.write_image('MaassForms_p_vs_Ap_symCombined'+'.png')

fig.show()


# ### Add normalization
# 
# The sign change for average a_p for even and odd Maass forms, led to the normalization by symmetry $\sigma$:
# $$a_n= a_n(-1)^\sigma$$

# In[7]:


DF_plot = DF_known.copy()
# Apply normalization
DF_plot[PRIME_COLUMNS] = DF_plot[PRIME_COLUMNS].apply(lambda x: x*((-1)^(DF_plot['s'])))
# Group the data by Fricke sign and average
DF_ave = DF_plot[my_columns].groupby(groups).mean()
new_cols = list(DF_ave.index)

# Transpose the data for plotting
dfT=DF_ave.transpose()
dfT['p'] = PRIME_COLUMNS

# Make nicer for plotting
dfT=dfT[[1,-1,'p']]
new_cols=dfT.columns

# Create Scatter Plot
fig = px.scatter(dfT,x='p',y=new_cols,color_discrete_map={'1':'blue','-1':'red'})
fig.update_layout(title=r'$p \text{ vs average } a_p \text{ for normalized Maass Forms}$',
                  title_x=0.5,
                  template='plotly_white',
                  xaxis_title="p",
                  yaxis_title=r"$\text{average } a_p$",
                  legend_title='Fricke Sign',
                  autosize=False,
                  width=1000,
                  height=600)

fig.write_image('MaassForms_p_vs_Ap_normalize'+'.png')

fig.show()


# In[ ]:





# In[ ]:





# # Connection between Fricke sign and level
# 
# If p divides the level, $N$, then $a_p=\frac{-w_p}{\sqrt{p}}$ and Fricke sign 
# $w_N=\prod_{p|N} w_p$. 
# 
# If Fricke sign is unknown then we don't know the 
# $a_p$ when $p|N$ and they are defined to be 0 in this data.
# 
# Note: In this dataset the level, $N$, ranges from 1 to 105. 
# - There is no impact of this if $N=1$, although Fricke sign is equal to 1 for all of those Maass forms.
# - There is no relationship in this dataset between primes and Fricke sign for primes bigger than the largest level (105).
# 
# CONCERN: the value of the Fricke sign is computable from what's in the data.
#  - If we use the dataset with just $a_p$ for primes $p <1000$, its computatable as above. 
#  - If we use the dataset with $a_n$ for all $n<1000$, its also computable as $$w=w_N=\text{sign}(a_n)\prod_{p|N} (-1).$$

# In[8]:


DF_known = DF_all.copy()
mask = DF_known['w']!=0
DF_known = DF_known[mask]
# pick level
N=30
sample_maass_form=DF_known[DF_known["N"]==N].iloc[int(5)]
print(sample_maass_form[['label','w','N','s']])
print(" ")
sample_level=sample_maass_form.N
fricke_sign=sample_maass_form.w
print(f'For sample Maass form in level, N= {sample_level}, with prime divisors {prime_divisors(sample_level)}')
wp_list=[]
for p in prime_divisors(sample_level):
    ap=sample_maass_form[str(p)]
    print(f'   a_{p} = {ap} = -w_{p}*{round(1/sqrt(p),6)}')
    print(f'   So, w_{p} = {-1*round(ap*sqrt(p),2)}')
    wp_list.append(-1*round(ap*sqrt(p),2))
    print("   ")
print(f'Finally, w_N = product{wp_list}={fricke_sign}')
print(f'a_{sample_level}={sample_maass_form[str(N)]}')


# # Compare effects of zeroing out coefficients where prime divides the level on murmurations

# ## Create another dataset
# - To test if how much this effects LDA analysis, create dataset that zeros out all values where $n$ and $N$ have factors in common.

# In[9]:


## this takes a few minutes to run
DF_zero_all=DF_all.copy()
for i in range(len(DF_zero_all)):
    #print("index",i)
    level=DF_zero_all.loc[int(i),'N']
    fricke=DF_zero_all.loc[int(i),'w'] ##unknown fricke sign, already zero in this case
    if level==1 or fricke==0:
        continue
    else: 
        #print(level)
        for n in range(1001):
            if gcd(n,int(level))!=1:
                DF_zero_all.loc[int(i),str(n)]=int(0)


# In[10]:


col_set=[str(n) for n in range(1,1001) if gcd(n,sample_level)>1]
DF_zero_all[DF_zero_all["label"] == sample_maass_form.label][col_set]


# In[ ]:





# In[11]:


# start at 3, because 2 is weird
# NOTE: zeroing out when the prime divides the level and removing the $a_p$ when 
# when the prime divides the level produced similar results, so kept only one in final data.
loprime=3
val_dict={}
for w in [1,-1]:
    df=DF_prime.copy()
    df_zero_test=DF_zero_all.copy()
    #df[PRIME_COLUMNS] = df[PRIME_COLUMNS].apply(lambda x: x*((-1)^(df['s'])))
    df = df[df['w']==w ]
    df_z = df_zero_test[df_zero_test['w']==w ]
    #sym_remove=[]
    sym_keep=[]
    sym_zero=[]
    for p in primes(loprime,105):
        df_test=df[df['N']%p!=0]
        #sym_remove.append(df_test[str(p)].mean())
        sym_keep.append(df[str(p)].mean())
        sym_zero.append(df_z[str(p)].mean())
    val_dict[f'w={w},correct']=sym_keep
    #val_dict[f'w={w},remove']=sym_remove
    val_dict[f'w={w},zero']=sym_zero
        
df_test=pd.DataFrame(val_dict)
df_test['p']=[str(p) for p in primes(loprime,105)]


# In[12]:


# change notation to be consistent with paper
new_cols=[r"$w=1,a_p$", r"$w=1,a_p^{'}$",r"$w=-1,a_p$", r"$w=-1,a_p^{'}$",'p']
df_test.columns=new_cols


# In[13]:


y_cols = [n for n in df_test.columns if n!="p"]
fig = px.scatter(df_test,x='p',y=y_cols, color_discrete_map={new_cols[0]:'blue',new_cols[1]:'cyan',new_cols[2]:'red',new_cols[3]:'tomato'})
fig.update_layout(title=r'$\text{Comparing average } a_p \text{ values}$',
                  title_x=0.5,
                  template='plotly_white',
                  xaxis_title="$p$",
                  yaxis_title=r"$\text{average } a_p$",
                  legend_title='Fricke Sign',
                  autosize=False,
                  width=1200,
                  height=600)
fig.write_image('Comparing_average_ap_values'+'.png')
fig.show()


# In[ ]:





# In[ ]:





# # Part 3: Using LDA to classify Fricke sign from $a_p$ and $a_n$ values
# 
# See [geeksforgeeks.org](https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/) for nice description of how LDA works.

# ## LDA accuracy code
# 
# Compare accuracy from LDA on: 
# 
# 1. All Maass forms in training data
# 2. Even Maass forms in training data
# 3. Odd Maass forms in training data

# In[14]:


def lda_maass_sym(x_train,y_train,s_train,x_val,y_val,s_val,sym=None):

    if sym == 'Even':
        # Restrict the data
        x = x_train[s_train == 0]
        y = y_train[s_train == 0]
        xv = x_val[s_val == 0]
        yv = y_val[s_val == 0]
    elif sym == 'Odd':
        # Restrict the data
        x = x_train[s_train != 0]
        y = y_train[s_train != 0]
        xv = x_val[s_val != 0]
        yv = y_val[s_val != 0]
    else:
        x = x_train
        y = y_train
        xv = x_val
        yv = y_val
        

    # Do the LDA
    lda = LinearDiscriminantAnalysis(solver="svd",n_components=1)
    lda.fit(x,y)

    # Score the model on the validation set
    
    print(f'Performing LDA on {len(x)} training observations with symmetry restricted to {sym}')
    print(f'LDA Accuracy on {len(xv)} validation observations')
    print(lda.score(xv,yv))
    
    return lda


# ## Prepare the Data
# 
# * We choose a random sample test size of 20% and training set of 80%
# * Stratify on group we are predicting (here on the Fricke Sign $w$)
# * Then we split the training set into validation size 20% and training size 20% - in case we need to tune.
# * Choose dataset from: 
#    - ALL coeficients, i.e., all $a_n$ for $n<1000$  -OR- PRIME coefficients, i.e., all $a_p$ for prime $p<1000$
#    - zeroing out $a_n$ where $\gcd(n,N)>1$ -OR- keep original $a_n$ which contain information about Fricke sign 

# In[17]:


groups = 'w' # What we want to predict
random_state=42

for n_Flag in ["PRIMES","ALL"]:
    for zero_flag in [True, False]:

        print("data set is ", n_Flag)
        print("zeroing out primes that divide the level:", zero_flag)

        ## Get the desired dataset and columns
        if n_Flag=="ALL":
            cols = ALL_n_COLS
            if zero_flag:
                DF_sample=DF_zero_all.copy()
            else:
                DF_sample=DF_all.copy()
        if n_Flag=="PRIMES":
            cols = PRIME_COLUMNS
            if zero_flag:
                DF_sample=DF_zero_all.copy()
                DF_sample = DF_sample[['label','N','R','s','w','cond']+PRIME_COLUMNS]
            else:
                DF_sample=DF_prime.copy()

        # # Normalize by (-1)^s
        DF_sample[cols] = DF_sample[cols].apply(lambda x: x*((-1)^(DF_sample['s'])))

        # # Separate into known and unknown Fricke sign
        DF_unk=DF_sample.copy()
        mask = DF_unk['w']==0    
        DF_unk = DF_unk[mask]

        DF_known=DF_sample.copy()
        mask = DF_sample['w']!=0    
        DF_known = DF_known[mask]

        # # Get features and labels
        x = DF_known[cols].values # Features are An/ap values
        y = DF_known[groups].values # Labels are Fricke sign groups='w'
        s = DF_known['s']

        # Get test, train data sets - stratify sample on y=Fricke Sign
        x_train_val, x_test, y_train_val, y_test, s_train_val, s_test = train_test_split(x, y, s,
                                                                                            test_size=float(0.2),
                                                                                            random_state=random_state, 
                                                                                            stratify=y)

        # Break training data into train and validate sets - stratify sample on y=Fricke Sign
        x_train, x_val, y_train, y_val, s_train, s_val = train_test_split(x_train_val, y_train_val, s_train_val,
                                                                              test_size=float(0.2), 
                                                                              random_state=random_state, 
                                                                              stratify=y_train_val)

        #print("size of training set:",len(x_train))
        #print("size of testing set:",len(x_val))
        print("  ")
        print("LDA for ALL Maass forms in given dataset")
        lda_all = lda_maass_sym(x_train,y_train,s_train,x_val,y_val,s_val)
        print(" ")
        print("LDA for EVEN Maass forms in given dataset")
        lda_even=lda_maass_sym(x_train,y_train,s_train,x_val,y_val,s_val,sym="Even")
        print(" ")
        print("LDA for ODD Maass forms in given dataset")
        lda_odd=lda_maass_sym(x_train,y_train,s_train,x_val,y_val,s_val,sym="Odd")
        print(" ")


# In[18]:


#Comparing these four methods as we increase the number of a_n or a_p

score_dict={}

groups = 'w' # What we want to predict
random_state=42

for n_Flag in ["PRIMES","ALL"]:
    for zero_flag in [True, False]:

        print("data set is ", n_Flag)
        print("zeroing out primes that divide the level:", zero_flag)

        ## Get the desired dataset and columns
        if n_Flag=="ALL":
            cols = ALL_n_COLS
            if zero_flag:
                DF_sample=DF_zero_all.copy()
            else:
                DF_sample=DF_all.copy()
        if n_Flag=="PRIMES":
            cols = PRIME_COLUMNS
            if zero_flag:
                DF_sample=DF_zero_all.copy()
                DF_sample = DF_sample[['label','N','R','s','w','cond']+PRIME_COLUMNS]
            else:
                DF_sample=DF_prime.copy()

        # # Normalize by (-1)^s
        DF_sample[cols] = DF_sample[cols].apply(lambda x: x*((-1)^(DF_sample['s'])))

        # # Separate into known and unknown Fricke sign
        DF_unk=DF_sample.copy()
        mask = DF_unk['w']==0    
        DF_unk = DF_unk[mask]

        DF_known=DF_sample.copy()
        mask = DF_sample['w']!=0    
        DF_known = DF_known[mask]

        # # Get features and labels
        x = DF_known[cols].values # Features are An/ap values
        y = DF_known[groups].values # Labels are Fricke sign groups='w'
        s = DF_known['s']

        # Get test, train data sets - stratify sample on y=Fricke Sign
        x_train_val, x_test, y_train_val, y_test, s_train_val, s_test = train_test_split(x, y, s,
                                                                                            test_size=float(0.2),
                                                                                            random_state=random_state, 
                                                                                            stratify=y)

        # Break training data into train and validate sets - stratify sample on y=Fricke Sign
        x_train, x_val, y_train, y_val, s_train, s_val = train_test_split(x_train_val, y_train_val, s_train_val,
                                                                              test_size=float(0.2), 
                                                                              random_state=random_state, 
                                                                              stratify=y_train_val)

        ## Increasing percentage of a_n/a_p
        bins=20

        scores = []
        num_ap = []

        for num in range(1,bins+1):
            num_set=round(len(cols)/bins)
            x_train_new=x_train[:,:num*num_set]
            lda = LinearDiscriminantAnalysis(solver="svd",n_components=1)
            lda.fit(x_train_new,y_train)
            scores.append(lda.score(x_val[:,:num*num_set],y_val))
            num_ap.append(num*num_set)
        
        score_dict[(n_Flag,zero_flag)]=[scores,num_ap]
        
name_dict={('PRIMES', True):r"$a'_p$",
    ('PRIMES', False):r'$a_p$',
    ('ALL', True):r"$a'_n$",
    ('ALL', False):r'$a_n$'
          }
comp_scores = pd.DataFrame()
for n_Flag in ["PRIMES","ALL"]:
    for zero_flag in [True, False]:
        key=(n_Flag,zero_flag)
        comp_scores[name_dict[key]]=score_dict[key][0]
        comp_scores[str(key)+'_numap']=score_dict[key][1]
        comp_scores[str(key)+'_percent_coef']=comp_scores[str(key)+'_numap']/comp_scores[str(key)+'_numap'].max().n()
        
comp_scores['percent_coef']=comp_scores[str(key)+'_percent_coef']
comp_scores[['percent_coef',r'$a_n$', r"$a'_n$", r'$a_p$',r"$a'_p$"]].head()          


# In[19]:


fig = px.line(comp_scores,x='percent_coef',y=[r'$a_n$', r"$a'_n$", r'$a_p$',r"$a'_p$"])
fig.update_layout(title=r'$\text{ Accuracy as percentage of coefficients} \text{ increases}$',
                  title_x=0.5,
                  template='plotly_white',
                  xaxis_title=r"$\text{percentage of coefficients}$",
                  yaxis_title=r"$\text{accuracy }$",
                  legend_title='LDA method',
                  autosize=False,
                  width=1000,
                  height=600)
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Predicting Fricke signs for Maass forms with unknown Fricke signs in the dataset

# In[ ]:


## Best prediction dataset is ALL n and not zeroing out primes that divide the level
## Fix that dataset to use for predictions

DF_sample=DF_all.copy()
cols=ALL_n_COLS
groups=['w']

# # Normalize by (-1)^s
DF_sample[cols] = DF_sample[cols].apply(lambda x: x*((-1)^(DF_sample['s'])))

# # Separate into known and unknown Fricke sign
DF_unk=DF_sample.copy()
mask = DF_unk['w']==0    
DF_unk = DF_unk[mask]

mask = DF_sample['w']!=0    
DF_known = DF_sample[mask]

# # Get features and labels
x = DF_known[cols].values # Features are An/ap values
y = DF_known[groups].values # Labels are Fricke sign groups='w'
s = DF_known['s']

# Get test, train data sets - stratify sample on y=Fricke Sign
x_train_val, x_test, y_train_val, y_test, s_train_val, s_test = train_test_split(x, y, s,
                                                                                    test_size=float(0.2),
                                                                                    random_state=random_state, 
                                                                                    stratify=y)

# Break training data into train and validate sets - stratify sample on y=Fricke Sign
x_train, x_val, y_train, y_val, s_train, s_val = train_test_split(x_train_val, y_train_val, s_train_val,
                                                                      test_size=float(0.2), 
                                                                      random_state=random_state, 
                                                                      stratify=y_train_val)


# In[ ]:


# Now what about the ones where we don't know the fricke score?
def predict_fricke(x_train,y_train,s_train,x_val,y_val,s_val,sval,DF_unk):

    if sval == 1:
        sym = 'Odd'
    elif sval == 0:
        sym = 'Even'
    elif sval is None:
        sym = None
    else:
        print(f'Warning symmetry value should be 0 or 1 or None, setting sym=None')
        sym = None  
    lda = lda_maass_sym(x_train,y_train,s_train,x_val,y_val,s_val,sym=sym)
    
    print(f'Number of observation with \n Unknown Fricke Sign, and s={sval}: {len(DF_unk)}')
    
    # Make a prediction
    #cols = MAASS_COLUMNS
    x_unk = DF_unk[cols].values
    scores = lda.predict(x_unk)

    print('Adding column: predicted_w')
    DF_unk['predicted_w'] = scores

    return DF_unk


# In[ ]:


DF_unk_even = predict_fricke(x_train,y_train,s_train,x_val,y_val,s_val,0,DF_unk)
print(" ")
DF_unk_odd = predict_fricke(x_train,y_train,s_train,x_val,y_val,s_val,1,DF_unk)
print(" ")
DF_unk = predict_fricke(x_train,y_train,s_train,x_val,y_val,s_val,None,DF_unk)


# In[ ]:





# ## Plot of Predictions for Unknown Fricke Sign

# In[ ]:


sval=0
DF_unk_sval = predict_fricke(x_train,y_train,s_train,x_val,y_val,s_val,sval,DF_unk)

groups = 'predicted_w'

COLS = PRIME_COLUMNS
my_columns = COLS.copy()
my_columns += [groups]

DF_plot = DF_unk.copy()

mask = DF_plot['s']==sval
DF_plot = DF_plot[mask]

DF_ave = DF_plot[my_columns].groupby(groups).mean()
new_cols = list(DF_ave.index)

dfT_unk=DF_ave.transpose()
dfT_unk.columns = new_cols
dfT_unk['p'] = COLS

groups = 'w'

COLS = PRIME_COLUMNS
my_columns = COLS.copy()
my_columns += [groups]

DF_plot=DF_known.copy()

mask = DF_plot['s']==sval
DF_plot = DF_plot[mask]

DF_ave = DF_plot[my_columns].groupby(groups).mean()
new_cols = list(DF_ave.index)

dfT_known=DF_ave.transpose()
dfT_known.columns = new_cols
dfT_known['p'] = COLS

dfT_combined = pd.merge(dfT_known,dfT_unk,on='p',how='inner')
dfT_combined.rename(columns={'-1_x':'w=-1 known', '1_x':'w=1 known', '-1_y':'w=-1 unknown', '1_y':'w=1 unknown'},inplace=True)
new_cols = [i for i in list(dfT_combined.keys()) if i !='p']


# In[ ]:


for sval in [0,1]:

    if sval==1:
        title1=r'$p \text{ vs average } a_p \text{ for ODD Maass forms with known and unknown Fricke sign}$'
        filename='Predicted_and_Rigor_MaassForms_p_vs_Ap_odd.png'
    if sval==0:
        title1=r'$p \text{ vs average } a_p \text{ for EVEN Maass forms with known and unknown Fricke sign}$'
        filename='Predicted_and_Rigor_MaassForms_p_vs_Ap_even.png'

    fig = px.scatter(dfT_combined,x='p',y=new_cols,
                     symbol_map={'w=-1 known':'circle', 
                                 'w=1 known':'circle', 'w=-1 unk':'diamond', 'w=1 unk':'diamond'},
                     color_discrete_map={'w=-1 known': 'brown',  #'#E66900', 
                                         'w=1 known': 'Darkgreen',  #'#40B0A6' , ##F to A
                                         'w=-1 unknown':'#E1BE6A', 
                                         'w=1 unknown':'Skyblue'})
                        #symbol_sequence=symbols)
    fig.update_layout(title=title1,
                      title_x=0.5,
                      xaxis_title="$p$",
                      template='plotly_white',
                      yaxis_title=r"$\text{ average } a_p$",
                      legend_title='Fricke Sign',
                      autosize=False,
                      width=1000,
                      height=600)

    #fig.write_image(filename)

    fig.show()

