
# coding: utf-8

# In[116]:

import os
print (os.getcwd())


# In[117]:

# Uncomment line below for Windows env
# os.chdir(os.getcwd() + '\\data')

# Uncomment line below for Linux env
os.chdir(os.getcwd() + '/data')

# In[118]:

import numpy as np
import pandas as pd


# In[119]:

container=pd.read_csv('container.csv')
print (container)

# In[121]:

print (container)


# In[122]:

item=pd.read_csv("item.csv")
print (item)


# In[123]:

order=pd.read_csv("order.csv")
print (order)


# In[124]:

order.groupby(['ITEM_NUMBER']).size()


# In[125]:

inter=pd.merge(order, item, on='ITEM_NUMBER')
print (inter)


# In[126]:

inter.head()


# In[127]:

container.rename(columns={'CONTAINER_TYPE': 'CARTON_TYPE'}, inplace=True)


# In[128]:

final=pd.merge(inter,container, on='CARTON_TYPE')
print (final)


# In[129]:

final.head()


# In[130]: this generates the vaious combinations for  the 4 set box sizes 

import itertools
combo=list(itertools.combinations(range(13), 4))
print (combo)


# In[131]:

dt=np.dtype('int,int,int,int')
tr=np.array(combo,dtype=dt)


# In[132]:

tr


# In[133]: f0,f1,f2,f3 represent the four boxes in each set

one=tr['f0']
two=tr['f1']
three=tr['f2']
four=tr['f3']


# In[134]:

type(one)


# In[135]:

combin=np.column_stack((one,two,three,four))


# In[136]:

combin=combin.T


# In[137]:

print(combin)


# In[138]:

container1=np.asmatrix(container)


# In[139]:

space=np.ones((4,715))
space=np.asmatrix(space)


# In[140]:This function calculates the wasted space for each item and returns the bestset

def wasted(l,b,h):
    for i in range(715):
        for j in range(4):
            if  (l >container1[combin[j,i],4] or b>container1[combin[j,i],5]or h>container1[combin[j,i],6]):
                space[j,i]=0
            else:
                space[j,i]=((container1[combin[j,i],4]*container1[combin[j,i],5]*container1[combin[j,i],6])-(l*b*h)) 

    space[space == 0] = float("inf")
    cal=space.sum(axis=0)
    cal1=np.asarray(cal)
    best=np.argmin(cal1)
    bestset=combin[:,best]
    return bestset        
        
            
            
            
              


# In[141]:

itemset=list()
for index, row in item.iterrows():
    ans=wasted(row['LENGTH'], row['WIDTH'] , row['HEIGHT'])
    itemset.append(ans)


# In[142]:

itemset


# In[28]:

itemset1=np.array(itemset)
     


# In[29]:

itemset1=itemset1.T


# In[179]:

container1[itemset1[1,1],4]


# In[30]:

itemset1


# In[31]: This function calculates the wasted space for all the items from the bestsets outputed from the previous function(wasted)

def cal(l,b,h):
    left=list()
    for i in range(4):
        ans=((container1[itemset1[i,0],4]*container1[itemset1[i,0],5]*container1[itemset1[i,0],6])-(l*b*h))+ ((container1[itemset1[i,1],4]*container1[itemset1[i,1],5]*container1[itemset1[i,1],6])-(l*b*h))+ ((container1[itemset1[i,2],4]*container1[itemset1[i,2],5]*container1[itemset1[i,2],6])-(l*b*h)) + ((container1[itemset1[i,3],4]*container1[itemset1[i,3],5]*container1[itemset1[i,3],6])-(l*b*h))
        left.append(ans)
                
    return left         


# In[32]:

lspace=list()
for index, row in item.iterrows():
    ans=cal(row['LENGTH'], row['WIDTH'] , row['HEIGHT'])
    lspace.append(ans)


# In[143]:

lspace


# In[144]:

lspace1=np.array(lspace)


# In[145]:

lspace1


# In[146]:

lspace1[lspace1<0.0]


# In[194]:

tot=lspace1.sum(axis=0)


# In[195]:

len(tot)


# In[196]:It prints the index of the set that has minimum wasted space 

m = min(i for i in tot if i > 0)

print("Position:", np.where(tot==m))



# In[199]:

newred=itemset1[:,np.argmin(tot)]


# In[200]:The reduced set

newred





# In[152]:

space[space == 0] = float("inf")


# In[153]:

container


# In[44]:

cal=space.sum(axis=0)
print (cal)


# In[45]:

cal1=np.asarray(cal)


# In[46]:

np.argmin(cal1)


# In[47]:

min(float("inf"),9)


# In[48]:

cal1[:,714]


# In[49]:

type(final)


# In[50]:calculates the wasted space for the existing order table

final1=np.asmatrix(final)
r=final.shape[0]
initial=np.array([])
for i in range(r):
    # spa=((final1[i,15]*final1[i,16]*final1[i,17])-(final1[i,8]*final1[i,9]*final1[i,10]))
    spa=((float(final1[i,15])*float(final1[i,16])*float(final1[i,17]))-(float(final1[i,8])*float(final1[i,9])*float(final1[i,10])))
    initial= np.append(initial,spa)


# In[51]:

sum(initial)


# In[52]:

r


# In[53]:

range(r)


# In[54]:

type(final)


# In[55]:

cdata=final[['LENGTH', 'WIDTH','HEIGHT']]


# In[56]: running the clustering algorithm 

from sklearn import cluster

k = 4
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(cdata)


# In[57]:

labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# In[58]:

labels


# In[201]:

centroids


# In[207]:

centroids1=np.round(centroids)


# In[208]: initial set of dimensions from the clustering procedure

centroids1


# In[202]:

cdata1=final[['LENGTH', 'WIDTH','HEIGHT','FULL_LENGTH','FULL_WIDTH','FULL_HEIGHT']]


# In[203]:

l = final[['LENGTH']].plot() # added 6/15 from previous script
l=np.asarray(l)


# In[204]:

for i, c in enumerate(cdata.columns):
    print (i,c)


# In[72]:

cdata1


# In[64]:


cdata.to_csv('cdata.csv')


# In[226]: The following lines generate the frequency table 

freq=cdata1.groupby(['LENGTH', 'WIDTH','HEIGHT']).count()


# In[227]:

freq.drop('FULL_HEIGHT',axis=1, inplace=True)


# In[228]:

freq.drop('FULL_WIDTH',axis=1, inplace=True)


# In[229]:

freq.rename(columns={'FULL_LENGTH': 'Frequency'}, inplace=True)


# In[240]:

freq=freq.reset_index()


# In[260]:

freq.nlargest(10, 'Frequency')


# In[261]:Items with the top 10 frequencies

freqtable=freq.nlargest(10, 'Frequency')


# In[262]:

freqtable=pd.DataFrame(freqtable)


# In[345]:

cdata[cdata['HEIGHT'] == 14]


# In[325]: The methodology for determined by comparing the centroid table given by the clustering procedure, and the frequency table(freqtable) of top 10 items. 

first=centroids1[0,0]
second=centroids1[1,0]
third=centroids1[2,0]
fourth=centroids1[3,0]
freqtable1=np.matrix(freqtable)
freqtable2=freqtable1[:,0]


# In[326]: For length , each length from the centroid table is subtracted from the length in the top 10 frequency table. The index with the least difference is taken as the new length from the frequency table

diff1=freqtable2-first
diff2=freqtable2-second
diff3=freqtable2-third
diff4=freqtable2-fourth


# In[327]:

diff1=np.abs(diff1)
diff2=np.abs(diff2)
diff3=np.abs(diff3)
diff4=np.abs(diff4)





# In[329]:

set1=freqtable1[np.argmin(diff1),]
set2=freqtable1[np.argmin(diff2),]
set3=freqtable1[np.argmin(diff3),]
set4=freqtable1[np.argmin(diff4),]

# In[328]:

set1


# In[330]: the set with the updated length

sets=np.row_stack((set1,set2,set3,set4))


# In[331]: for width, due to the wide variation in its value, I multiplied the width from the output set by a factor of 2 to encompass the variation

for i in range(0, 4):
    sets[i,1]=sets[i,1]*2
    


# In[355]:

cdata['HEIGHT'].max()


# In[356]: For height , the dimension set with the highest frequency was assigned a height of 6 ,and the set with the lowest frequency a height of 18. This conforms to the erector spec limits as well ensure all the variation is captured.

minimum=sets[:,3].argmin()
maximum=sets[:,3].argmax()
sets[maximum,2]=6
sets[minimum,2]=18

# print sets
# In[333]:This ensures that the length conforms to the erector specs

for i in range(0, 4):
    if sets[i,0]>24:
        sets[i,0]=24
    elif sets[i,0]<13:
        sets[i,0]=13


# In[335]: This ensures that the width conforms to the erector specs

for i in range(0, 4):
    if sets[i,1]>20:
        sets[i,1]=20
    elif sets[i,1]<10:
        sets[i,1]=10


# In[337]:This ensures that the height conforms to the erector specs

for i in range(0, 4):
    if sets[i,2]>18:
        sets[i,2]=18
    elif sets[i,2]<6:
        sets[i,2]=6


# In[357]:The best set 

sets

print(sets)

# In[282]:

final = final.drop(final[final.LENGTH == 0].index)


# In[80]:

final = final.drop(final[final.WIDTH == 0].index)


# In[81]:

final = final.drop(final[final.HEIGHT == 0].index)


# In[378]:

final = final[final.HEIGHT != 0]


# In[379]:

final.head()


# In[408]: The assignment of dimensions for the best set and reduced set was done manually, because it can change depending on the data and what the output is. However the user can use this template of ifelse to assign values.

def newlc(lrow):
    if lrow['LENGTH'] > 0 and lrow['LENGTH']<= 13 and lrow['WIDTH']<=11 and lrow['HEIGHT']<=6 :
        return 13 
    elif lrow['LENGTH'] > 12 and lrow['LENGTH']<= 19 and lrow['WIDTH']<=14 or lrow['HEIGHT']>=6 :
        return 19
    else:
        return 24 
    


# In[409]:

def newwc(wrow):
    if wrow['LENGTH'] > 0 and wrow['LENGTH']<= 13 and wrow['WIDTH']<=10 and wrow['HEIGHT']<=6 :
        return 10
    elif wrow['LENGTH'] > 13 and wrow['LENGTH']<= 19 and wrow['WIDTH']<=14 or wrow['HEIGHT']>=6 :
        return 14
    else: 
        return 20 
 


# In[410]:

def newhc(hrow):
    if hrow['LENGTH'] > 0 and hrow['LENGTH']<= 19 and hrow['WIDTH']<=14 and hrow['HEIGHT']<=6 :
        return 6
    else:
        return 18 
 


# In[411]:

finalcluster=final.copy(deep=True)


# In[412]:

finalredset=final.copy(deep=True)


# In[413]:

finalcluster['newl'] = finalcluster.apply(newlc, axis=1)
finalcluster['neww'] = finalcluster.apply(newwc, axis=1)
finalcluster['newh'] = finalcluster.apply(newhc, axis=1)


# In[414]:

finalcluster1=finalcluster[['LENGTH','WIDTH','HEIGHT','newl','neww','newh']]


# In[415]:

finalcluster1


# In[418]:

def newl(lrow):
    if lrow['LENGTH'] > 0 and lrow['LENGTH']<= 12 and lrow['WIDTH']<=18 and lrow['HEIGHT']<=3 :
        return 12 
    elif lrow['LENGTH'] > 12 and lrow['LENGTH']<= 23 and lrow['WIDTH']<=13 and lrow['HEIGHT']<=8 :
        return 23
    else:
        return 26
   


# In[419]:

def neww(wrow):
    if wrow['LENGTH'] > 0 and wrow['LENGTH']<= 12 and wrow['WIDTH']<=18 and wrow['HEIGHT']<=3 :
        return 18 
    elif wrow['LENGTH'] > 12 and wrow['LENGTH']<= 23 and wrow['WIDTH']<=13 and wrow['HEIGHT']<=6 :
        return 13
    elif wrow['LENGTH'] > 12 and wrow['LENGTH']<= 23 and wrow['WIDTH']<=13 and wrow['HEIGHT']<=8 :
        return 10
    else:
        return 23 
    


# In[420]:

def newh(hrow):
    if hrow['LENGTH'] > 0 and hrow['LENGTH']<= 12 and hrow['WIDTH']<=18 and hrow['HEIGHT']<=3 :
        return 3 
    elif hrow['LENGTH'] > 12 and hrow['LENGTH']<= 23 and hrow['WIDTH']<=13 and hrow['HEIGHT']<=6 :
        return 6 
    elif hrow['LENGTH'] > 12 and hrow['LENGTH']<= 23 and hrow['WIDTH']<=13 and hrow['HEIGHT']<=8 :
        return 8
    else:
        return 4 
    


# In[421]:

finalredset['newl'] = finalredset.apply(newl, axis=1)
finalredset['neww'] = finalredset.apply(neww, axis=1)
finalredset['newh'] = finalredset.apply(newh, axis=1)


# In[422]:

finalredset1=finalredset[['LENGTH','WIDTH','HEIGHT','newl','neww','newh']]


# In[423]:

finalredset1


# In[63]:

newred


# In[425]:

finalr1=np.asmatrix(finalredset1)
r=final.shape[0]


# In[426]:

finalr1


# In[427]: calculates wasted space for reduced set

initial=np.array([])
for i in range(r):
    spa=((finalr1[i,3]*finalr1[i,4]*finalr1[i,5])-(finalr1[i,0]*finalr1[i,1]*finalr1[i,2]))
    initial= np.append(initial,spa)


# In[428]:

sum(initial)


# In[416]:calculates wasted space for best set

finalc1=np.asmatrix(finalcluster1)
r=final.shape[0]
initial=np.array([])
for i in range(r):
    spa=((finalc1[i,3]*finalc1[i,4]*finalc1[i,5])-(finalc1[i,0]*finalc1[i,1]*finalc1[i,2]))
    initial= np.append(initial,spa)


# In[417]:

sum(initial)

#calculates the percentage of different dimensions

perc=finalcluster1.groupby('newl').count()
perc.drop('newh',axis=1, inplace=True)
perc.rename(columns={'neww': 'Frequency'}, inplace=True)
perc=perc.reset_index()
Total = perc['Frequency'].sum()
perc['percentage%'] = perc['Frequency']/Total*100
perc

# [[   18.5    20.     18.   1983. ]
#  [   24.     10.      6.   2855. ]
#  [   13.     20.      6.   2406. ]
#  [   13.     20.      6.   2406. ]]