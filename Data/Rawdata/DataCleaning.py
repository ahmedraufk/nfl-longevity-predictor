import numpy as np
import pandas as pd


def fromCSV(fname):
    arr = pd.read_csv(fname,header=0).values
    return arr

def dropCols(arr, drop_list):
    return np.delete(arr, drop_list, axis=1)

def inNFL(arr, idx):
    #Only use players who played at least one snap in the NFL
    new_arr=[]
    for i in range(arr.shape[0]):
        if not pd.isnull(arr[i,idx]):
            new_arr.append(arr[i])
    return np.array(new_arr)

def sinceYear(arr, year, idx):
    #Only use players who got drafted between 2000 and specified year
    new_arr=[]
    for i in range(arr.shape[0]):
        if arr[i,idx] <= year:
            new_arr.append(arr[i])
    return np.array(new_arr)

def seperateByPosition(arr,idx):
    # Seperate into 5 groups: 
    # Quaterbacks, 
    # Skill positions (WR, CB, DB, SS, FS, S), 
    # Trenches (OL, NT, DT, OT, OG, C, DE),
    # Linebackers (ILB, OLB, EDGE, LB)
    # Running backs (RB, FB)
    trenches_str = ['OL', 'NT', 'DT', 'OT', 'OG', 'C', 'DE']
    lb_str = ['ILB', 'OLB', 'EDGE', 'LB']
    skill_str = ['WR', 'CB', 'DB', 'SS', 'FS', 'S']
    rb_str = ['RB', 'FB']

    rb=[]
    qb=[]
    skill=[]
    trenches=[]
    linebackers=[]
    for i in range(arr.shape[0]):
        if arr[i,idx] == 'QB':
            qb.append(arr[i])
        elif arr[i,idx] in rb_str:
            rb.append(arr[i])
        elif arr[i,idx] in lb_str:
            linebackers.append(arr[i])
        elif arr[i,idx] in skill_str:
            skill.append(arr[i])
        elif arr[i,idx] in trenches_str:
            trenches.append(arr[i])
    
    return np.array(qb),np.array(skill),np.array(trenches),np.array(linebackers),np.array(rb)

def dropMissingData(arr, num_nan):
    #If num_nan features is NaN, drop the data point
    new_arr=[]
    for i in range(arr.shape[0]):
        if not np.count_nonzero(pd.isnull(arr[i])) > num_nan:
            new_arr.append(arr[i])
    return np.array(new_arr)

def cleanWithMean(arr):
    for j in range(arr.shape[1]):
        if pd.isnull(arr[:,j]).any():
            m = np.nanmean(arr[:,j],dtype='float32')
            for i in range(arr[:,j].shape[0]):
                if pd.isnull(arr[i,j]):
                    arr[i,j] = round(m,1)
    return arr

def createID(arr, name_idx, year_idx):
    # ids=[]
    new_arr = np.append(arr,np.zeros([len(arr),1]),1)
    for i in range(new_arr.shape[0]):
        new_arr[i,new_arr.shape[1]-1]= arr[i,name_idx]+str(arr[i,year_idx])
    
    # np.concatenate((arr,np.array(ids)),1)
    return new_arr


def cleanLabels(arr, idx):
    new_arr=[]
    for i in range(arr.shape[0]):
        if arr[i,idx] != 'Player':
            new_arr.append(arr[i])
    return np.array(new_arr)

def combineData(data, labels):
    # return pd.merge(data, labels, on='id')
    return data.set_index('id').join(labels.set_index('id'))

def dropNum(arr, idx):
    new_arr = []
    for i in range(arr.shape[0]):
        if '0' not in arr[i,idx]:
            new_arr.append(arr[i])
    return np.array(new_arr)


labels = fromCSV('AllNFLPLAYERS.csv')
labels = dropCols(labels, [0,1,4,6,7,8,9,10,12,13])
labels = dropMissingData(labels, 0)
labels = cleanLabels(labels, 0)

labels = createID(labels, 0, 2)

labels = dropCols(labels, [0,2])



data = fromCSV('combine_data.csv')

data = dropCols(data, [12, 13, 14, 15])

data = inNFL(data, 11)

data = sinceYear(data, 2012, 10)

data = dropMissingData(data, 1)

data = cleanWithMean(data)
# print(data[0])
data = dropNum(data,0)
# print(data[0])

data=createID(data, 0, 10)
data=dropCols(data,[10,11])
# print(data[0])
# print(labels[0])
dataDF = pd.DataFrame(data)
dataDF.columns = ['name','pos','ht','wt','forty','vert','bench','jump', 'cone','shuttle','id']
labelsDF = pd.DataFrame(labels)
labelsDF.columns = ['g', 'pb', 'id']
complete_data = combineData(dataDF,labelsDF)
complete_data=complete_data.fillna(0)

np_data = np.array(complete_data)

# np_data = np.nan_to_num(np_data)

data_qb,data_skill,data_trenches,data_linebackers,data_rb = seperateByPosition(np_data, 1)

data_qb = dropCols(data_qb, [6])


# print('QB: N=' + str(data_qb.shape[0]))
# print('RB: N=' + str(data_rb.shape[0]))
# print('Skill: N=' + str(data_skill.shape[0]))
# print('Trenches: N='+ str(data_trenches.shape[0]))
# print('LB: N=' + str(data_linebackers.shape[0]))

arrs = [data_qb,data_skill,data_trenches,data_linebackers,data_rb]
arr_str = ['data_qb','data_skill','data_trenches','data_linebackers','data_rb']

for i in range(len(arrs)):
    pd.DataFrame(arrs[i]).to_csv(arr_str[i] + '.csv')





