import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as dm

#task 1
#1
data=pd.read_csv("C:/Coding/Programming/Application_of_ml/mushrooms.csv")
print(data)#reading the dataset
data.columns#reads the columns
#2 z
head=data.head()
print(head)#reads the first five lines
df=pd.DataFrame(data)
print(df.describe())
#3
new_dataset = data[['class','population','habitat']]
new_dataset=data.sample(n=50)#randomly sampling the number of rows
print(new_dataset)


#task2
#1
dataset=pd.read_csv("C:/Coding/Programming/Application_of_ml/Bengaluru_House_Data.csv")
print(dataset)
missing_values = dataset.isnull().sum()#the missing values
print(missing_values)
clean_missing=dataset.dropna()
print(clean_missing)
head=clean_missing.head()
print(head)
new_dataset=clean_missing.dropna(axis=1)#this the dataset with no null values
print(new_dataset)
#2
area_dummies=pd.get_dummies(new_dataset['area_type'],prefix='area_type')
availability_dummies=pd.get_dummies(new_dataset['availability'],prefix='availability')
location_dummies=pd.get_dummies(new_dataset['location'],prefix='location')
size_dummies=pd.get_dummies(new_dataset['size'],prefix='size')
society_dummies=pd.get_dummies(new_dataset['society'],prefix='society')
new_dataset=pd.concat([new_dataset,area_dummies,availability_dummies,location_dummies,size_dummies,society_dummies])
new_dataset=new_dataset.drop(['area_type','availability','location','size','society'],axis=1)
print(new_dataset)
 #3
aggregation_df=pd.DataFrame(new_dataset)
grouped_by_sqft=aggregation_df.groupby('total_sqft')
# agg_data=grouped_by_sqft.agg({"bath","balcony",['sum','mean','count']})
# print(agg_data)


#task4

# merge_dataset=pd.merge(data,dataset, left_on='area_type', right_on='Education', how='inner')
# print(merge_dataset)
employee_dataset=pd.read_csv("C:/Coding/Programming/Application_of_ml/Employee.csv")
concat_dataset=pd.concat([employee_dataset,dataset],axis=0)
print(concat_dataset)
print(concat_dataset.describe())
null_values=concat_dataset.isnull().sum()
print(null_values)

iris=pd.read_csv("C:/Coding/Programming/Application_of_ml/Iris.csv")
print(iris)

#pairplot
sns.countplot(x='Species',data=iris)
plt.title("Count of each species")
plt.show()


# Convert the "species" column to a categorical data type
# iris['species'] = iris['species'].astype('category')

# Pairplot with hue to color data points based on the 'species' column
# sns.countplot(x='Species', data=iris)
# plt.show()
def scatter_3d(x1,x2,x3,y):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection = '3d')

    ax.set_xlabel(x1.name)
    ax.set_ylabel(x2.name)
    ax.set_zlabel(x3.name)
    ax.scatter(x1, x2, x3,c=y,cmap='viridis')
    plt.show()
    x1=iris['SepalLengthCm']
    x2=iris['SepalWidthCm']
    x3=iris['PetalLengthCm']
    y=iris['species_value']
    scatter_3s(x1,x2,x3,y)
    plt.show()

# heatmap = iris.drop(['Id','Species'], axis=1)


#Exploring Numpy
first_arr=dm.array([1,2,3,4,5,6,7,8,9,10])
second_arr=dm.array([11,12,13,14,15,16,17,18,19,20])

#add
add=first_arr+second_arr
#sub
sub=second_arr-first_arr
#multi
multi=first_arr*second_arr
#division
with dm.errstate(divide='ignore',invalid='ignore'):
    division_result=second_arr/first_arr
print(division_result)
print(add)
print(sub)
print(multi)

#array manipulation
#1
matrix_1=first_arr.reshape(2,5)
matrix_2=second_arr.reshape(2,5)
print(matrix_1)
print(matrix_2)
#2
trans_matrix=dm.transpose(matrix_1)
trans_matrix1=dm.transpose(matrix_2)
print(trans_matrix)
print(trans_matrix1)

fallt_array=trans_matrix.flatten()
flattern_array1=trans_matrix1.flatten()

stack_arr=dm.vstack((fallt_array,flattern_array1))
print(stack_arr)

#7 Statistics
house=pd.read_csv("C:/Coding/Programming/Application_of_ml/Bengaluru_House_Data.csv")
# print(house.head())
df=pd.DataFrame(house) 
#mean
range=df['price']
mean1=range.mean()
median1=range.median()
std1=range.std()
mode1=range.mode()[0]#extracts series,we extract first element
print("The mean:",mean1,'median:',median1, 'mode:',mode1,'std1',std1)

#max and min
max1=max(range)
min1=min(range)
print("max:",max1,'min:',min)

#normalize
normalized=(range-mean1)/std1
print("normalized array:",normalized)

#boolean
bool_arr= range > 55
print('boolean array:',bool_arr)

extarcted_arr=range[bool_arr]
print("extracted_element:",extarcted_arr)

#Random Module
random_matrix=dm.random.rand(3,3)
print('random matrix:',random_matrix)

random_int=dm.random.randint(1,101,size=10)
print('random_int:',random_int)

#shuffling
dm.random.shuffle(random_int)
print("shuffling:",random_int)

#task 10
sqrt1=dm.sqrt(random_int)
print('Square root is:',sqrt1)

exp1=dm.exp(sqrt1)
print('exponential of:',exp1)

#11
mat_a=dm.random.rand(3,3)
print('mat_a:',mat_a)

vec_b=dm.random.rand(3,1)
print('vec_b:',vec_b)

result=dm.dot(mat_a,vec_b)
print('Dot product mat_a and vec_b:',result)

#12
two_d=dm.arange(1,10).reshape(3,3)
print('two_d:',two_d)

row_means=dm.mean(two_d,axis=1,keepdims=True)
normalized_matrix=two_d-row_means
print('Normalized Matrix:',normalized_matrix)