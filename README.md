# Dealing with Categorical Variables

## Introduction

So far, we have assumed that our predictors (independent variables) are numeric. How can we incorporate categorical data into our regression models as well? This lesson demonstrates how to use an approach called one-hot encoding to do just this.

## Objectives

You will be able to:

- Determine whether variables are categorical or numeric
- Describe why dummy variables are necessary
- Use one-hot encoding to create dummy variables

## Variable Types: Numeric and Categorical

Let's look at the Auto MPG dataset:


```python
import pandas as pd
data = pd.read_csv("auto-mpg.csv")
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
    </tr>
    <tr>
      <th>388</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
    </tr>
    <tr>
      <th>389</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
    </tr>
    <tr>
      <th>390</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
    </tr>
    <tr>
      <th>391</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 9 columns</p>
</div>



We'll also engineer a new feature, `make`, using the `car name` feature:


```python
data["make"] = data["car name"].str.split().apply(lambda x: x[0])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
      <th>make</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
      <td>chevrolet</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
      <td>buick</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
      <td>plymouth</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
      <td>amc</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
      <td>ford</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford mustang gl</td>
      <td>ford</td>
    </tr>
    <tr>
      <th>388</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
      <td>vw pickup</td>
      <td>vw</td>
    </tr>
    <tr>
      <th>389</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage</td>
      <td>dodge</td>
    </tr>
    <tr>
      <th>390</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
      <td>ford ranger</td>
      <td>ford</td>
    </tr>
    <tr>
      <th>391</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
      <td>chevy s-10</td>
      <td>chevy</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 10 columns</p>
</div>



We can look at the `pandas` data types for this dataset using `.info()`:


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 392 entries, 0 to 391
    Data columns (total 10 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   mpg           392 non-null    float64
     1   cylinders     392 non-null    int64  
     2   displacement  392 non-null    float64
     3   horsepower    392 non-null    int64  
     4   weight        392 non-null    int64  
     5   acceleration  392 non-null    float64
     6   model year    392 non-null    int64  
     7   origin        392 non-null    int64  
     8   car name      392 non-null    object 
     9   make          392 non-null    object 
    dtypes: float64(3), int64(5), object(2)
    memory usage: 30.8+ KB


Without digging any further into the _meaning_ of these columns, this print-out tells us that we _can_ use all columns except for `car name` and `make` in a multiple linear regression, without the model crashing.

However a better modeling process would attempt to make a distinction between which of the variables are genuinely representing numbers, and which are actually representing categories.

### Numeric Variables

Numeric variables can be either continuous or discrete.

***Continuous*** variables correspond to "real numbers" in mathematics, and floating point numbers in code. Essentially these variables can have any value on the number line, and usually have a decimal place in their code representation.

***Discrete*** numeric variables typically correspond to "whole numbers" in mathematics, and integers in code. These variables have gaps between their values.

Below we plot `weight`, an example of a continuous variable, and `model year`, an example of a discrete variable, vs. the target, `mpg`.


```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))

data.plot.scatter(x="weight", y="mpg", ax=ax1)
data.plot.scatter(x="model year", y="mpg", ax=ax2);
```


    
![png](index_files/index_9_0.png)
    


You can tell that `model year` is discrete because of the gaps between the vertical lines of values, whereas `weight` is continuous because it's more filled in, like a "cloud", and doesn't have those gaps.

### Categorical Variables

Categorical variables can actually be strings _or_ numbers.

***String*** categorical variables will be fairly obvious due to their data type (`object` in `pandas`). For example, `make` is a categorical variable. It cannot be used in a scatter plot, and it will cause an error if you try to use it in a multiple regression model without additional transformations.

However it can be represented by a bar plot. For example, we can plot the mean `mpg`, grouped by `make`.


```python
fig, ax = plt.subplots(figsize=(12,5))
data.groupby("make").mean('mpg').plot.bar(y='mpg', ax=ax);
```


    
![png](index_files/index_12_0.png)
    


***Discrete*** number categorical variables can be more difficult to spot. For example, `origin` is actually a categorical variable in this dataset, even though it is encoded as a number.


```python
data["origin"].value_counts()
```




    origin
    1    245
    3     79
    2     68
    Name: count, dtype: int64



An `origin` of 1 means the car maker is from the United States, 2 means the car maker is from Europe, and 3 means the car maker is from Asia.


```python
data[["make", "origin"]].groupby("make").first().sort_values("origin")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
    </tr>
    <tr>
      <th>make</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>amc</th>
      <td>1</td>
    </tr>
    <tr>
      <th>plymouth</th>
      <td>1</td>
    </tr>
    <tr>
      <th>pontiac</th>
      <td>1</td>
    </tr>
    <tr>
      <th>hi</th>
      <td>1</td>
    </tr>
    <tr>
      <th>ford</th>
      <td>1</td>
    </tr>
    <tr>
      <th>dodge</th>
      <td>1</td>
    </tr>
    <tr>
      <th>mercury</th>
      <td>1</td>
    </tr>
    <tr>
      <th>chrysler</th>
      <td>1</td>
    </tr>
    <tr>
      <th>oldsmobile</th>
      <td>1</td>
    </tr>
    <tr>
      <th>chevrolet</th>
      <td>1</td>
    </tr>
    <tr>
      <th>chevroelt</th>
      <td>1</td>
    </tr>
    <tr>
      <th>capri</th>
      <td>1</td>
    </tr>
    <tr>
      <th>cadillac</th>
      <td>1</td>
    </tr>
    <tr>
      <th>buick</th>
      <td>1</td>
    </tr>
    <tr>
      <th>chevy</th>
      <td>1</td>
    </tr>
    <tr>
      <th>saab</th>
      <td>2</td>
    </tr>
    <tr>
      <th>renault</th>
      <td>2</td>
    </tr>
    <tr>
      <th>vokswagen</th>
      <td>2</td>
    </tr>
    <tr>
      <th>volkswagen</th>
      <td>2</td>
    </tr>
    <tr>
      <th>peugeot</th>
      <td>2</td>
    </tr>
    <tr>
      <th>opel</th>
      <td>2</td>
    </tr>
    <tr>
      <th>triumph</th>
      <td>2</td>
    </tr>
    <tr>
      <th>mercedes</th>
      <td>2</td>
    </tr>
    <tr>
      <th>mercedes-benz</th>
      <td>2</td>
    </tr>
    <tr>
      <th>volvo</th>
      <td>2</td>
    </tr>
    <tr>
      <th>fiat</th>
      <td>2</td>
    </tr>
    <tr>
      <th>bmw</th>
      <td>2</td>
    </tr>
    <tr>
      <th>audi</th>
      <td>2</td>
    </tr>
    <tr>
      <th>vw</th>
      <td>2</td>
    </tr>
    <tr>
      <th>mazda</th>
      <td>3</td>
    </tr>
    <tr>
      <th>maxda</th>
      <td>3</td>
    </tr>
    <tr>
      <th>honda</th>
      <td>3</td>
    </tr>
    <tr>
      <th>subaru</th>
      <td>3</td>
    </tr>
    <tr>
      <th>toyota</th>
      <td>3</td>
    </tr>
    <tr>
      <th>toyouta</th>
      <td>3</td>
    </tr>
    <tr>
      <th>datsun</th>
      <td>3</td>
    </tr>
    <tr>
      <th>nissan</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



(Looking at the list above, you might notice some typos in the `make` column. We'll address those later!)

Discrete categorical variables like `origin` can be represented with either a scatter plot or a bar plot.


```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))

data.plot.scatter(x="origin", y="mpg", ax=ax1)
data.groupby("origin").mean('mpg').plot.bar(y='mpg', ax=ax2);
```


    
![png](index_files/index_18_0.png)
    


### Identifying Numeric vs. Categorical Variables

In some cases, the data type clearly indicates what kind of variable it should be. A **continuous** variable is essentially always **numeric**, and a **string** variable is essentially always **categorical**.

For **discrete** variables, you need to investigate the values as well as any provided documentation. Then ask yourself:

> Is an increase of 2 in this variable twice as much as an increase of 1?

If 2 is "twice as much" as 1, that means it is reasonable to treat the variable as a numeric discrete variable. If not, the variable should be treated as categorical.

Going back to our examples above:

* `model year`: Is an increase of 2 years twice as much as an increase of 1 year?
  * This seems like a reasonable way to think about the data, so we'll treat it as numeric
* `origin`: Is an increase of 2 (US to Asia) twice as much as an increase of 1 (US to Europe, or Europe to Asia)?
  * It's hard to make sense of this. Treating `origin` as categorical makes a lot more sense

## Transforming Categorical Variables with One-Hot Encoding

In order to use a categorical variable in a model, we'll create multiple ***dummy variables***, one for each category of the categorical variable.

First we'll walk through how this could be done step-by-step, then show you the `get_dummies` method that can achieve this more quickly and efficiently.

### Creating Dummy Variables from Scratch

Let's create a copy of our data that only includes the `origin` column.


```python
origin_df = data[["origin"]].copy()
origin_df.sample(10, random_state=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81</th>
      <td>3</td>
    </tr>
    <tr>
      <th>165</th>
      <td>3</td>
    </tr>
    <tr>
      <th>351</th>
      <td>3</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2</td>
    </tr>
    <tr>
      <th>379</th>
      <td>3</td>
    </tr>
    <tr>
      <th>236</th>
      <td>1</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1</td>
    </tr>
    <tr>
      <th>80</th>
      <td>3</td>
    </tr>
    <tr>
      <th>333</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



The intuition here is, _what if we create a column that just says whether `origin` is equal to 1?_

We might do something like this:


```python
origin_df["origin_us"] = origin_df["origin"] == 1
origin_df.sample(10, random_state=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
      <th>origin_us</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81</th>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>165</th>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>351</th>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>379</th>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>236</th>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>80</th>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>333</th>
      <td>3</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Except, our StatsModels model is expecting _integers_, not _booleans_, so we convert `True` to 1 and `False` to 0:


```python
origin_df["origin_us"] = (origin_df["origin"] == 1).apply(int)
origin_df.sample(10, random_state=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
      <th>origin_us</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>165</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>351</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>379</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>236</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>80</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>333</th>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Then we could repeat the process for European origin and Asian origin:


```python
origin_df["origin_eu"] = (origin_df["origin"] == 2).apply(int)
origin_df["origin_as"] = (origin_df["origin"] == 3).apply(int)
origin_df.sample(10, random_state=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
      <th>origin_us</th>
      <th>origin_eu</th>
      <th>origin_as</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>165</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>351</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>379</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>236</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>333</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Each of these newly-created variables, `origin_us`, `origin_eu`, and `origin_as`, are _dummy_ variables. They are called this because the "real" variable is `origin`, and these are just stand-ins.

The overall process of creating a dummy variable for each value of `origin` is called ***one-hot encoding***. The name "one-hot" comes from digital circuitry, and it means that when you look across all of the dummy variables from one original variable, only one of them should have a value of 1, and the rest should be 0.

### One-Hot Encoding with `pandas`

Instead of creating a new line of code for each value of a column, you can use the `get_dummies` function from `pandas` ([documentation here](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)).


```python
origin_df = data[["origin"]].copy()
origin_df.sample(10, random_state=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81</th>
      <td>3</td>
    </tr>
    <tr>
      <th>165</th>
      <td>3</td>
    </tr>
    <tr>
      <th>351</th>
      <td>3</td>
    </tr>
    <tr>
      <th>119</th>
      <td>2</td>
    </tr>
    <tr>
      <th>379</th>
      <td>3</td>
    </tr>
    <tr>
      <th>236</th>
      <td>1</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1</td>
    </tr>
    <tr>
      <th>80</th>
      <td>3</td>
    </tr>
    <tr>
      <th>333</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
origin_df = pd.get_dummies(origin_df, columns=["origin"], dtype=int)
origin_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_1</th>
      <th>origin_2</th>
      <th>origin_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>389</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>391</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 3 columns</p>
</div>



Some things to note about this version of one-hot encoding:

* The original column (`origin`) has been removed
* The names of the new columns come from the original column name `"origin"` + `_` + the value (1, 2, or 3)
  * If you want these to be more descriptive, consider changing their values _before_ one-hot encoding. For example, you could replace 1, 2, and 3 with "us", "eu", and "as" to be more similar to the example above. This choice is up to you, since these are the names that will appear in the regression results

We can also do one-hot encoding on the entire DataFrame at once, just specifying the columns we consider to be categorical:


```python
pd.get_dummies(data, columns=["origin", "make"], dtype=int)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>car name</th>
      <th>origin_1</th>
      <th>origin_2</th>
      <th>...</th>
      <th>make_renault</th>
      <th>make_saab</th>
      <th>make_subaru</th>
      <th>make_toyota</th>
      <th>make_toyouta</th>
      <th>make_triumph</th>
      <th>make_vokswagen</th>
      <th>make_volkswagen</th>
      <th>make_volvo</th>
      <th>make_vw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>chevrolet chevelle malibu</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>buick skylark 320</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>plymouth satellite</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>amc rebel sst</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>ford torino</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>ford mustang gl</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>vw pickup</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>389</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>dodge rampage</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>ford ranger</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>391</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>chevy s-10</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 48 columns</p>
</div>



Note that you can skip specifying a `columns` argument and `get_dummies` will automatically create dummy variables for all columns with a data type of `object` or `category`. This is a convenient shortcut if your dataset is set up appropriately, but in this case we specified the `columns` because:

1. `car name` is type `object` but we don't actually want to one-hot encode it. We'll drop it before feeding it into the final model, but for now it's there for informational purposes.
2. `origin` is type `int` but we want to treat it as a category and one-hot encode it. If we wanted to change the data type so that `get_dummies` would automatically encode `origin`, we could run `data["origin"] = data["origin"].astype("category")`

## The Dummy Variable Trap

Due to the nature of how dummy variables are created, one variable can be predicted from all of the others. For example, if you know that `origin_1` is 0 and `origin_2` is 0, then you already know that `origin_3` must be 1.

We demonstrate this in code below.


```python
origin_df["origin_1_prediction"] = 1 - origin_df["origin_2"] - origin_df["origin_3"]
origin_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_1</th>
      <th>origin_2</th>
      <th>origin_3</th>
      <th>origin_1_prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>388</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>389</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>390</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>391</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 4 columns</p>
</div>



Our `origin_1_prediction` matches our `origin_1` value 100% of the time:


```python
(origin_df["origin_1_prediction"] == origin_df["origin_1"]).value_counts(normalize=True)
```




    True    1.0
    Name: proportion, dtype: float64



This is known as perfect ***multicollinearity*** and it can be a problem for regression. Multicollinearity will be covered in depth later but the basic idea behind perfect multicollinearity is that you can *perfectly* predict what one variable will be using some combination of the other variables.

When features in a linear regression have perfect multicollinearity due to the algorithm for creating dummy variables, this is known as the ***dummy variable trap***.

Fortunately, the dummy variable trap can be avoided by simply dropping one of the dummy variables. You can do this by subsetting the dataframe manually or, more conveniently, by passing `drop_first=True` into `get_dummies()`: 


```python
pd.get_dummies(data, columns=["origin"], drop_first=True, dtype=int)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>car name</th>
      <th>make</th>
      <th>origin_2</th>
      <th>origin_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>chevrolet chevelle malibu</td>
      <td>chevrolet</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>buick skylark 320</td>
      <td>buick</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>plymouth satellite</td>
      <td>plymouth</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>amc rebel sst</td>
      <td>amc</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>ford torino</td>
      <td>ford</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86</td>
      <td>2790</td>
      <td>15.6</td>
      <td>82</td>
      <td>ford mustang gl</td>
      <td>ford</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52</td>
      <td>2130</td>
      <td>24.6</td>
      <td>82</td>
      <td>vw pickup</td>
      <td>vw</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>389</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84</td>
      <td>2295</td>
      <td>11.6</td>
      <td>82</td>
      <td>dodge rampage</td>
      <td>dodge</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79</td>
      <td>2625</td>
      <td>18.6</td>
      <td>82</td>
      <td>ford ranger</td>
      <td>ford</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>391</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82</td>
      <td>2720</td>
      <td>19.4</td>
      <td>82</td>
      <td>chevy s-10</td>
      <td>chevy</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 11 columns</p>
</div>



Because this dataframe no longer includes `origin_1`, there is no longer enough information to perfectly predict `origin_2` or `origin_3`. The perfect multicollinearity has been eliminated!

## Multiple Regression with One-Hot Encoded Variables

Let's go ahead and create a linear regression model with `weight`, `model year`, and `origin`.


```python
y = data["mpg"]
X = data[["weight", "model year", "origin"]]
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>model year</th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3504</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3693</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3436</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3433</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3449</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>2790</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>388</th>
      <td>2130</td>
      <td>82</td>
      <td>2</td>
    </tr>
    <tr>
      <th>389</th>
      <td>2295</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>390</th>
      <td>2625</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>391</th>
      <td>2720</td>
      <td>82</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 3 columns</p>
</div>




```python
X = pd.get_dummies(X, columns=["origin"], drop_first=True, dtype=int)
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>model year</th>
      <th>origin_2</th>
      <th>origin_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3504</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3693</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3436</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3433</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3449</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>2790</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>2130</td>
      <td>82</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>389</th>
      <td>2295</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>2625</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>391</th>
      <td>2720</td>
      <td>82</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 4 columns</p>
</div>




```python
import statsmodels.api as sm

model = sm.OLS(y, sm.add_constant(X))
results = model.fit()

print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    mpg   R-squared:                       0.819
    Model:                            OLS   Adj. R-squared:                  0.817
    Method:                 Least Squares   F-statistic:                     437.9
    Date:                Wed, 19 Jul 2023   Prob (F-statistic):          3.53e-142
    Time:                        13:59:57   Log-Likelihood:                -1026.1
    No. Observations:                 392   AIC:                             2062.
    Df Residuals:                     387   BIC:                             2082.
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -18.3069      4.017     -4.557      0.000     -26.205     -10.409
    weight        -0.0059      0.000    -22.647      0.000      -0.006      -0.005
    model year     0.7698      0.049     15.818      0.000       0.674       0.866
    origin_2       1.9763      0.518      3.815      0.000       0.958       2.995
    origin_3       2.2145      0.519      4.268      0.000       1.194       3.235
    ==============================================================================
    Omnibus:                       32.293   Durbin-Watson:                   1.251
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               58.234
    Skew:                           0.507   Prob(JB):                     2.26e-13
    Kurtosis:                       4.593   Cond. No.                     7.39e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.39e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.


### Interpreting Model Results

Now, how do we interpret these results?

Just like any other multiple regression model, we can look at the F-statistic p-value to see if it's statistically significant (it is!) and at the adjusted R-Squared to see the proportion of variance explained (around 82%).

The `weight`, and `model year` interpretations are also very similar to previous models we've created. For each increase of 1 lb in weight, we see an associated drop of about 0.006 MPG. For each increase of 1 in model year, we see an associated increase of about 0.77 MPG.

Dropping the first variable affects the interpretation of the other regression coefficients. The dropped category becomes what is known as the ***reference category***. The regression coefficients that result from fitting the remaining variables represent the change *relative* to the reference.

In this regression, an `origin` of 1 (i.e. US origin) is the reference category. This has implications for the interpretation of `const` as well as the other `origin` features.

First, `const` means that all other variables are 0. This means `weight` is 0, `model year` is 0, and `origin` is category 1 (i.e. US origin).

`origin_2` means the difference associated with a car being from a European car maker vs. a US car maker. In other words, compared to US car makers, we see an associated increase of about 2 MPG for European car makers.

`origin_3` is also comparing to US car makers. We see an associated increase of about 2.2 MPG for Asian car makers compared to US car makers.

## Level Up: One-Hot Encoding with Scikit-Learn

The machine learning library scikit-learn also has functionality for one-hot encoding ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)). It is essential to use this approach to one-hot encoding in a predictive machine learning context, and optional to use it in an inferential context like we are currently using.


```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop="first", sparse_output=False)
```

`drop="first"` is equivalent to `drop_first=True` in `pd.get_dummies`. `sparse=False` specifies that we want the result to be a NumPy array rather than a [sparse matrix](https://docs.scipy.org/doc/scipy/reference/sparse.html). Sparse matrices are more efficient in their use of memory space but can't be converted to dataframes as easily.

This approach does not allow you to specify certain columns and pass the entire dataframe in. Instead, you need to create a dataframe with only the column(s) that require one-hot encoding.

For this example we'll select just `origin`.


```python
data_cat = data[["origin"]].copy()
data_cat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>1</td>
    </tr>
    <tr>
      <th>388</th>
      <td>2</td>
    </tr>
    <tr>
      <th>389</th>
      <td>1</td>
    </tr>
    <tr>
      <th>390</th>
      <td>1</td>
    </tr>
    <tr>
      <th>391</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 1 columns</p>
</div>



The result from the scikit-learn one-hot encoder is also not a dataframe.


```python
ohe.fit(data_cat)

ohe.transform(data_cat)
```




    array([[0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 0.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [1., 0.],
           [0., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [1., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.]])



We will need to create a new dataframe ourselves.


```python
data_cat_ohe = pd.DataFrame(
    data=ohe.transform(data_cat),
    columns=[f"origin_{cat}" for cat in ohe.categories_[0][1:]]
)
data_cat_ohe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>origin_2</th>
      <th>origin_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>389</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>391</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 2 columns</p>
</div>



Then we can append the one-hot encoded data back with the numeric data to create an overall X dataframe:


```python
X_sklearn = pd.concat([data[["weight", "model year"]], data_cat_ohe], axis=1)
X_sklearn
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>model year</th>
      <th>origin_2</th>
      <th>origin_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3504</td>
      <td>70</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3693</td>
      <td>70</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3436</td>
      <td>70</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3433</td>
      <td>70</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3449</td>
      <td>70</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>387</th>
      <td>2790</td>
      <td>82</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>388</th>
      <td>2130</td>
      <td>82</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>389</th>
      <td>2295</td>
      <td>82</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>390</th>
      <td>2625</td>
      <td>82</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>391</th>
      <td>2720</td>
      <td>82</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>392 rows × 4 columns</p>
</div>



Then we can plug that dataframe into the model, with the same results as `pd.get_dummies`:


```python
model_2 = sm.OLS(y, sm.add_constant(X_sklearn))
results_2 = model_2.fit()

print(results.params)
print(results_2.params)
```

    const        -18.306944
    weight        -0.005887
    model year     0.769849
    origin_2       1.976306
    origin_3       2.214534
    dtype: float64
    const        -18.306944
    weight        -0.005887
    model year     0.769849
    origin_2       1.976306
    origin_3       2.214534
    dtype: float64


This may seem like a lot of extra work, but the key difference is that the scikit-learn `ohe` object "remembers" the categories that it created, and can apply the same transformation to a future dataset. This is necessary in a machine learning context, but you can consider it optional for now.

## Summary

Great! In this lesson, you learned about categorical variables and how they are different from numeric variables. You also learned how to include them in your multiple linear regression model using dummy variables. You also learned about the dummy variable trap and how it can be avoided.
