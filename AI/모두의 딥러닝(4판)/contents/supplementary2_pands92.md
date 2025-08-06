## 특별 부록 2


# 데이터 분석을 위한 판다스


[<img src="https://raw.githubusercontent.com/taehojo/taehojo.github.io/master/assets/images/linktocolab.png" align="left"/> ](https://colab.research.google.com/github/taehojo/deeplearning_4th/blob/master/colab/supplementary2_pands92-colab.ipynb)


### 데이터 분석의 필수 라이브러리, 판다스(Pandas) 


판다스(Pandas)는 데이터 분석과 관련된 다양한 기능을 제공하는 파이썬 라이브러리입니다. 데이터를 쉽게 조작하고 다룰 수 있도록 도와주기 때문에 딥러닝, 머신러닝을 스터디 하면 반드시 함께 배우게 됩니다. 판다스 매뉴얼과 판다스 홈페이지의 Cheat Sheet등을 조합해 가장 많이 쓰는 판다스 함수들을 모았습니다. 실전에서 바로 써먹는 92개의 판다스 공식을 확인해 보시기 바랍니다.


## A. 데이터 만들기


```python
# 1. 판다스 라이브러리 불러오기
import pandas as pd
```


```python
# 2. 데이터 프레임 만들기
df = pd.DataFrame(              # df라는 변수에 데이터 프레임을 담아 줍니다.
        {"a" : [4 ,5, 6, 7],    # 열 이름을 지정해 주고 시리즈 형태로 데이터를 저장합니다. 
        "b" : [8, 9, 10, 11],
        "c" : [12, 13, 14, 15]},
        index = [1, 2, 3, 4])   # 인덱스는 1,2,3으로 정해 줍니다.
```


```python
# 3. 데이터 프레임 출력
df
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 4. 데이터의 열 이름을 따로 지정해서 만들기
df = pd.DataFrame(
        [[4, 8, 12],
        [5, 9, 13],
        [6, 10, 14],
        [7, 11, 15]],
        index=[1, 2, 3, 4],
        columns=['a', 'b', 'c'])  # 열 이름을 따로 정해줄 수 있습니다.  
df
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 5. 인덱스가 두 개인 데이터 프레임 만들기
df = pd.DataFrame(
    {"a" : [4 ,5, 6, 7], 
     "b" : [8, 9, 10, 11],
     "c" : [12, 13, 14, 15]},
    index = pd.MultiIndex.from_tuples(      # 인덱스를 튜플로 지정합니다. 
        [('d', 1), ('d', 2), ('e', 1), ('e', 2)],     
        names=['n', 'v']))                  # 인덱스 이름을 지정합니다.
df
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## B. 데이터 정렬하기


```python
# 6. 특정 열의 값을 기준으로 정렬하기
df.sort_values('a', ascending=False)  # ascending=False를 적어주면 역순으로 정렬합니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 7. 열 이름 변경하기 
df.rename(columns = {'c':'d'})  #c 열의 이름을 d로 변경합니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>d</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 8. 인덱스 값 초기화하기
df.reset_index()
```


**출력 결과:**


<div class="output_html">
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
      <th>n</th>
      <th>v</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>d</td>
      <td>1</td>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d</td>
      <td>2</td>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>1</td>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e</td>
      <td>2</td>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 9. 인덱스 순서대로 정렬하기
df.sort_index()
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 10. 특정 열 제거하기
df.drop(columns=['a', 'b']) 
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## C. 행 추출하기


```python
# 11. 맨 위의 행 출력하기
df.head(2)  # 2행을 출력합니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 12. 맨 아래 행 출력하기
df.tail(2)  # 2행을 출력합니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 13. 특정 열의 값을 통해 추출하기 
df[df["a"] > 4]          # a열 중 4보다 큰 값이 있을 경우 해당 행을 추출합니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d</th>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 14. 특정 열에 특정 값이 있을 경우 추출하기
df[df["a"] == 6]       # a열 중 6이 있을 경우 해당 행을 추출합니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 15. 특정 열에 특정 값이 없을 경우 추출하기
df[df["a"] != 5]        # a열 중 5가 없을 경우 해당 행을 추출합니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 16. 특정 열에 특정 숫자가 있는지 확인하기
df[df['a'].isin([4])]  # 원하는 숫자를 리스트([int]) 형식으로 써줍니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 17. 특정 비율로 데이터 샘플링하기
df.sample(frac=0.75)  # 실행할 때마다 정해진 비율 만큼 랜덤하게 추출합니다. 
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>e</th>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 18. 특정 개수 만큼 데이터 샘플링하기
df.sample(n=3)  # 실행할 때마다 n에서 정한 만큼 랜덤하게 추출합니다. 
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>e</th>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 19. 특정 열에서 큰 순서대로 불러오기
df.nlargest(3,'a')   # a열에서 큰 순서대로 3개를 불러와 보겠습니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>d</th>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 20. 특정 열에서 작은 순서대로 불러오기
df.nsmallest(3,'a')   # a열에서 작은 순서대로 3개를 불러와 보겠습니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## D. 열 추출하기


```python
# 21.인덱스의 범위로 불러오기

# 0부터 세므로 첫 번째 줄은 인덱스 0, 4째 줄은 인덱스 3이 됩니다. 
df.iloc[1:4]    # [a:b]의 경우 a 인덱스부터 b-1인덱스까지 불러오라는 의미입니다.# a열에서 큰 순서대로 3개를 불러와 보겠습니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d</th>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 22.첫 인덱스를 지정해 불러오기
df.iloc[2:]    # [a:]는 a인덱스부터 마지막 인덱스까지 불러오라는 의미입니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 23.마지막 인덱스를 지정해 불러오기
df.iloc[:3]    # [:b]는 처음부터 b-1 인덱스까지 불러오라는 의미입니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 24.모든 인덱스를 불러오기
df.iloc[:]    # [:]는 모든 인덱스를 불러오라는 의미입니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 25.특정 열을 지정하여 가져오기
df[['a', 'b']]   # a열과 b열을 가져오라는 의미입니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 26.조건을 만족하는 열 가져오기
df.filter(regex='c')    # 열 이름에 c라는 문자가 포함되어 있으면 출력하라는 의미입니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 27.특정 문자가 포함되지 않는 열 가져오기
df.filter(regex='^(?!c$).*')    # 열 이름에 c라는 문자가 포함되어 있지 않으면 출력하라는 의미입니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## E. 행과 열 추출하기


```python
# 28.특정한 행과 열을 지정해 가져오기
#df.loc[가져올 행,가져올 열]의 형태로 불러옵니다.
df.loc[:, 'a':'c']     # 모든 인덱스에서, a열부터 c열까지를 가져오라는 의미입니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 29. 인덱스로 특정 행과 열 가져오기
df.iloc[0:3, [0, 2]]   # 0 인덱스부터 2인덱스까지, 0번째 열과 2번째 열을 가져오라는 의미입니다. (첫 열이 0번째입니다.)
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>13</td>
    </tr>
    <tr>
      <th>e</th>
      <th>1</th>
      <td>6</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 30. 특정 열에서 조건을 만족하는 행과 열 가져오기
df.loc[df['a'] > 5, ['a', 'c']]  # a열의 값이 5보다 큰 경우의 a열과 c열을 출력하라는 의미입니다.
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 31. 인덱스를 이용해 특정 조건을 만족하는 값 불러오기 
df.iat[1, 2]    # 1번째 인덱스에서 2번째 열 값을 가져옵니다. 
```


**출력 결과:**


```
13
```


## F. 중복 데이터 다루기


```python
# 실습을 위해 중복된 값이 포함된 데이터 프레임을 만들겠습니다.
df = pd.DataFrame(
    {"a" : [4 ,5, 6, 7, 7], 
     "b" : [8, 9, 10, 11, 11],
     "c" : [12, 13, 14, 15, 15]},
    index = pd.MultiIndex.from_tuples(     
        [('d', 1), ('d', 2), ('e', 1), ('e', 2), ('e',3)],     
        names=['n', 'v']))                 
df
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 32. 특정 열에 어떤 값이 몇 개 들어 있는지 알아보기 
df['a'].value_counts()    
```


**출력 결과:**


```
7    2
4    1
5    1
6    1
Name: a, dtype: int64
```


```python
# 33. 데이터 프레임의 행이 몇 개인지 세어보기
len(df)
```


**출력 결과:**


```
5
```


```python
# 34. 데이터 프레임의 행이 몇 개인지, 열이 몇 개인지 세어보기
df.shape
```


**출력 결과:**


```
(5, 3)
```


```python
# 35. 특정 열에 유니크한 값이 몇 개인지 세어보기
df['a'].nunique()
```


**출력 결과:**


```
4
```


```python
# 36. 데이터 프레임의 형태를 한눈에 보기
df.describe()
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.00000</td>
      <td>5.00000</td>
      <td>5.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.80000</td>
      <td>9.80000</td>
      <td>13.80000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.30384</td>
      <td>1.30384</td>
      <td>1.30384</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.00000</td>
      <td>8.00000</td>
      <td>12.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.00000</td>
      <td>9.00000</td>
      <td>13.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.00000</td>
      <td>10.00000</td>
      <td>14.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.00000</td>
      <td>11.00000</td>
      <td>15.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.00000</td>
      <td>11.00000</td>
      <td>15.00000</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 37. 중복된 값 제거하기
df = df.drop_duplicates()
df
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## G. 데이터 파악하기


```python
# 38. 각 열의 합 보기
df.sum()
```


**출력 결과:**


```
a    22
b    38
c    54
dtype: int64
```


```python
# 39. 각 열의 값이 모두 몇 개인지 보기
df.count()
```


**출력 결과:**


```
a    4
b    4
c    4
dtype: int64
```


```python
# 40. 각 열의 중간 값 보기
df.median()
```


**출력 결과:**


```
a     5.5
b     9.5
c    13.5
dtype: float64
```


```python
# 41. 특정 열의 평균 값 보기
df['b'].mean()
```


**출력 결과:**


```
9.5
```


```python
# 42. 각 열의 25%, 75%에 해당하는 수 보기
df.quantile([0.25,0.75])
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.25</th>
      <td>4.75</td>
      <td>8.75</td>
      <td>12.75</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>6.25</td>
      <td>10.25</td>
      <td>14.25</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 43. 각 열의 최솟값 보기
df.min()
```


**출력 결과:**


```
a     4
b     8
c    12
dtype: int64
```


```python
# 44. 각 열의 최댓값 보기
df.max()
```


**출력 결과:**


```
a     7
b    11
c    15
dtype: int64
```


```python
# 45. 각 열의 표준편차 보기
df.std()
```


**출력 결과:**


```
a    1.290994
b    1.290994
c    1.290994
dtype: float64
```


```python
# 46. 데이터 프레임 각 값에 일괄 함수 적용하기 
import numpy as np
df.apply(np.sqrt)  # 제곱근 구하기
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>2.000000</td>
      <td>2.828427</td>
      <td>3.464102</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.236068</td>
      <td>3.000000</td>
      <td>3.605551</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>2.449490</td>
      <td>3.162278</td>
      <td>3.741657</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.645751</td>
      <td>3.316625</td>
      <td>3.872983</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## H. 결측치 다루기


```python
# 넘파이 라이브러리를 이용해 null 값이 들어 있는 데이터 프레임 만들기 

df = pd.DataFrame( 
    {"a" : [4 ,5, 6, np.nan], 
     "b" : [7, 8, np.nan, 9], 
     "c" : [10, np.nan, 11, 12]},    
    index = pd.MultiIndex.from_tuples(
        [('d', 1), ('d', 2), ('e', 1), ('e', 2)],     
        names=['n', 'v']))
df
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6.0</td>
      <td>NaN</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>9.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 47. null 값인지 확인하기
pd.isnull(df)
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 48. null 값이 아닌지를 확인하기
pd.notnull(df)
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 49. null 값이 있는 행 삭제하기
df_notnull = df.dropna()
df_notnull
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d</th>
      <th>1</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 50. null 값을 특정한 값으로 대체하기
df_fillna = df.fillna(13)
df_fillna
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>8.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6.0</td>
      <td>13.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.0</td>
      <td>9.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 51. null 값을 특정한 계산 결과으로 대체하기
df_fillna_mean = df.fillna(df['a'].mean())   # a열의 평균 값으로 대체
df_fillna_mean
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>8.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6.0</td>
      <td>5.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>9.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## I. 새로운 열 만들기


```python
# 새로운 열 만들기 실습을 위한 데이터 프레임 만들기 
df = pd.DataFrame(
    {"a" : [4 ,5, 6, 7], 
     "b" : [8, 9, 10, 11],
     "c" : [12, 13, 14, 15]},
    index = pd.MultiIndex.from_tuples(      # 인덱스를 튜플로 지정합니다. 
        [('d', 1), ('d', 2), ('e', 1), ('e', 2)],     
        names=['n', 'v']))                  # 인덱스 이름을 지정합니다.
df
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>





```python
# 52. 조건에 맞는 새 열 만들기
df['sum'] = df['a']+df['b']+df['c']
df
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>sum</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
      <td>27</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 53. assign()을 이용해 조건에 맞는 새 열 만들기
df = df.assign(multiply=lambda df: df['a']*df['b']*df['c'])    # a,b,c열의 값을 모두 더해 d열을 만들어 줍니다.
df
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>sum</th>
      <th>multiply</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
      <td>24</td>
      <td>384</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
      <td>27</td>
      <td>585</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
      <td>30</td>
      <td>840</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
      <td>33</td>
      <td>1155</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 54. 숫자형 데이터를 구간으로 나누기 
df['qcut'] = pd.qcut(df['a'], 2, labels=["600이하","600이상"])  # a열을 2개로 나누어 각각 새롭게 레이블을 만들라는 의미
df
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>sum</th>
      <th>multiply</th>
      <th>qcut</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
      <td>24</td>
      <td>384</td>
      <td>600이하</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
      <td>27</td>
      <td>585</td>
      <td>600이하</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
      <td>30</td>
      <td>840</td>
      <td>600이상</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
      <td>33</td>
      <td>1155</td>
      <td>600이상</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 55. 기준 값 이하와 이상을 모두 통일시키기
df['clip'] = df['a'].clip(lower=5,upper=6)  # a열에서 5이하는 모두 5로, 6 이상은 모두 6으로 변환
df
```


**출력 결과:**


<div class="output_html">
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
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>sum</th>
      <th>multiply</th>
      <th>qcut</th>
      <th>clip</th>
    </tr>
    <tr>
      <th>n</th>
      <th>v</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">d</th>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
      <td>24</td>
      <td>384</td>
      <td>600이하</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
      <td>27</td>
      <td>585</td>
      <td>600이하</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">e</th>
      <th>1</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
      <td>30</td>
      <td>840</td>
      <td>600이상</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
      <td>33</td>
      <td>1155</td>
      <td>600이상</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 56. 최댓값 불러오기
df.max(axis=0)   #axis=0은 행과 행 비교, axis=1은 열과 열 비교
```


**출력 결과:**


```
a               7
b              11
c              15
sum            33
multiply     1155
qcut        600이상
clip            6
dtype: object
```


```python
# 57. 최솟값 불러오기
df.min(axis=0)    
```


**출력 결과:**


```
a               4
b               8
c              12
sum            24
multiply      384
qcut        600이하
clip            5
dtype: object
```


## J. 행과 열 변환하기


```python
# 열을 행으로, 행을 열로 변형하기

# 실습을 위해 새로운 데이터 만들기
df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})
df
```


**출력 결과:**


<div class="output_html">
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 58. 모든 열을 행으로 변환하기 
pd.melt(df)
```


**출력 결과:**


<div class="output_html">
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
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>B</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>C</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>C</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 59. 하나의 열만 행으로 이동시키기
pd.melt(df, id_vars=['A'], value_vars=['B'])    # A열만 그대로, B열은 행으로 이동시키기
```


**출력 결과:**


<div class="output_html">
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
      <th>A</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>B</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>B</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 60. 여러 개의 열을 행으로 이동시키기
df_melt = pd.melt(df, id_vars=['A'], value_vars=['B','C'])    # A열만 그대로, B열과 C열은 행으로 이동시키기
df_melt
```


**출력 결과:**


<div class="output_html">
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
      <th>A</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>B</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>B</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>C</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>C</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>C</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
#61. 특정 열의 값을 기준으로 새로운 열 만들기 
df_pivot = df_melt.pivot(index='A', columns='variable', values='value')  # A열을 새로운 인덱스로 만들고, B열과 C열을 이에 따라 정리
df_pivot
```


**출력 결과:**


<div class="output_html">
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
      <th>variable</th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>A</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>c</th>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
#62. 원래 데이터 형태로 되돌리기
df_origin = df_pivot.reset_index()  # 인덱스를 리셋
df_origin.columns.name = None       # 인덱스 열의 이름을 초기화
df_origin
```


**출력 결과:**


<div class="output_html">
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## K. 시리즈 데이터 연결하기


```python
# 시리즈 데이터 만들기
s1 = pd.Series(['a', 'b'])
s1
```


**출력 결과:**


```
0    a
1    b
dtype: object
```


```python
# 시리즈 데이터 2
s2 = pd.Series(['c', 'd'])
s2
```


**출력 결과:**


```
0    c
1    d
dtype: object
```


```python
# 63. 시리즈 데이터 합치기
pd.concat([s1, s2])
```


**출력 결과:**


```
0    a
1    b
0    c
1    d
dtype: object
```


```python
# 64. 데이터 병합 시 새로운 인덱스 만들기
pd.concat([s1, s2], ignore_index=True)
```


**출력 결과:**


```
0    a
1    b
2    c
3    d
dtype: object
```


```python
# 65. 계층적 인덱스 추가하고 열 이름 지정하기
pd.concat([s1, s2], 
          keys=['s1', 's2'], 
          names=['Series name', 'Row ID'])
```


**출력 결과:**


```
Series name  Row ID
s1           0         a
             1         b
s2           0         c
             1         d
dtype: object
```


## L. 데이터 프레임 연결하기


```python
# 데이터 프레임 합치기

# 데이터 프레임 1
df1 = pd.DataFrame([['a', 1], 
                    ['b', 2]],
                       columns=['letter', 'number'])
df1
```


**출력 결과:**


<div class="output_html">
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
      <th>letter</th>
      <th>number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 데이터 프레임 2
df2 = pd.DataFrame([['c', 3], 
                    ['d', 4]],
                   columns=['letter', 'number'])
df2
```


**출력 결과:**


<div class="output_html">
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
      <th>letter</th>
      <th>number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 데이터 프레임 3
df3 = pd.DataFrame([['c', 3, 'cat'], 
                    ['d', 4, 'dog']],
                       columns=['letter', 'number', 'animal'])
df3
```


**출력 결과:**


<div class="output_html">
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
      <th>letter</th>
      <th>number</th>
      <th>animal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c</td>
      <td>3</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d</td>
      <td>4</td>
      <td>dog</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
#데이터 프레임 4
df4 = pd.DataFrame([['bird', 'polly'], 
                    ['monkey', 'george']],
                   columns=['animal', 'name'])
df4
```


**출력 결과:**


<div class="output_html">
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
      <th>animal</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bird</td>
      <td>polly</td>
    </tr>
    <tr>
      <th>1</th>
      <td>monkey</td>
      <td>george</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 66. 데이터 프레임 합치기
pd.concat([df1, df2])
```


**출력 결과:**


<div class="output_html">
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
      <th>letter</th>
      <th>number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 67. 열의 수가 다른 두 데이터 프레임 합치기
pd.concat([df1, df3])
```


**출력 결과:**


<div class="output_html">
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
      <th>letter</th>
      <th>number</th>
      <th>animal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>c</td>
      <td>3</td>
      <td>cat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d</td>
      <td>4</td>
      <td>dog</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 68. 함께 공유하는 열만 합치기
pd.concat([df1, df3], join="inner")
```


**출력 결과:**


<div class="output_html">
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
      <th>letter</th>
      <th>number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>c</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 69. 열 이름이 서로 다른 데이터 합치기
pd.concat([df1, df4], axis=1)
```


**출력 결과:**


<div class="output_html">
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
      <th>letter</th>
      <th>number</th>
      <th>animal</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>bird</td>
      <td>polly</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>2</td>
      <td>monkey</td>
      <td>george</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## M. 데이터 병합하기


```python
# 실습을 위한 데이터 프레임 만들기 1
adf = pd.DataFrame({"x1" : ["A","B","C"], 
                    "x2": [1,2,3]})
adf
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 데이터 프레임 만들기 2
bdf = pd.DataFrame({"x1" : ["A","B","D"], 
                    "x3": ["T","F","T"]})
bdf
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>T</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D</td>
      <td>T</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 데이터 프레임 만들기 3
cdf = pd.DataFrame({"x1" : ["B","C","D"], 
                    "x2": [2,3,4]})
cdf
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 70. 왼쪽 열을 축으로 병합하기
pd.merge(adf, bdf, how='left', on='x1')   
# x1을 키로 해서 병합, 왼쪽(adf)를 기준으로
# 왼쪽의 adf에는 D가 없으므로 해당 값은 NaN으로 변환 
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>T</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 71. 오른쪽 열을 축으로 병합하기 
pd.merge(adf, bdf, how='right', on='x1')   
# x1을 키로 해서 병합, 오른쪽(bdf)을 기준으로
# 오른쪽의 bdf에는 C가 없으므로 해당 값은 NaN으로 변환 
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1.0</td>
      <td>T</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D</td>
      <td>NaN</td>
      <td>T</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 72. 공통의 값만 병합
pd.merge(adf, bdf, how='inner', on='x1')
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>T</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 73. 모든 값을 병합
pd.merge(adf, bdf, how='outer', on='x1')
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1.0</td>
      <td>T</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>NaN</td>
      <td>T</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 74. 특정한 열을 비교하여 공통 값이 존재하는 경우만 가져오기
adf[adf.x1.isin(bdf.x1)]  
# adf와 bdf의 특정한 열을 비교하여 공통 값이 존재하는 경우만 가져오기
# adf.x1열과 bdf.x1열은 A,B가 같다. 따라서 adf의 해당 값만 출력
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 75. 공통 값이 존재하는 경우 해당 값을 제외하고 병합하기
adf[~adf.x1.isin(bdf.x1)]  # 해당 값만 빼고 출력
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 76. 공통의 값이 있는 것만 병합
pd.merge(adf, cdf)
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 77. 모두 병합
pd.merge(adf, cdf, how='outer')
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 78. 어디서 병합되었는지 표시하기
pd.merge(adf, cdf, how='outer', indicator=True)
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>2</td>
      <td>both</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>3</td>
      <td>both</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>4</td>
      <td>right_only</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 79. 원하는 병합 만 남기기 
pd.merge(adf, cdf, how='outer', indicator=True).query('_merge == "left_only"')
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>left_only</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 80. merge 컬럼 없애기 
pd.merge(adf, cdf, how='outer', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])
```


**출력 결과:**


<div class="output_html">
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
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
</div>


## N.  데이터 가공하기


```python
# 실습을 위해 데이터 프레임을 만들어 줍니다.
df = pd.DataFrame(
        {"a" : [4 ,5, 6, 7],   # 열 이름을 지정해 주고 시리즈 형태로 데이터를 저장합니다. 
        "b" : [8, 9, 10, 11],
        "c" : [12, 13, 14, 15]},
        index = [1, 2, 3, 4])  # 인덱스는 1,2,3으로 정해 줍니다.
df
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 81. 행 전체를 한 칸 아래로 이동하기
df.shift(1)
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>8.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.0</td>
      <td>9.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.0</td>
      <td>10.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 82. 행 전체를 한 칸 위로 이동하기
df.shift(-1)
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>9.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.0</td>
      <td>10.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.0</td>
      <td>11.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 83. 윗 행부터 누적해서 더하기
df.cumsum()
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>17</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>27</td>
      <td>39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>38</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 84. 새 행과 이전 행을 비교하면서 최댓값을 출력
df.cummax()
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>11</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 85. 새 행과 이전 행을 비교하면서 최솟값을 출력
df.cummin()
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>
</div>


```python
# 85. 윗 행부터 누적해서 곱하기
df.cumprod()
```


**출력 결과:**


<div class="output_html">
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>72</td>
      <td>156</td>
    </tr>
    <tr>
      <th>3</th>
      <td>120</td>
      <td>720</td>
      <td>2184</td>
    </tr>
    <tr>
      <th>4</th>
      <td>840</td>
      <td>7920</td>
      <td>32760</td>
    </tr>
  </tbody>
</table>
</div>
</div>


### O. 그룹별로 집계하기


```python
# 실습을 위해 데이터 불러오기
# 모두의 딥러닝 15장, 주택 가격 예측 데이터를 불러 옵니다.
df = pd.read_csv("../data/house_train.csv")
df
```


**출력 결과:**


<div class="output_html">
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
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
      <th>1455</th>
      <td>1456</td>
      <td>60</td>
      <td>RL</td>
      <td>62.0</td>
      <td>7917</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>175000</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>20</td>
      <td>RL</td>
      <td>85.0</td>
      <td>13175</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>210000</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>70</td>
      <td>RL</td>
      <td>66.0</td>
      <td>9042</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>Shed</td>
      <td>2500</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>266500</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>1459</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>9717</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142125</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>1460</td>
      <td>20</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>147500</td>
    </tr>
  </tbody>
</table>
<p>1460 rows × 81 columns</p>
</div>
</div>


```python
# 87. 그룹 지정 및 그룹별 데이터 수 표시
df.groupby(by="YrSold").size()   # 팔린 연도를 중심으로 그룹을 만든 후, 각 그룹별 수를 표시
```


**출력 결과:**


```
YrSold
2006    314
2007    329
2008    304
2009    338
2010    175
dtype: int64
```


```python
#88. 그룹 지정 후 원하는 컬럼 표시하기 
df.groupby(by="YrSold")['LotArea'].mean()   # 팔린 연도를 중심으로 그룹을 만들 후, 각 그룹별 주차장의 넓이를 표시. 
```


**출력 결과:**


```
YrSold
2006    10489.458599
2007    10863.686930
2008    10587.687500
2009    10294.248521
2010    10220.645714
Name: LotArea, dtype: float64
```


```python
# 89. 밀집도 기준으로 순위 부여하기 
df['SalePrice'].rank(method='dense')   # 각 집 값은 밀집도를 기준으로 몇 번째인가
```


**출력 결과:**


```
0       413.0
1       340.0
2       443.0
3       195.0
4       495.0
        ...  
1455    315.0
1456    416.0
1457    528.0
1458    200.0
1459    222.0
Name: SalePrice, Length: 1460, dtype: float64
```


```python
# 90. 최저 값을 기준으로 순위 부여하기
df['SalePrice'].rank(method='min')    # 각 집 값이 최저 값을 기준으로 몇 번째인가
```


**출력 결과:**


```
0       1072.0
1        909.0
2       1135.0
3        490.0
4       1236.0
         ...  
1455     828.0
1456    1076.0
1457    1285.0
1458     524.0
1459     591.0
Name: SalePrice, Length: 1460, dtype: float64
```


```python
# 91. 순위를 비율로 표시하기  
df['SalePrice'].rank(pct=True)    # 집 값의 순위를 비율로 표시 (0=가장 싼 집, 1=가장 비싼 집)
```


**출력 결과:**


```
0       0.734247
1       0.622603
2       0.777740
3       0.342123
4       0.848973
          ...   
1455    0.569863
1456    0.738356
1457    0.880137
1458    0.358904
1459    0.404795
Name: SalePrice, Length: 1460, dtype: float64
```


```python
# 92. 동일 순위에 대한 처리 방법 정하기 
df['SalePrice'].rank(method='first')   # 순위가 같을 때 순서가 빠른 사람을 상위로 처리하기
```


**출력 결과:**


```
0       1072.0
1        909.0
2       1135.0
3        490.0
4       1236.0
         ...  
1455     836.0
1456    1080.0
1457    1285.0
1458     524.0
1459     591.0
Name: SalePrice, Length: 1460, dtype: float64
```

