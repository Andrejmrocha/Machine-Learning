# Machine-Learning
# Tratamento de dados

Para esta atividade, vamos fazer um tratamento de dados a partir do [dataset](dataset_casas.csv)


```python
import pandas as pd

df = pd.read_csv('dataset_casas.csv')


df.sample(10)
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
      <th>Quartos</th>
      <th>Área Construída</th>
      <th>Banheiros</th>
      <th>Vagas</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>4</td>
      <td>185m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 1.280.000</td>
    </tr>
    <tr>
      <th>115</th>
      <td>4</td>
      <td>262m²</td>
      <td>4</td>
      <td>5+</td>
      <td>R$ 990.000</td>
    </tr>
    <tr>
      <th>194</th>
      <td>4</td>
      <td>376m²</td>
      <td>4</td>
      <td>5+</td>
      <td>R$ 2.900.000</td>
    </tr>
    <tr>
      <th>203</th>
      <td>3</td>
      <td>88m²</td>
      <td>2</td>
      <td>2</td>
      <td>R$ 380.000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>4</td>
      <td>235m²</td>
      <td>4</td>
      <td>2</td>
      <td>R$ 1.230.000</td>
    </tr>
    <tr>
      <th>68</th>
      <td>3</td>
      <td>150m²</td>
      <td>3</td>
      <td>4</td>
      <td>R$ 850.000</td>
    </tr>
    <tr>
      <th>107</th>
      <td>4</td>
      <td>250m²</td>
      <td>4</td>
      <td>5+</td>
      <td>R$ 2.200.000</td>
    </tr>
    <tr>
      <th>113</th>
      <td>4</td>
      <td>276m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 1.400.043</td>
    </tr>
    <tr>
      <th>234</th>
      <td>3</td>
      <td>119m²</td>
      <td>4</td>
      <td>R$ 562.108</td>
      <td>Profissional</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>80m²</td>
      <td>2</td>
      <td>2</td>
      <td>R$ 260.026</td>
    </tr>
  </tbody>
</table>
</div>



### O dataset foi carregado

Iniciando a conversão de tipos com a feature 'Quartos'


```python
# erros='coerce' significa que onde não for possível a conversão será setado NaN
df['Quartos'] = pd.to_numeric(df['Quartos'], errors='coerce', downcast='integer')

df.iloc[85:96]
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
      <th>Quartos</th>
      <th>Área Construída</th>
      <th>Banheiros</th>
      <th>Vagas</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85</th>
      <td>NaN</td>
      <td>420m²</td>
      <td>3</td>
      <td>5+</td>
      <td>R$ 1.200.000</td>
    </tr>
    <tr>
      <th>86</th>
      <td>4.0</td>
      <td>411m²</td>
      <td>5+</td>
      <td>R$ 3.200.000</td>
      <td>Profissional</td>
    </tr>
    <tr>
      <th>87</th>
      <td>4.0</td>
      <td>140m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 590.000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>4.0</td>
      <td>270m²</td>
      <td>4</td>
      <td>5+</td>
      <td>R$ 950.000</td>
    </tr>
    <tr>
      <th>89</th>
      <td>4.0</td>
      <td>223m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 1.300.145</td>
    </tr>
    <tr>
      <th>90</th>
      <td>3.0</td>
      <td>93m²</td>
      <td>4</td>
      <td>3</td>
      <td>R$ 387.900</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4.0</td>
      <td>275m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 1.950.194</td>
    </tr>
    <tr>
      <th>92</th>
      <td>3.0</td>
      <td>93m²</td>
      <td>2</td>
      <td>2</td>
      <td>R$ 300.000</td>
    </tr>
    <tr>
      <th>93</th>
      <td>4.0</td>
      <td>187m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 599.000</td>
    </tr>
    <tr>
      <th>94</th>
      <td>4.0</td>
      <td>272m²</td>
      <td>4</td>
      <td>5+</td>
      <td>R$ 1.800.000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>4.0</td>
      <td>226m²</td>
      <td>4</td>
      <td>R$ 700.000</td>
      <td>IPTU R$ 500</td>
    </tr>
  </tbody>
</table>
</div>



Com uma simples amostra de 10 casas, podemos ver que algumas casas contém NaN na coluna 'Quartos'.
Isso acontece porque algumas casas possuem 5 ou mais quartos, representados por 5+, ao encontrar o '+' durante a conversão ele definiu o valor como NaN já que não seria possível converter 5+ em um int ou float usando a linha de código acima. Por isso vamos usar um método simples, o replace, onde podemos substituir uma cadeia de caracteres por outra:


```python
# Vamos reiniciar nosso df a partir do dataset original
df = pd.read_csv('dataset_casas.csv')

df['Quartos'] = df['Quartos'].str.replace('+', '')
```

Para evitar que aconteça o mesmo com os demais campos já vamos aplicar replace também a Banheiros e Vagas


```python
df['Banheiros'] = df['Banheiros'].str.replace('+', '')
df['Vagas'] = df['Vagas'].str.replace('+', '')

# Vamos verificar a mesma amostra
df.iloc[85:96]
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
      <th>Quartos</th>
      <th>Área Construída</th>
      <th>Banheiros</th>
      <th>Vagas</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85</th>
      <td>5</td>
      <td>420m²</td>
      <td>3</td>
      <td>5</td>
      <td>R$ 1.200.000</td>
    </tr>
    <tr>
      <th>86</th>
      <td>4</td>
      <td>411m²</td>
      <td>5</td>
      <td>R$ 3.200.000</td>
      <td>Profissional</td>
    </tr>
    <tr>
      <th>87</th>
      <td>4</td>
      <td>140m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 590.000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>4</td>
      <td>270m²</td>
      <td>4</td>
      <td>5</td>
      <td>R$ 950.000</td>
    </tr>
    <tr>
      <th>89</th>
      <td>4</td>
      <td>223m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 1.300.145</td>
    </tr>
    <tr>
      <th>90</th>
      <td>3</td>
      <td>93m²</td>
      <td>4</td>
      <td>3</td>
      <td>R$ 387.900</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4</td>
      <td>275m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 1.950.194</td>
    </tr>
    <tr>
      <th>92</th>
      <td>3</td>
      <td>93m²</td>
      <td>2</td>
      <td>2</td>
      <td>R$ 300.000</td>
    </tr>
    <tr>
      <th>93</th>
      <td>4</td>
      <td>187m²</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 599.000</td>
    </tr>
    <tr>
      <th>94</th>
      <td>4</td>
      <td>272m²</td>
      <td>4</td>
      <td>5</td>
      <td>R$ 1.800.000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>4</td>
      <td>226m²</td>
      <td>4</td>
      <td>R$ 700.000</td>
      <td>IPTU R$ 500</td>
    </tr>
  </tbody>
</table>
</div>



Antes de iniciarmos a conversão, precisamos modificar a coluna 'Área Construída' e apagar a unidade 'm²' usando o replace novamente


```python
# Verificando se alguma casa não contém a feature Área Construída em m²
casas_sem_area = df[~df['Área Construída'].str.contains('m²', na=False, case=False)]
casas_sem_area
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
      <th>Quartos</th>
      <th>Área Construída</th>
      <th>Banheiros</th>
      <th>Vagas</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Realizando o replace
df['Área Construída'] = df['Área Construída'].str.replace('m²', '')
df.iloc[39:50]
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
      <th>Quartos</th>
      <th>Área Construída</th>
      <th>Banheiros</th>
      <th>Vagas</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>3</td>
      <td>95</td>
      <td>2</td>
      <td>3</td>
      <td>R$ 395.000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>3</td>
      <td>93</td>
      <td>4</td>
      <td>3</td>
      <td>R$ 387.900</td>
    </tr>
    <tr>
      <th>41</th>
      <td>4</td>
      <td>272</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 1.800.000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>5</td>
      <td>520</td>
      <td>4</td>
      <td>5</td>
      <td>R$ 4.590.005</td>
    </tr>
    <tr>
      <th>43</th>
      <td>3</td>
      <td>122</td>
      <td>2</td>
      <td>4</td>
      <td>R$ 395.000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>3</td>
      <td>318</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 1.050.000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3</td>
      <td>80</td>
      <td>2</td>
      <td>1</td>
      <td>R$ 440.000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>5</td>
      <td>330</td>
      <td>4</td>
      <td>5</td>
      <td>R$ 2.100.000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>4</td>
      <td>215</td>
      <td>4</td>
      <td>5</td>
      <td>R$ 1.019.000</td>
    </tr>
    <tr>
      <th>48</th>
      <td>3</td>
      <td>113</td>
      <td>2</td>
      <td>3</td>
      <td>R$ 465.012</td>
    </tr>
    <tr>
      <th>49</th>
      <td>3</td>
      <td>173</td>
      <td>4</td>
      <td>4</td>
      <td>R$ 1.100.000</td>
    </tr>
  </tbody>
</table>
</div>



Agora é a vez de 'Valor'


```python
# Verificando se alguma casa não contém a feature Valor em R$
casas_sem_valor = df[~df['Valor'].str.startswith('R$', na=False)]
casas_sem_valor

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
      <th>Quartos</th>
      <th>Área Construída</th>
      <th>Banheiros</th>
      <th>Vagas</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>4</td>
      <td>411</td>
      <td>5</td>
      <td>R$ 3.200.000</td>
      <td>Profissional</td>
    </tr>
    <tr>
      <th>95</th>
      <td>4</td>
      <td>226</td>
      <td>4</td>
      <td>R$ 700.000</td>
      <td>IPTU R$ 500</td>
    </tr>
    <tr>
      <th>116</th>
      <td>3</td>
      <td>140</td>
      <td>2</td>
      <td>3</td>
      <td>Direto com o proprietário</td>
    </tr>
    <tr>
      <th>120</th>
      <td>3</td>
      <td>192</td>
      <td>3</td>
      <td>R$ 499.000</td>
      <td>Profissional</td>
    </tr>
    <tr>
      <th>129</th>
      <td>3</td>
      <td>175</td>
      <td>4</td>
      <td>5</td>
      <td>Direto com o proprietário</td>
    </tr>
    <tr>
      <th>173</th>
      <td>5</td>
      <td>339</td>
      <td>4</td>
      <td>5</td>
      <td>Direto com o proprietário</td>
    </tr>
    <tr>
      <th>200</th>
      <td>5</td>
      <td>330</td>
      <td>4</td>
      <td>5</td>
      <td>Direto com o proprietário</td>
    </tr>
    <tr>
      <th>206</th>
      <td>4</td>
      <td>200</td>
      <td>4</td>
      <td>5</td>
      <td>Direto com o proprietário</td>
    </tr>
    <tr>
      <th>234</th>
      <td>3</td>
      <td>119</td>
      <td>4</td>
      <td>R$ 562.108</td>
      <td>Profissional</td>
    </tr>
    <tr>
      <th>252</th>
      <td>4</td>
      <td>270</td>
      <td>4</td>
      <td>5</td>
      <td>Direto com o proprietário</td>
    </tr>
    <tr>
      <th>259</th>
      <td>3</td>
      <td>111</td>
      <td>3</td>
      <td>R$ 499.000</td>
      <td>Profissional</td>
    </tr>
    <tr>
      <th>281</th>
      <td>4</td>
      <td>127</td>
      <td>3</td>
      <td>R$ 506.000</td>
      <td>Profissional</td>
    </tr>
  </tbody>
</table>
</div>



Verificamos que em algumas casas a coluna 'Banheiros' ou 'Vagas' está ausente o que gerou o preenchimento do valor com outro campo, como não será possível utilizar uma casa com algum campo vazio ela será excluída na análise


```python
# excluíndo as casas encontradas sem o valor correto
df = df.drop(casas_sem_valor.index)
```


```python
# Replace no R$
df['Valor'] = df['Valor'].str.replace('R$ ', '').str.replace('.', '')
df.iloc[85:96]
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
      <th>Quartos</th>
      <th>Área Construída</th>
      <th>Banheiros</th>
      <th>Vagas</th>
      <th>Valor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85</th>
      <td>5</td>
      <td>420</td>
      <td>3</td>
      <td>5</td>
      <td>1200000</td>
    </tr>
    <tr>
      <th>87</th>
      <td>4</td>
      <td>140</td>
      <td>4</td>
      <td>4</td>
      <td>590000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>4</td>
      <td>270</td>
      <td>4</td>
      <td>5</td>
      <td>950000</td>
    </tr>
    <tr>
      <th>89</th>
      <td>4</td>
      <td>223</td>
      <td>4</td>
      <td>4</td>
      <td>1300145</td>
    </tr>
    <tr>
      <th>90</th>
      <td>3</td>
      <td>93</td>
      <td>4</td>
      <td>3</td>
      <td>387900</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4</td>
      <td>275</td>
      <td>4</td>
      <td>4</td>
      <td>1950194</td>
    </tr>
    <tr>
      <th>92</th>
      <td>3</td>
      <td>93</td>
      <td>2</td>
      <td>2</td>
      <td>300000</td>
    </tr>
    <tr>
      <th>93</th>
      <td>4</td>
      <td>187</td>
      <td>4</td>
      <td>4</td>
      <td>599000</td>
    </tr>
    <tr>
      <th>94</th>
      <td>4</td>
      <td>272</td>
      <td>4</td>
      <td>5</td>
      <td>1800000</td>
    </tr>
    <tr>
      <th>96</th>
      <td>5</td>
      <td>257</td>
      <td>4</td>
      <td>5</td>
      <td>1700000</td>
    </tr>
    <tr>
      <th>97</th>
      <td>3</td>
      <td>155</td>
      <td>4</td>
      <td>4</td>
      <td>1100000</td>
    </tr>
  </tbody>
</table>
</div>



## Iniciando a conversão de tipo object para int/float


```python
df.dtypes
```




    Quartos            object
    Área Construída    object
    Banheiros          object
    Vagas              object
    Valor              object
    dtype: object




```python
df['Quartos'] = pd.to_numeric(df['Quartos'], errors='coerce', downcast='integer')
df['Área Construída'] = df['Área Construída'].astype(float)
df['Banheiros'] = pd.to_numeric(df['Banheiros'], errors='coerce', downcast='integer')
df['Vagas'] = pd.to_numeric(df['Vagas'], errors='coerce', downcast='integer')
df['Valor'] = df['Valor'].astype(float)
df.dtypes
```




    Quartos               int8
    Área Construída    float64
    Banheiros             int8
    Vagas                 int8
    Valor              float64
    dtype: object



## Salvando em um novo dataset


```python
# Verificando a coluna 'Quartos'
quartos_formato_correto = pd.to_numeric(df['Quartos'], errors='coerce', downcast='integer').notna().all()
print("Quartos em formato correto:", quartos_formato_correto)
```

    Quartos em formato correto: True
    


```python
# Verificando a coluna 'Banheiros'
banheiros_formato_correto = pd.to_numeric(df['Banheiros'], errors='coerce', downcast='integer').notna().all()
print("Banheiros em formato correto:", banheiros_formato_correto)
```

    Banheiros em formato correto: True
    


```python
# Verificando a coluna 'Vagas'
vagas_formato_correto = pd.to_numeric(df['Vagas'], errors='coerce', downcast='integer').notna().all()
print("Vagas em formato correto:", vagas_formato_correto)
```

    Vagas em formato correto: True
    


```python
# Verificando a coluna 'Área Construída'
area_construida_formato_correto = pd.to_numeric(df['Área Construída'], errors='coerce').notna().all()
print("Área Construída em formato correto:", area_construida_formato_correto)
```

    Área Construída em formato correto: True
    


```python
# Verificando a coluna 'Valor'
valor_formato_correto = pd.to_numeric(df['Valor'], errors='coerce').notna().all()
print("Valor em formato correto:", valor_formato_correto)
```

    Valor em formato correto: True
    


```python
df.to_csv('dataset_casas_atualizado.csv', index=False)
```

# Modelo de regressão: previsão de valores de casas em Eusébio/CE

### A partir do dataset [dataset_casas](dataset_casas_atualizado.csv) iremos iniciar nosso modelo de predição de valores de casas.

Features: Quantidade de quartos, Área Construída, Quantidade de banheiros, Quantidade de vagas na garagem.

Label: Valor


```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
```

### Carregando o dataset que será usado


```python
df = pd.read_csv('dataset_casas_atualizado.csv')

x = df[['Quartos', 'Área Construída', 'Banheiros', 'Vagas']].values
y = df[['Valor']].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

```

### Criando a regressão linear


```python
model = LinearRegression()
model.fit(x_train, y_train)
```

### Avaliação - Coeficiente de Determinação


```python
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)


r2
```




    0.9268424613922833



### Criando uma nova casa


```python
nova_casa = np.array([[4, 250, 3, 2]])
```

        4        quartos
        250m2    área construída
        3        banheiros
        2        vagas na garagem

### Realizando a predição


```python
predicao = model.predict(nova_casa)
print(f'O valor da casa é R$ {predicao[0][0]:,.2f}')
```

    O valor da casa é R$ 2,093,616.34
    

### Criando uma nova casa


```python
nova_casa_2 = np.array([[2,150, 2, 1]])
```

        2        quartos
        150m2    área construída
        2        banheiros
        1        vagas na garagem

### Realizando a predição


```python
predicao2 = model.predict(nova_casa_2)
print(f'O valor da casa é R$ {predicao2[0][0]:,.2f}')
```

    O valor da casa é R$ 1,501,065.06
    

### Criando uma nova casa


```python
nova_casa_3 = np.array([[4,350, 5, 3]])
```

        4        quartos
        350m2    área construída
        5        banheiros
        3        vagas na garagem

### Realizando a predição


```python
predicao3 = model.predict(nova_casa_3)
print(f'O valor da casa é R$ {predicao3[0][0]:,.2f}')
```

    O valor da casa é R$ 3,176,195.99

O dataset contém 284 casas, com informações coletadas a partir de anúncios
    
