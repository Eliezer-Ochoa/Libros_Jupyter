## Pandas
Es una librería de Python especializada en el manejo y análisis de estructuras de datos. pandas es un paquete de Python que proporciona estructuras de datos. Pandas depende de Numpy, la librería que añade un potente tipo matricial a Python. Los principales tipos de datos que pueden representarse con pandas son: Datos tabulares con columnas de tipo heterogéneo con etiquetas en columnas y filas.

## 1. Utilice la siguiente convención de importación


```python
import pandas as pd
```

## 2. Estructura de datos de Pandas
## Series


```python
s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])
print(s)
```

    a    3
    b   -5
    c    7
    d    4
    dtype: int64
    

## 3. DataFrame


```python
 data = {'Country': ['Belgium', 'India', 'Brazil'], 
 'Capital': ['Brussels', 'New Delhi', 'Brasília'],
 'Population': [11190846, 1303171035, 207847528]}
print(data)
```

    {'Country': ['Belgium', 'India', 'Brazil'], 'Capital': ['Brussels', 'New Delhi', 'Brasília'], 'Population': [11190846, 1303171035, 207847528]}
    

## 4. DataFrame


```python
df = pd.DataFrame(data, 
 columns=['Country', 'Capital', 'Population'])
print(df)
```

       Country    Capital  Population
    0  Belgium   Brussels    11190846
    1    India  New Delhi  1303171035
    2   Brazil   Brasília   207847528
    

## 5.  I/O
## Leer y escribir CSV


```python
pd.read_csv('archivoSVC.csv', header=None, nrows=5)

```


```python
 df.to_csv('myDataFrame.csv')
```

## 6. Leer y escribir a excel


```python
pd.read_excel('datos.xlsx')
```


```python
writer = ExcelWriter('ejemplo.xlsx')
```


```python
df.to_excel(writer)
```


```python
writer.save()
```

## 7.Leer multiples hojas del mismo archivo


```python
xlsx = pd.ExcelFile('ejemplo.xlsx')
```


```python
df = pd.read_excel('ejemplo.xlsx', index_col = 0)
print(df)
```

## 8. Pidiendo ayuda


```python
help(pd.Series.loc)
```

    Help on property:
    
        Access a group of rows and columns by label(s) or a boolean array.
        
        ``.loc[]`` is primarily label based, but may also be used with a
        boolean array.
        
        Allowed inputs are:
        
        - A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
          interpreted as a *label* of the index, and **never** as an
          integer position along the index).
        - A list or array of labels, e.g. ``['a', 'b', 'c']``.
        - A slice object with labels, e.g. ``'a':'f'``.
        
          .. warning:: Note that contrary to usual python slices, **both** the
              start and the stop are included
        
        - A boolean array of the same length as the axis being sliced,
          e.g. ``[True, False, True]``.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above)
        
        See more at :ref:`Selection by Label <indexing.label>`
        
        Raises
        ------
        KeyError:
            when any items are not found
        
        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.iloc : Access group of rows and columns by integer position(s).
        DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the
            Series/DataFrame.
        Series.loc : Access group of values using labels.
        
        Examples
        --------
        **Getting values**
        
        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...      index=['cobra', 'viper', 'sidewinder'],
        ...      columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8
        
        Single label. Note this returns the row as a Series.
        
        >>> df.loc['viper']
        max_speed    4
        shield       5
        Name: viper, dtype: int64
        
        List of labels. Note using ``[[]]`` returns a DataFrame.
        
        >>> df.loc[['viper', 'sidewinder']]
                    max_speed  shield
        viper               4       5
        sidewinder          7       8
        
        Single label for row and column
        
        >>> df.loc['cobra', 'shield']
        2
        
        Slice with labels for row and single label for column. As mentioned
        above, note that both the start and stop of the slice are included.
        
        >>> df.loc['cobra':'viper', 'max_speed']
        cobra    1
        viper    4
        Name: max_speed, dtype: int64
        
        Boolean list with the same length as the row axis
        
        >>> df.loc[[False, False, True]]
                    max_speed  shield
        sidewinder          7       8
        
        Conditional that returns a boolean Series
        
        >>> df.loc[df['shield'] > 6]
                    max_speed  shield
        sidewinder          7       8
        
        Conditional that returns a boolean Series with column labels specified
        
        >>> df.loc[df['shield'] > 6, ['max_speed']]
                    max_speed
        sidewinder          7
        
        Callable that returns a boolean Series
        
        >>> df.loc[lambda df: df['shield'] == 8]
                    max_speed  shield
        sidewinder          7       8
        
        **Setting values**
        
        Set value for all items matching the list of labels
        
        >>> df.loc[['viper', 'sidewinder'], ['shield']] = 50
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4      50
        sidewinder          7      50
        
        Set value for an entire row
        
        >>> df.loc['cobra'] = 10
        >>> df
                    max_speed  shield
        cobra              10      10
        viper               4      50
        sidewinder          7      50
        
        Set value for an entire column
        
        >>> df.loc[:, 'max_speed'] = 30
        >>> df
                    max_speed  shield
        cobra              30      10
        viper              30      50
        sidewinder         30      50
        
        Set value for rows matching callable condition
        
        >>> df.loc[df['shield'] > 35] = 0
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       0
        sidewinder          0       0
        
        **Getting values on a DataFrame with an index that has integer labels**
        
        Another example using integers for the index
        
        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...      index=[7, 8, 9], columns=['max_speed', 'shield'])
        >>> df
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8
        
        Slice with integer labels for rows. As mentioned above, note that both
        the start and stop of the slice are included.
        
        >>> df.loc[7:9]
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8
        
        **Getting values with a MultiIndex**
        
        A number of examples using a DataFrame with a MultiIndex
        
        >>> tuples = [
        ...    ('cobra', 'mark i'), ('cobra', 'mark ii'),
        ...    ('sidewinder', 'mark i'), ('sidewinder', 'mark ii'),
        ...    ('viper', 'mark ii'), ('viper', 'mark iii')
        ... ]
        >>> index = pd.MultiIndex.from_tuples(tuples)
        >>> values = [[12, 2], [0, 4], [10, 20],
        ...         [1, 4], [7, 1], [16, 36]]
        >>> df = pd.DataFrame(values, columns=['max_speed', 'shield'], index=index)
        >>> df
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36
        
        Single label. Note this returns a DataFrame with a single index.
        
        >>> df.loc['cobra']
                 max_speed  shield
        mark i          12       2
        mark ii          0       4
        
        Single index tuple. Note this returns a Series.
        
        >>> df.loc[('cobra', 'mark ii')]
        max_speed    0
        shield       4
        Name: (cobra, mark ii), dtype: int64
        
        Single label for row and column. Similar to passing in a tuple, this
        returns a Series.
        
        >>> df.loc['cobra', 'mark i']
        max_speed    12
        shield        2
        Name: (cobra, mark i), dtype: int64
        
        Single tuple. Note using ``[[]]`` returns a DataFrame.
        
        >>> df.loc[[('cobra', 'mark ii')]]
                       max_speed  shield
        cobra mark ii          0       4
        
        Single tuple for the index with a single label for the column
        
        >>> df.loc[('cobra', 'mark i'), 'shield']
        2
        
        Slice from index tuple to single label
        
        >>> df.loc[('cobra', 'mark i'):'viper']
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36
        
        Slice from index tuple to index tuple
        
        >>> df.loc[('cobra', 'mark i'):('viper', 'mark ii')]
                            max_speed  shield
        cobra      mark i          12       2
                   mark ii          0       4
        sidewinder mark i          10      20
                   mark ii          1       4
        viper      mark ii          7       1
    
    

## 9. Selección

## Obteniendo un elemento del arreglo b


```python
s['b']
```




    -5



## 10. Obtener subconjunto de un DataFrame


```python
df[1:] 
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
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Brazil</td>
      <td>Brasília</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>



## 11. Selección, indexación booleana y configuración
#### Seleccionar un valor por fila y columna (por posición


```python
 df.iloc[0], [0]
```




    (Country        Belgium
     Capital       Brussels
     Population    11190846
     Name: 0, dtype: object, [0])




```python
df.iat[0,0]
```




    'Belgium'



## 12. Seleccionar un valor por label de fila y label de columna (por label)


```python
#df.loc([0], ['Country'])

```


```python
#df.at([0], ['Country'])
```

## 13. Seleccionar una sola fila de subconjunto de filas (Por label, y posición)


```python
df.iloc[2]
```




    Country          Brazil
    Capital        Brasília
    Population    207847528
    Name: 2, dtype: object




```python
#df.ix[1,'Capital']
```

## 14.Indexación booleana


```python
s[~(s > 1)] 
```




    b   -5
    dtype: int64




```python
s[(s < -1) | (s > 2)]
```




    a    3
    b   -5
    c    7
    d    4
    dtype: int64




```python
df[df['Population']>1200000000] 
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
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
  </tbody>
</table>
</div>



## 15. Ajustes


```python
s['a'] = 6
```

## 16. Leer y escribir a SQL or tabla de base de datos


```python
from sqlalchemy import create_engine
import pandas as pd
```


```python
engine = create_engine('sqlite:///:memory:')
```


```python
pd.read_sql("SELECT *FROM Personas;", engine)
```


    ---------------------------------------------------------------------------

    OperationalError                          Traceback (most recent call last)

    ~\Anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _execute_context(self, dialect, constructor, statement, parameters, *args)
       1248                     self.dialect.do_execute(
    -> 1249                         cursor, statement, parameters, context
       1250                     )
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\engine\default.py in do_execute(self, cursor, statement, parameters, context)
        579     def do_execute(self, cursor, statement, parameters, context=None):
    --> 580         cursor.execute(statement, parameters)
        581 
    

    OperationalError: no such table: Personas

    
    The above exception was the direct cause of the following exception:
    

    OperationalError                          Traceback (most recent call last)

    <ipython-input-38-6e1b06da8d9a> in <module>
    ----> 1 pd.read_sql("SELECT *FROM Personas;", engine)
    

    ~\Anaconda3\lib\site-packages\pandas\io\sql.py in read_sql(sql, con, index_col, coerce_float, params, parse_dates, columns, chunksize)
        434             coerce_float=coerce_float,
        435             parse_dates=parse_dates,
    --> 436             chunksize=chunksize,
        437         )
        438 
    

    ~\Anaconda3\lib\site-packages\pandas\io\sql.py in read_query(self, sql, index_col, coerce_float, parse_dates, params, chunksize)
       1216         args = _convert_params(sql, params)
       1217 
    -> 1218         result = self.execute(*args)
       1219         columns = result.keys()
       1220 
    

    ~\Anaconda3\lib\site-packages\pandas\io\sql.py in execute(self, *args, **kwargs)
       1085     def execute(self, *args, **kwargs):
       1086         """Simple passthrough to SQLAlchemy connectable"""
    -> 1087         return self.connectable.execute(*args, **kwargs)
       1088 
       1089     def read_table(
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\engine\base.py in execute(self, statement, *multiparams, **params)
       2177 
       2178         connection = self._contextual_connect(close_with_result=True)
    -> 2179         return connection.execute(statement, *multiparams, **params)
       2180 
       2181     def scalar(self, statement, *multiparams, **params):
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\engine\base.py in execute(self, object_, *multiparams, **params)
        980         """
        981         if isinstance(object_, util.string_types[0]):
    --> 982             return self._execute_text(object_, multiparams, params)
        983         try:
        984             meth = object_._execute_on_connection
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _execute_text(self, statement, multiparams, params)
       1153             parameters,
       1154             statement,
    -> 1155             parameters,
       1156         )
       1157         if self._has_events or self.engine._has_events:
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _execute_context(self, dialect, constructor, statement, parameters, *args)
       1251         except BaseException as e:
       1252             self._handle_dbapi_exception(
    -> 1253                 e, statement, parameters, cursor, context
       1254             )
       1255 
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _handle_dbapi_exception(self, e, statement, parameters, cursor, context)
       1471                 util.raise_from_cause(newraise, exc_info)
       1472             elif should_wrap:
    -> 1473                 util.raise_from_cause(sqlalchemy_exception, exc_info)
       1474             else:
       1475                 util.reraise(*exc_info)
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_from_cause(exception, exc_info)
        396     exc_type, exc_value, exc_tb = exc_info
        397     cause = exc_value if exc_value is not exception else None
    --> 398     reraise(type(exception), exception, tb=exc_tb, cause=cause)
        399 
        400 
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\util\compat.py in reraise(tp, value, tb, cause)
        150             value.__cause__ = cause
        151         if value.__traceback__ is not tb:
    --> 152             raise value.with_traceback(tb)
        153         raise value
        154 
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _execute_context(self, dialect, constructor, statement, parameters, *args)
       1247                 if not evt_handled:
       1248                     self.dialect.do_execute(
    -> 1249                         cursor, statement, parameters, context
       1250                     )
       1251         except BaseException as e:
    

    ~\Anaconda3\lib\site-packages\sqlalchemy\engine\default.py in do_execute(self, cursor, statement, parameters, context)
        578 
        579     def do_execute(self, cursor, statement, parameters, context=None):
    --> 580         cursor.execute(statement, parameters)
        581 
        582     def do_execute_no_params(self, cursor, statement, context=None):
    

    OperationalError: (sqlite3.OperationalError) no such table: Personas
    [SQL: SELECT *FROM Personas;]
    (Background on this error at: http://sqlalche.me/e/e3q8)



```python
pd.to_sql('myDf', engine)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-39-4c74bda5807e> in <module>
    ----> 1 pd.to_sql('myDf', engine)
    

    ~\Anaconda3\lib\site-packages\pandas\__init__.py in __getattr__(name)
        212 
        213             return Panel
    --> 214         raise AttributeError("module 'pandas' has no attribute '{}'".format(name))
        215 
        216 
    

    AttributeError: module 'pandas' has no attribute 'to_sql'


## 17. Borrando


```python
 s.drop(['a', 'c']) 
```




    b   -5
    d    4
    dtype: int64




```python
 #df.drop('Country', axis=0)
```

## 18. Ordenar y rank


```python
df.sort_index()
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
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <td>1</td>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Brazil</td>
      <td>Brasília</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>




```python
 df.sort_values(by='Country')
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
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Brazil</td>
      <td>Brasília</td>
      <td>207847528</td>
    </tr>
    <tr>
      <td>1</td>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.rank() 
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
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



## 19. Recuperación de información de series/dataframes

## Información básica


```python
df.shape
```




    (3, 3)




```python
df.index
```




    RangeIndex(start=0, stop=3, step=1)




```python
df.columns
```




    Index(['Country', 'Capital', 'Population'], dtype='object')




```python
df.info() 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 3 columns):
    Country       3 non-null object
    Capital       3 non-null object
    Population    3 non-null int64
    dtypes: int64(1), object(2)
    memory usage: 200.0+ bytes
    


```python
df.count()
```




    Country       3
    Capital       3
    Population    3
    dtype: int64



## 20. Summary


```python
df.sum() 
```




    Country              BelgiumIndiaBrazil
    Capital       BrusselsNew DelhiBrasília
    Population                   1522209409
    dtype: object




```python
df.cumsum()
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
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <td>1</td>
      <td>BelgiumIndia</td>
      <td>BrusselsNew Delhi</td>
      <td>1314361881</td>
    </tr>
    <tr>
      <td>2</td>
      <td>BelgiumIndiaBrazil</td>
      <td>BrusselsNew DelhiBrasília</td>
      <td>1522209409</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.min()
df.max()
```




    Country            India
    Capital        New Delhi
    Population    1303171035
    dtype: object




```python
#df.idxmin()/df.idxmax()
```


```python
 df.describe()
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
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>5.074031e+08</td>
    </tr>
    <tr>
      <td>std</td>
      <td>6.961346e+08</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.119085e+07</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.095192e+08</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>2.078475e+08</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>7.555093e+08</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.303171e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.mean()
```




    Population    5.074031e+08
    dtype: float64




```python
df.median()
```




    Population    207847528.0
    dtype: float64



## 21. Aplicando funciones


```python
 f = lambda x: x*2
```


```python
 df.apply(f)
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
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>BelgiumBelgium</td>
      <td>BrusselsBrussels</td>
      <td>22381692</td>
    </tr>
    <tr>
      <td>1</td>
      <td>IndiaIndia</td>
      <td>New DelhiNew Delhi</td>
      <td>2606342070</td>
    </tr>
    <tr>
      <td>2</td>
      <td>BrazilBrazil</td>
      <td>BrasíliaBrasília</td>
      <td>415695056</td>
    </tr>
  </tbody>
</table>
</div>




```python
 df.applymap(f)
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
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>BelgiumBelgium</td>
      <td>BrusselsBrussels</td>
      <td>22381692</td>
    </tr>
    <tr>
      <td>1</td>
      <td>IndiaIndia</td>
      <td>New DelhiNew Delhi</td>
      <td>2606342070</td>
    </tr>
    <tr>
      <td>2</td>
      <td>BrazilBrazil</td>
      <td>BrasíliaBrasília</td>
      <td>415695056</td>
    </tr>
  </tbody>
</table>
</div>



## 22. Alineamiento de datos
### Alineamiento de datos internos


```python
s3 = pd.Series([7, -2, 3], index=['a', 'c', 'd'])
print(s3)
```

    a    7
    c   -2
    d    3
    dtype: int64
    


```python
s + s3
```




    a    13.0
    b     NaN
    c     5.0
    d     7.0
    dtype: float64



## 23. Operaciones aritméticas con métodos llenos


```python
 s.add(s3, fill_value=0)
```




    a    13.0
    b    -5.0
    c     5.0
    d     7.0
    dtype: float64




```python
s.sub(s3, fill_value=2)
```




    a   -1.0
    b   -7.0
    c    9.0
    d    1.0
    dtype: float64




```python
 s.div(s3, fill_value=4)
```




    a    0.857143
    b   -1.250000
    c   -3.500000
    d    1.333333
    dtype: float64




```python
s.mul(s3, fill_value=3)
```




    a    42.0
    b   -15.0
    c   -14.0
    d    12.0
    dtype: float64



# 24. Libro número dos de trucos de pandas
## Remodelación de datos
#### Creación de los datos


```python
data = {'Date': ['2016-03-01','2016-03-02','2016-03-01','2016-03-03','2016-03-02','2016-03-03'],
        'Type': ['a','b','c','a','a','c'],
        'Value': [11.432,13.031,20.784,99.906,1.303,20.784]}

df2 = pd.DataFrame(data,
 columns=['Date', 'Type', 'Value'])
```

## 25. pivote


```python
df3 = df2.pivot(index='Date', columns='Type',values='Value')
```

## 26. pilar/desapilar


```python
stacked = df4.stack()
stacked.unstack()
```

## 27. Derretir


```python
pd.melt(df2,
        id_vars=["Date"],
        value_vars=["Type","Value"],
        value_name="Observations")
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
      <th>Date</th>
      <th>variable</th>
      <th>Observations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2016-03-01</td>
      <td>Type</td>
      <td>a</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2016-03-02</td>
      <td>Type</td>
      <td>b</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2016-03-01</td>
      <td>Type</td>
      <td>c</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016-03-03</td>
      <td>Type</td>
      <td>a</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2016-03-02</td>
      <td>Type</td>
      <td>a</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2016-03-03</td>
      <td>Type</td>
      <td>c</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2016-03-01</td>
      <td>Value</td>
      <td>11.432</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2016-03-02</td>
      <td>Value</td>
      <td>13.031</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2016-03-01</td>
      <td>Value</td>
      <td>20.784</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2016-03-03</td>
      <td>Value</td>
      <td>99.906</td>
    </tr>
    <tr>
      <td>10</td>
      <td>2016-03-02</td>
      <td>Value</td>
      <td>1.303</td>
    </tr>
    <tr>
      <td>11</td>
      <td>2016-03-03</td>
      <td>Value</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>



## 28. Column-index,series pairs


```python
df.iteritems()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-9-9095ee47606f> in <module>
    ----> 1 df.iteritems()
    

    NameError: name 'df' is not defined


## 29. Row-index,series pairs


```python
df.iterrows()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-10-39972d0c1263> in <module>
    ----> 1 df.iterrows()
    

    NameError: name 'df' is not defined


## 30. Advanced Indexing
## Selecting


```python
df3.loc[:,(df3>1).any()]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-b090aa008831> in <module>
    ----> 1 df3.loc[:,(df3>1).any()]
    

    NameError: name 'df3' is not defined


## 31. Select cols with NaN



```python
df3.loc[:,df3.isnull().any()]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-12-5551b408dd6c> in <module>
    ----> 1 df3.loc[:,df3.isnull().any()]
    

    NameError: name 'df3' is not defined


## 32. Select cols without NAN


```python
df3.loc[:,df3.notnull().all()]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-13-df243a903b6b> in <module>
    ----> 1 df3.loc[:,df3.notnull().all()]
    

    NameError: name 'df3' is not defined


## 33. Find same elements


```python
df[(df.Country.isin(df2.Type))]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-14-7e10ce95c245> in <module>
    ----> 1 df[(df.Country.isin(df2.Type))]
    

    NameError: name 'df' is not defined


## 34. Filter on values



```python
df3.filter(items=“a”,“b”])
```


      File "<ipython-input-15-c9a5e7c523ce>", line 1
        df3.filter(items=“a”,“b”])
                           ^
    SyntaxError: invalid character in identifier
    


## 35. Select specific elements



```python
df.select(lambda x: not x%5)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-16-ada00df27c6f> in <module>
    ----> 1 df.select(lambda x: not x%5)
    

    NameError: name 'df' is not defined


## 36. Subset the data



```python
s.where(s > 0)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-17-97e5678e4f40> in <module>
    ----> 1 s.where(s > 0)
    

    NameError: name 's' is not defined


## 37. Query DataFrame



```python
df6.query('second > first')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-30-b468d82f4321> in <module>
    ----> 1 df6.query('second > first')
    

    NameError: name 'df6' is not defined


## 38. Set the index



```python
df.set_index('Country')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-29-1daea7912c21> in <module>
    ----> 1 df.set_index('Country')
    

    NameError: name 'df' is not defined


## 39. Reset the index



```python
df4 = df.reset_index()
print(df4)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-28-2883de234e79> in <module>
    ----> 1 df4 = df.reset_index()
          2 print(df4)
    

    NameError: name 'df' is not defined


## 40. Renamme DataFrame


```python

df = df.rename(index=str,
               columns={"Country":"cntry",
                        "Capital":"cptl",
                        "Population":"ppltn"})
print(df)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-27-073ec4f0a0ac> in <module>
    ----> 1 df = df.rename(index=str,
          2                columns={"Country":"cntry",
          3                         "Capital":"cptl",
          4                         "Population":"ppltn"})
          5 print(df)
    

    NameError: name 'df' is not defined


## 41. Reindexing


```python
s2 = s.reindex(['a','c','d','e','b'])
print(s2)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-26-03f167dbcdcd> in <module>
    ----> 1 s2 = s.reindex(['a','c','d','e','b'])
          2 print(s2)
    

    NameError: name 's' is not defined


## 42. Forward Filling


```python
df.reindex(range(4),
           method='Ffill')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-25-5eb4c61bb2b3> in <module>
    ----> 1 df.reindex(range(4),
          2            method='Ffill')
    

    NameError: name 'df' is not defined


## 43. Backward Filling


```python
s3 = s.reindex(range(5),
               method='bfill')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-24-08ea604afe6a> in <module>
    ----> 1 s3 = s.reindex(range(5),
          2                method='bfill')
    

    NameError: name 's' is not defined


## 44. Multilndexing


```python
arrays = [np.array([1,2,3]),
          np.array([5,4,3])]
df5= pd.DataFrame(np.random.rand(3, 2), index=arrays)
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples,
                                  names=['first', 'second'])
df6= pd.DataFrame(np.random.rand(3, 2), index=index)
df2.set_index(["Date", "Type"])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-23-bad635b78d5c> in <module>
    ----> 1 arrays = [np.array([1,2,3]),
          2           np.array([5,4,3])]
          3 df5= pd.DataFrame(np.random.rand(3, 2), index=arrays)
          4 tuples = list(zip(*arrays))
          5 index = pd.MultiIndex.from_tuples(tuples,
    

    NameError: name 'np' is not defined


## 45. Duplicate Data
## Return unique values


```python
s3.unique()
print(s3)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-22-e1701c0a89b2> in <module>
    ----> 1 s3.unique()
          2 print(s3)
    

    NameError: name 's3' is not defined


## 46. Check duplicates 


```python
df2.duplicated('Type')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-21-d5559bf77f04> in <module>
    ----> 1 df2.duplicated('Type')
    

    NameError: name 'df2' is not defined


## 47. Drop duplicates


```python
df2.drop_duplicates('Type', keep='last')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-20-60719893bb35> in <module>
    ----> 1 df2.drop_duplicates('Type', keep='last')
    

    NameError: name 'df2' is not defined


## 48. Check index duplicates


```python
df.index.duplicated()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-19-4e9f22415a99> in <module>
    ----> 1 df.index.duplicated()
    

    NameError: name 'df' is not defined


## 49. Grouping Data
## Aggregation


```python
df2.groupby(by=['Date','Type']).mean()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-18-bcd7e5a7e509> in <module>
    ----> 1 df2.groupby(by=['Date','Type']).mean()
    

    NameError: name 'df2' is not defined


df4.groupby(level=0).sum()


```python
df4.groupby(level=0).agg({'a':lambda x:sum(x)/len(x),
                          'b': np.sum})
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-17-e09b3786ae39> in <module>
    ----> 1 df4.groupby(level=0).agg({'a':lambda x:sum(x)/len(x),
          2                           'b': np.sum})
    

    NameError: name 'df4' is not defined


## 50. Transformation


```python
customSum = lambda x: (x+x%2)
df4.groupby(level=0).transform(customSum)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-16-f1cc800171a1> in <module>
          1 customSum = lambda x: (x+x%2)
    ----> 2 df4.groupby(level=0).transform(customSum)
    

    NameError: name 'df4' is not defined


## 51. Mising Data, Drop NaN values


```python
df.dropna()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-15-482195c7ff1f> in <module>
    ----> 1 df.dropna()
    

    NameError: name 'df' is not defined


## 52.Fill NaN values with a predetermined value



```python
df3.fillna(df3.mean())
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-14-b577ff9ca6f0> in <module>
    ----> 1 df3.fillna(df3.mean())
    

    NameError: name 'df3' is not defined


## 53. Replace values with others


```python
df2.replace("a","f")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-13-a85d6acf5060> in <module>
    ----> 1 df2.replace("a","f")
    

    NameError: name 'df2' is not defined


## 54. Combining Data


```python
data1 = pd.DataFrame({'X1': ['a','b','c'], 'X2': [11.432,1.303, 99.906]}); data1
data2 = pd.DataFrame({'X1': ['a','b','d'], 'X3': [20.78,"NaN", 20.784]}); data2
print(data1)
print(data2)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-12-ada1192a2413> in <module>
    ----> 1 data1 = pd.DataFrame({'X1': ['a','b','c'], 'X2': [11.432,1.303, 99.906]}); data1
          2 data2 = pd.DataFrame({'X1': ['a','b','d'], 'X3': [20.78,"NaN", 20.784]}); data2
          3 print(data1)
          4 print(data2)
    

    NameError: name 'pd' is not defined


## 55. Merge


```python
pd.merge(data1,
         data2,
        how='left',
        on='X1')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-11-03249bc990e7> in <module>
    ----> 1 pd.merge(data1,
          2          data2,
          3         how='left',
          4         on='X1')
    

    NameError: name 'pd' is not defined



```python
pd.merge(data1,
         data2,
        how='right',
        on='X1')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-10-585c9d32ff1b> in <module>
    ----> 1 pd.merge(data1,
          2          data2,
          3         how='right',
          4         on='X1')
    

    NameError: name 'pd' is not defined



```python
pd.merge(data1,
         data2,
        how='inner',
        on='X1')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-9-8592766e2445> in <module>
    ----> 1 pd.merge(data1,
          2          data2,
          3         how='inner',
          4         on='X1')
    

    NameError: name 'pd' is not defined



```python
pd.merge(data1,
         data2,
        how='outer',
        on='X1')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-8-30ae9bd55fae> in <module>
    ----> 1 pd.merge(data1,
          2          data2,
          3         how='outer',
          4         on='X1')
    

    NameError: name 'pd' is not defined


## 56 Join


```python
data1.join(data2, how='right')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-7-32df29ebadaa> in <module>
    ----> 1 data1.join(data2, how='right')
    

    NameError: name 'data1' is not defined


## 57. Concatenate
## Vertical


```python

s.append(s2)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-1b7756f7ae54> in <module>
    ----> 1 s.append(s2)
    

    NameError: name 's' is not defined


## 58.Horizontal/vertical



```python
pd.concat([s,s2],axis=1, keys=['One','Two'])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-c30eef370f1c> in <module>
    ----> 1 pd.concat([s,s2],axis=1, keys=['One','Two'])
    

    NameError: name 'pd' is not defined



```python
pd.concat([data1, data2], axis=1, join='inner')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-8d15ae9c456b> in <module>
    ----> 1 pd.concat([data1, data2], axis=1, join='inner')
    

    NameError: name 'pd' is not defined


## 59. Dates


```python
df2['Date']= pd.to_datetime(df2['Date'])
df2['Date']= pd.date_range('2000-1-1',
                            periods=6,
                            freq='M')
dates = [datetime(2012,5,1), datetime(2012,5,2)]
index = pd.DatetimeIndex(dates)
index = pd.date_range(datetime(2012,2,1), end, freq='BM')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-c0a5021f8444> in <module>
    ----> 1 df2['Date']= pd.to_datetime(df2['Date'])
          2 df2['Date']= pd.date_range('2000-1-1',
          3                             periods=6,
          4                             freq='M')
          5 dates = [datetime(2012,5,1), datetime(2012,5,2)]
    

    NameError: name 'pd' is not defined


## 60. Visualization


```python
import matplotlib.pyplot as plt
s.plot()
plt.show()
print(s)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-8063338a8065> in <module>
          1 import matplotlib.pyplot as plt
    ----> 2 s.plot()
          3 plt.show()
          4 print(s)
    

    NameError: name 's' is not defined



```python
df2.plot()
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-ba5ab7381d57> in <module>
    ----> 1 df2.plot()
          2 plt.show()
    

    NameError: name 'df2' is not defined

