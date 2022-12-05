```python
import ipyleaflet
from ipyleaflet import Map
import pandas as pd
from matplotlib import pyplot as plt
import ipywidgets
from ipyleaflet import basemaps, Map
from ipyleaflet import Map, GeoJSON, Marker
import geocoder
from vega_datasets import data
import json
import requests
from bs4 import BeautifulSoup
from ipyleaflet import Map, Heatmap
from random import uniform
import ipyleaflet
import json
import pandas as pd
from branca.colormap import linear
import branca.colormap as cm
import numpy as np

```


```python
#Search String
pd.set_option('display.max_columns', None)
city = 'Jersey City'
state = 'NJ'
search_str = city + ', ' +state
print('Search string:', search_str)

```

    Search string: Jersey City, NJ



```python
#API for For Sale
url = "https://zillow-com1.p.rapidapi.com/propertyExtendedSearch"
querystring = {"location":search_str,
              "sort": "Square_Feet",
            "status":"recentlySold",
               "doz":"36m"}

headers = {
    'x-rapidapi-host': "zillow-com1.p.rapidapi.com",
    'x-rapidapi-key': "e92efa8d75mshe835187053569b3p1dde6cjsnbb3aa7f9a4b8"
    }
```


```python
response1 = requests.request("GET", url, headers=headers, params=querystring)
```


```python
#Transform to json 
response1_json = response1.json()
```


```python
#Normalizing Json in pandas 
homes = pd.json_normalize(data=response1_json['props'])
print('Num of rows:', len(homes))
print('Num of cols:', len(homes.columns))
```

    Num of rows: 40
    Num of cols: 23



```python
homes['pricepersquarefoot'] = homes.price/homes.livingArea
```


```python

```


```python
sold = homes
```


```python
response = requests.request("GET", url, headers=headers, params=querystring2)
```


```python
#Transform to json 
response_json = response.json()
```


```python
#Normalizing Json in pandas 
homes = pd.json_normalize(data=response1_json['props'])
print('Num of rows:', len(homes))
print('Num of cols:', len(homes.columns))
```

    Num of rows: 23
    Num of cols: 20



```python
homes.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23 entries, 0 to 22
    Data columns (total 20 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   dateSold                0 non-null      object 
     1   propertyType            23 non-null     object 
     2   lotAreaValue            23 non-null     float64
     3   address                 23 non-null     object 
     4   imgSrc                  23 non-null     object 
     5   price                   23 non-null     int64  
     6   bedrooms                14 non-null     float64
     7   longitude               18 non-null     float64
     8   latitude                18 non-null     float64
     9   listingStatus           23 non-null     object 
     10  zpid                    23 non-null     object 
     11  contingentListingType   0 non-null      object 
     12  daysOnZillow            23 non-null     int64  
     13  bathrooms               14 non-null     float64
     14  livingArea              14 non-null     float64
     15  country                 23 non-null     object 
     16  currency                23 non-null     object 
     17  lotAreaUnit             23 non-null     object 
     18  hasImage                23 non-null     bool   
     19  listingSubType.is_FSBA  23 non-null     bool   
    dtypes: bool(2), float64(6), int64(2), object(10)
    memory usage: 3.4+ KB



```python
homes.groupby(['propertyType'])['pricepersquarefoot'].describe().to_clipboard()
```


```python
#MultiSold = pd.json_normalize(data=response_json['props'])
```


```python
#Cleaning the data
Charleston_Multi = Multi.drop(['zpid', 'address', 'imgSrc', 'listingDateTime','listingStatus', 'country',
                               'currency', 'hasImage', 'listingSubType.is_FSBA'], axis = 1)

Charleston_Multisold = MultiSold.drop(['zpid','address', 'imgSrc', 'listingDateTime','listingStatus', 'country',
                               'currency', 'hasImage'], axis = 1)
#Dropping NAs
Charleston_Multi = Charleston_Multi.dropna()
Charleston_MultiSold = Charleston_Multisold.dropna()

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Input In [15], in <cell line: 2>()
          1 #Cleaning the data
    ----> 2 Charleston_Multi = Multi.drop(['zpid', 'address', 'imgSrc', 'listingDateTime','listingStatus', 'country',
          3                                'currency', 'hasImage', 'listingSubType.is_FSBA'], axis = 1)
          5 Charleston_Multisold = MultiSold.drop(['zpid','address', 'imgSrc', 'listingDateTime','listingStatus', 'country',
          6                                'currency', 'hasImage'], axis = 1)
          7 #Dropping NAs


    File ~/opt/anaconda3/lib/python3.9/site-packages/pandas/util/_decorators.py:311, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        305 if len(args) > num_allow_args:
        306     warnings.warn(
        307         msg.format(arguments=arguments),
        308         FutureWarning,
        309         stacklevel=stacklevel,
        310     )
    --> 311 return func(*args, **kwargs)


    File ~/opt/anaconda3/lib/python3.9/site-packages/pandas/core/frame.py:4954, in DataFrame.drop(self, labels, axis, index, columns, level, inplace, errors)
       4806 @deprecate_nonkeyword_arguments(version=None, allowed_args=["self", "labels"])
       4807 def drop(
       4808     self,
       (...)
       4815     errors: str = "raise",
       4816 ):
       4817     """
       4818     Drop specified labels from rows or columns.
       4819 
       (...)
       4952             weight  1.0     0.8
       4953     """
    -> 4954     return super().drop(
       4955         labels=labels,
       4956         axis=axis,
       4957         index=index,
       4958         columns=columns,
       4959         level=level,
       4960         inplace=inplace,
       4961         errors=errors,
       4962     )


    File ~/opt/anaconda3/lib/python3.9/site-packages/pandas/core/generic.py:4267, in NDFrame.drop(self, labels, axis, index, columns, level, inplace, errors)
       4265 for axis, labels in axes.items():
       4266     if labels is not None:
    -> 4267         obj = obj._drop_axis(labels, axis, level=level, errors=errors)
       4269 if inplace:
       4270     self._update_inplace(obj)


    File ~/opt/anaconda3/lib/python3.9/site-packages/pandas/core/generic.py:4311, in NDFrame._drop_axis(self, labels, axis, level, errors, consolidate, only_slice)
       4309         new_axis = axis.drop(labels, level=level, errors=errors)
       4310     else:
    -> 4311         new_axis = axis.drop(labels, errors=errors)
       4312     indexer = axis.get_indexer(new_axis)
       4314 # Case for non-unique axis
       4315 else:


    File ~/opt/anaconda3/lib/python3.9/site-packages/pandas/core/indexes/base.py:6644, in Index.drop(self, labels, errors)
       6642 if mask.any():
       6643     if errors != "ignore":
    -> 6644         raise KeyError(f"{list(labels[mask])} not found in axis")
       6645     indexer = indexer[~mask]
       6646 return self.delete(indexer)


    KeyError: "['listingDateTime'] not found in axis"



```python
#Dropping Acres
#Charleston_Multi.drop(index = Charleston_Multi[Charleston_Multi["lotAreaUnit"] == 'acres'].index)
Charleston_MultiSold.drop(index = Charleston_Multisold[Charleston_Multisold["lotAreaUnit"] == 'acres'].index)
```


```python
#Summary Statistics 
#Charleston_Multi.describe()
#Charleston_Multi.sum()

#Charleston_MultiSold.describe()
#Charleston_MultiSold.sum()
```


```python
# create map
# basic_map = ipyleaflet.Map(zoom=1)
basic_map = ipyleaflet.Map(zoom=1)
```


```python
radio_button = ipywidgets.RadioButtons(options=['Positron', 'DarkMatter', 'WorldStreetMap', 'DeLorme', 
                                                'WorldTopoMap', 'WorldImagery', 'NatGeoWorldMap', 'HikeBike', 
                                                'HyddaFull', 'Night', 'ModisTerra', 'Mapnik', 'HOT', 'OpenTopoMap', 
                                                'Toner', 'Watercolor'],
                                       value='Positron', 
                                       description='map types:')

```


```python
def toggle_maps(map):
    if map == 'Positron': m = Map(zoom=2, basemap=basemaps.CartoDB.Positron)
    if map == 'DarkMatter': m = Map(zoom=1, basemap=basemaps.CartoDB.DarkMatter)
    if map == 'WorldStreetMap': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.Esri.WorldStreetMap)
    if map == 'DeLorme': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.Esri.DeLorme)
    if map == 'WorldTopoMap': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.Esri.WorldTopoMap)
    if map == 'WorldImagery': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.Esri.WorldImagery)
    if map == 'NatGeoWorldMap': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.Esri.NatGeoWorldMap)
    if map == 'HikeBike': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.HikeBike.HikeBike)
    if map == 'HyddaFull': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.Hydda.Full)
    if map == 'Night': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.NASAGIBS.ViirsEarthAtNight2012)
    if map == 'ModisTerra': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.NASAGIBS.ModisTerraTrueColorCR)
    if map == 'Mapnik': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.OpenStreetMap.Mapnik)
    if map == 'HOT': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.OpenStreetMap.HOT)
    if map == 'OpenTopoMap': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.OpenTopoMap)
    if map == 'Toner': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.Stamen.Toner)
    if map == 'Watercolor': m = Map(center=(32.7765, -79.9311), zoom=11, basemap=basemaps.Stamen.Watercolor)
    #display(m)
        
#ipywidgets.interact(toggle_maps, map=radio_button)    
```


```python
location = geocoder.osm('960 Morrison Dr, Charleston, SC 29403')

```


```python
# latitude and longitude of location
latlng = [location.lat, location.lng]
```


```python
# create map
Lee_and_Associates = Map(center=latlng)
```


```python
# marker
marker = Marker(location=latlng, title='960 Morrison Dr #400, Charleston, SC 29403')
Lee_and_Associates.add_layer(marker)
```


```python
#Marking Sold
Salesmap = Map(center=(32.7765, -79.9311), zoom=1)

# plot airport locations
for (index,row) in Charleston_MultiSold.iterrows():
    marker = Marker(location=[row.loc['latitude'],row.loc['longitude']],
                    title = "For Sale")
    Salesmap.add_layer(marker)
    
    marker = Marker(location=latlng, title='960 Morrison Dr #400, Charleston, SC 29403')
Lee_and_Associates.add_layer(marker)

```


```python
# display map    
Salesmap

```


```python
with open ('/Users/Tyler/Desktop/Zillow/map.geojson') as f:
    geo_json_charleston = json.load(f)
#Create map    
Salesmap1 = ipyleaflet.Map(center=(32.7765, -79.9311), zoom=8)
geo_layer_charleston = GeoJSON(data = geo_json_charleston,
                              style = {'color' : 'blue',
                                       'opacity': 1.0,
                                      'weight': 1.9,
                                      'fill': 'green',
                                      'fillOpacity': 0.5})
Salesmap1.add_layer(geo_layer_charleston)

display(Salesmap1)


# In[35]:


#Choropleth Map
# shapefiles can be converted to geojson with QGIS
with open('/Users/Tyler/Desktop/Zillow/map.geojson') as f:
    geo_json_data = json.load(f)


# In[ ]:





# In[89]:


#Scatter Plot Model

Price_Analysis = Charleston_Multi.sort_values('price', ascending = False)
#plt.scatter(Price_Analysis.index.values, Price_Analysis['price'])

Price_AnalysisSold = Charleston_MultiSold.sort_values('price', ascending = False)
#plt.scatter(Price_AnalysisSold.index.values, Price_AnalysisSold['price'])


# In[81]:


price = forsale['price']
sqft = forsale['livingArea']
myMax = max(max(forsale['price']),max(forsale['livingArea']))


# In[83]:


plt.scatter(forsale['price'],forsale['livingArea'], color = 'k')
plt.xlabel("Price")
plt.ylabel("SQFT")
plt.plot([0,myMax], [0,myMax])
plt.show()


# In[84]:


plt.hist(forsale.price)
plt.show()
```
