
# coding: utf-8

# # Great Value Identifier

# In[1]:

#importing the required modules

import ibmos2spark
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, weekofyear, datediff, month
from pyspark.sql.types import DateType, IntegerType
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import RandomForest, RandomForestModel


# In[2]:

# importing mock search-results test data from object storage

credentials = {
    'auth_url': 'https://identity.open.softlayer.com',
    'project_id': 'a52717a052f343a1b9020e4160a61913',
    'region': 'dallas',
    'user_id': 'c8d82fa9a27147ba8a9c1d984f5185d3',
    'username': 'member_157b57e8fa7b7b37664471ffe4919c0ba44b9c4f',
    'password': 'L7{xz9VVCq^6k80s'
}

configuration_name = 'os_4f7c12d0b952450db479df6fdbb30d48_configs'
bmos = ibmos2spark.bluemix(sc, credentials, configuration_name)

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
search_import = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load(bmos.url('FindHotelProjecttemp', 'search_test.csv'))

search_import.show()

# usersearch_id = search by 1 given user
# search_id = the search results
# In[3]:

#importing support data

support_data_raw = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load(bmos.url('FindHotelProjecttemp', 'support_data.csv'))
    
support_data = support_data_raw.select(['hotel_id', 'effrat', 'rating', 'popularity', 'property_type', 'place_type_id', 'rank', 'tot_facilities', 'tot_themes'])
support_data.show(5)


# In[4]:

#processing search results data wrt dates and timelines

search1 = search_import.select('search_id', search_import.date.cast(DateType()).alias('search_date'))
search2 = search_import.select('search_id', search_import.check_in.cast(DateType()).alias('checkin_date')).join(search1, on = 'search_id', how = 'inner')
search3 = search_import.select('search_id', search_import.check_out.cast(DateType()).alias('checkout_date')).join(search2, on = 'search_id', how = 'inner')
search4 = search3.select('search_id', month(search3.search_date).alias('search_month')).join(search3, on = 'search_id', how ='inner')
search5 = search4.select('search_id', month(search4.checkin_date).alias('checkin_month')).join(search4, on = 'search_id', how ='inner')
search6 = search5.withColumn('bookdays', datediff(col('checkout_date'), col('checkin_date')))
search_dates = search6.withColumn('searchdays', datediff(col('checkin_date'), col('search_date'))).join(search_import, on ='search_id', how = 'inner')


# In[5]:

#processing room description to arrive at total guets and effective USD

search_peps = search_dates.withColumn("record", F.explode(F.split("room_description", '\|')))
search_peps1 = search_peps.withColumn("record", F.split("record", ':')).withColumn("adults", F.col("record")[0]).withColumn("kids", F.split(F.col("record")[1], ','))
search_peps2 = search_peps1.withColumn("kids_num", F.when(F.size("kids") > 0, F.size("kids")).otherwise(0)).groupby("search_id").agg(F.sum("adults").alias("tot_adults"), F.sum("kids_num").alias("tot_kids"))
search_peps3 = search_peps2.withColumn("kids_eq", (col('tot_kids')/2)).drop('tot_kids')
search_peps4 = search_peps3.groupBy('search_id').agg((F.sum(search_peps3.kids_eq + search_peps3.tot_adults)).alias('tot_guests'))
search_join = search_peps4.join(search_dates, on='search_id', how='inner')
search_eff = search_join.withColumn('effUSD', (((col('total_usd')/col('bookdays'))/col('nb_rooms'))/col('tot_guests'))).na.fill(0)
search = search_eff.select(['search_id','hotel_id','checkin_month', 'search_month', 'searchdays', 'effUSD']).dropDuplicates()


# In[6]:

#preparing mock search data with support data for prediction

prepared_data =search.join(support_data, on='hotel_id', how='inner').select(['checkin_month', 'search_month', 'searchdays', 'effUSD', 'effrat', 'rating', 'popularity', 'property_type', 'place_type_id', 'rank', 'tot_facilities', 'tot_themes'])
prepared_data_rdd = prepared_data.rdd.map(lambda x: array(x, dtype=float))


# In[7]:

#importing saved Random forest model and predicting

gvim = RandomForestModel.load(sc, "gvimodel")
prediction = gvim.predict(prepared_data_rdd.map(lambda x: x)).collect()


# In[8]:

#the final predictions, 0 = Not great value, 1 = Great Value

prediction[0:]


# In[9]:

print(gvim.toDebugString())


#     

# # Ground Truth Generator

# 

# In[ ]:

#All the code below is a one time run in order to: 1)Prepare support data, 2)Train algorithm


# In[ ]:

#importing places data from object storage
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, weekofyear, datediff, month
from pyspark.sql.types import DateType
import ibmos2spark

place_raw = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load(bmos.url('FindHotelProjecttemp', 'place.csv'))
    
place = place_raw.select(['place_id', 'place_type_id', 'latitude', 'longitude', 'rank'])


# In[ ]:

#importing hotel data from object storage
#processing the hotel data

hotel_raw = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load(bmos.url('FindHotelProjecttemp', 'hotel.csv'))

hotel_theme = hotel_raw.select(['themes', 'hotel_id']).withColumn("theme", F.explode(F.split("themes", '\|')))
hotel_theme1 = hotel_theme.withColumn("tot_theme", F.when((hotel_theme["theme"] > 0), 1).otherwise(0)).groupby("hotel_id").agg(F.sum("tot_theme").alias("tot_themes")).drop('themes')
hotel_facs = hotel_raw.select(['facilities', 'hotel_id']).withColumn("facility", F.explode(F.split("facilities", '\|')))
hotel_facs1 = hotel_facs.withColumn("tot_facility", F.when((hotel_facs["facility"] > 0), 1).otherwise(0)).groupby("hotel_id").agg(F.sum("tot_facility").alias("tot_facilities")).drop('facilities')
hotel_fcth_join = hotel_facs1.join(hotel_theme1, on='hotel_id', how ='outer').dropDuplicates()
hotel_fcth = hotel_fcth_join.join(hotel_raw, on='hotel_id', how='right_outer').dropDuplicates()
hotel_filt_double = hotel_fcth.select(['hotel_id','rating','number_of_reviews', 'cleanliness_rating', 'service_rating', 'facilities_rating', 'location_rating', 'pricing_rating', 'rooms_rating', 'dining_rating', 'overall_rating', 'popularity', 'property_type', 'place_id', 'min_rate_usd', 'tot_themes','tot_facilities'])
hotel_filt_date = hotel_raw.select(['hotel_id', 'cheapest_check_in'])


# In[ ]:

#more processing 

hotel_double = hotel_filt_double.select(*(col(c).cast('float').alias(c) for c in hotel_filt_double.columns)).na.fill(0)
hotel_double_remain = hotel_filt_double.select(['hotel_id','rating', 'number_of_reviews','popularity', 'property_type', 'place_id', 'min_rate_usd', 'tot_themes','tot_facilities'])
hotel_sum = hotel_double.groupBy('hotel_id').agg((F.sum(hotel_double.cleanliness_rating + hotel_double.service_rating + hotel_double.facilities_rating + hotel_double.location_rating + hotel_double.rooms_rating + hotel_double.dining_rating + hotel_double.overall_rating)/7).alias('mean_rating'))
hotel_date = hotel_filt_date.select('hotel_id', hotel_filt_date.cheapest_check_in.cast(DateType()).alias('cheapest_date'))
#hotel_dd = hotel_date.select('hotel_id', weekofyear(hotel_date.cheapest_date).alias('cheapest_week')).join(hotel_sum, on = 'hotel_id', how = 'inner').drop('cheapest_date')
hotel_dd = hotel_date.select('hotel_id', month(hotel_date.cheapest_date).alias('cheapest_month')).join(hotel_sum, on = 'hotel_id', how = 'inner').drop('cheapest_date')
hotel_joi = hotel_dd.join(hotel_double_remain, on = 'hotel_id', how = 'inner')
hotel = hotel_joi.withColumn('effrat', (col('mean_rating')*col('number_of_reviews')))


# In[ ]:

#joining prepared place and hotel data

place_hotel = place.join(hotel, on = "place_id", how ='inner') #this dataset has been exported at support_data


# In[ ]:

#importing lead data from object storage

lead_raw = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load(bmos.url('FindHotelProjecttemp', 'lead.csv'))
    
lead_import = lead_raw.sample(True, 0.01, seed=0) #limiting to 350k due to infrastructure limitations


# In[ ]:

#processing historic lead data wrt timelines

#func =  udf(lambda x: datetime.strptime(x, '%Y-%m-%d'), DateType()) 
#lead1 = lead.withColumn('lead_date', func(col('test')))
lead1 = lead_import.select('lead_id', lead_import.date.cast(DateType()).alias('lead_date'))
lead2 = lead_import.select('lead_id', lead_import.check_in.cast(DateType()).alias('checkin_date')).join(lead1, on = 'lead_id', how = 'inner')
lead3 = lead_import.select('lead_id', lead_import.check_out.cast(DateType()).alias('checkout_date')).join(lead2, on = 'lead_id', how = 'inner')
lead4 = lead3.select('lead_id', month(lead3.lead_date).alias('lead_month')).join(lead3, on = 'lead_id', how ='inner')
lead5 = lead4.select('lead_id', month(lead4.checkin_date).alias('checkin_month')).join(lead4, on = 'lead_id', how ='inner')
lead6 = lead5.withColumn('bookdays', datediff(col('checkout_date'), col('checkin_date')))
lead_dates = lead6.withColumn('leaddays', datediff(col('checkin_date'), col('lead_date'))).join(lead_import, on ='lead_id', how = 'inner')


# In[ ]:

#processing historic room description, arriving at total guets and effective USD

lead_peps = lead_dates.withColumn("record", F.explode(F.split("room_description", '\|')))
lead_peps1 = lead_peps.withColumn("record", F.split("record", ':')).withColumn("adults", F.col("record")[0]).withColumn("kids", F.split(F.col("record")[1], ','))
lead_peps2 = lead_peps1.withColumn("kids_num", F.when(F.size("kids") > 0, F.size("kids")).otherwise(0)).groupby("lead_id").agg(F.sum("adults").alias("tot_adults"), F.sum("kids_num").alias("tot_kids"))
lead_peps3 = lead_peps2.withColumn("kids_eq", (col('tot_kids')/2)).drop('tot_kids')
lead_peps4 = lead_peps3.groupBy('lead_id').agg((F.sum(lead_peps3.kids_eq + lead_peps3.tot_adults)).alias('tot_guests'))
lead_join = lead_peps4.join(lead_dates, on='lead_id', how='inner')
lead_eff = lead_join.withColumn('effUSD', (((col('total_usd')/col('bookdays'))/col('nb_rooms'))/col('tot_guests'))).na.fill(0)
lead = lead_eff.select(['lead_id','hotel_id','checkin_month', 'lead_month', 'leaddays', 'effUSD']).dropDuplicates()


# In[ ]:

#preparing data for K-means clustering

data_float = place_hotel.select(['hotel_id', 'effrat', 'rating', 'popularity','property_type', 'place_type_id', 'rank', 'tot_facilities', 'tot_themes', 'latitude', 'longitude'])
data = data_float.select(*(col(c).cast("float").alias(c) for c in data_float.columns)).na.fill(0).dropDuplicates().sort('hotel_id').drop('hotel_id')


# In[ ]:

#K-means model training

import numpy as np
from math import sqrt
from pyspark.mllib.clustering import KMeans, KMeansModel
from sklearn.preprocessing import scale

data_array = data.rdd.map(lambda x: scale(np.array(x, dtype=float)))
#clusters = KMeans.train(data_array, 35000, maxIterations=15, initializationMode="random") #takes over 200 minutes


# In[ ]:

#model evaluation

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = (data_array.map(lambda point: error(point)).reduce(lambda x, y: x + y)) #takes 200+ mins for all hotel ids
print("Within Set Sum of Squared Error = " + str(WSSSE))


# In[ ]:

#saving model
#clusters.save(sc, "KMeansModelFH") #commenting to avoid overwrite


# In[ ]:

#loading saved model

from pyspark.mllib.clustering import KMeans, KMeansModel
clusters = KMeansModel.load(sc, "KMeansModelFH")


# In[ ]:

#extracting clusterid data 

results_rdd = data_array.map(lambda point: clusters.predict(point)).cache() # takes 200+mins for all hotels


# In[ ]:

#combining cluster points with respective hotel_place record

data_hotelid_rdd = data_float.select('hotel_id').rdd.flatMap(lambda x: x)
results_prdd_id = results_rdd.zipWithIndex()
hotelid_prdd_id = data_hotelid_rdd.zipWithIndex()
results_id_df = results_prdd_id.toDF(['cluster_id', 'id'])
hotelid_id_df = hotelid_prdd_id.toDF(['hotel_id', 'id'])
cluster_hotelid = hotelid_id_df.join(results_id_df, on='id', how='inner')
clustered_rawdata = cluster_hotelid.join(data_float, on = 'hotel_id', how = 'inner')


# In[ ]:

# identify if it was great value or not

clustered_data =clustered_rawdata.join(lead, on='hotel_id', how='inner').select(['lead_id','hotel_id','cluster_id', 'checkin_month', 'lead_month', 'leaddays', 'effUSD', 'effrat', 'rating', 'popularity', 'property_type', 'place_type_id', 'rank', 'tot_facilities', 'tot_themes'])
clustered_data_group = clustered_data.groupby('cluster_id').agg({'effUSD' : 'mean'}).withColumnRenamed('avg(effUSD)', 'meanUSD')
data_mean_usd = clustered_data_group.join(clustered_data, on = 'cluster_id', how = 'inner').dropDuplicates()
prepared_data = data_mean_usd.withColumn('gooddeal', F.when((data_mean_usd["meanUSD"]>=data_mean_usd["effUSD"]), 1).otherwise(0))
train_data = prepared_data.select(['checkin_month', 'lead_month', 'leaddays', 'effUSD', 'effrat', 'rating', 'popularity', 'property_type', 'place_type_id', 'rank', 'tot_facilities', 'tot_themes', 'gooddeal']).na.fill(0)


#    

# # Great Value Identifier Random Forest Model Train & Test

#   

# In[ ]:

#Preparing the training data 
from pyspark.mllib.regression import LabeledPoint
from numpy import array

data_raw = train_data.rdd.map(lambda x: LabeledPoint(x[12], array((x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8] ,x[9], x[10], x[11]), dtype=float)))
(trainingData, testData) = data_raw.randomSplit([0.7, 0.3])


# In[ ]:

#Training the random forest model
from pyspark.mllib.tree import RandomForest, RandomForestModel

model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={0:13, 1:13, 5:13, 11:14},
                                     numTrees=4, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=7, maxBins=15)


# In[ ]:


#Testing the trained model on the test data and evaluating error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())


# In[ ]:


#Saving the trained and tested model
model.save(sc, "gvimodel")

