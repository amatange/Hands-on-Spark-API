# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()

# Load datasets
logs_df = spark.read.csv('listening_logs.csv', header=True, inferSchema=True)
meta_df = spark.read.csv('songs_metadata.csv', header=True, inferSchema=True)
joined_df = logs_df.join(meta_df, on="song_id", how="inner")

# Task 1: User Favorite Genres
user_genre_counts = joined_df.groupby("user_id", "genre").agg(count("song_id").alias("genre_listen_count"))
user_window = Window.partitionBy("user_id").orderBy(desc("genre_listen_count"))
favorite_genres_df = user_genre_counts.withColumn("rank", row_number().over(user_window)).filter(col("rank") == 1).drop("rank")
favorite_genres_df.write.csv('outputs/task1', header=True, mode='overwrite')

# Task 2: Average Listen Time
avg_listen_time_df = logs_df.groupby("user_id").agg(round(avg("duration_sec"), 2).alias("avg_duration_sec"))
avg_listen_time_df.write.csv('outputs/task2', header=True, mode='overwrite')

# Task 3: Create your own Genre Loyalty Scores and rank them and list out top 10
total_user_listens = logs_df.groupby("user_id").agg(count("song_id").alias("total_listens"))
loyalty_df = favorite_genres_df.join(total_user_listens, on="user_id", how="inner")
loyalty_df = loyalty_df.withColumn("genre_loyalty_score", round((col("genre_listen_count") / col("total_listens")) * 100, 2))
top_10_loyal_users = loyalty_df.select("user_id", "genre", "genre_loyalty_score").orderBy(desc("genre_loyalty_score")).limit(10)
top_10_loyal_users.write.csv('outputs/task3', header=True, mode='overwrite')

# Task 4: Identify users who listen between 12 AM and 5 AM
night_owls_df = logs_df.withColumn("listen_hour", hour(col("timestamp"))).filter((col("listen_hour") >= 0) & (col("listen_hour") < 5)).select("user_id").distinct()
night_owls_df.write.csv('outputs/task4', header=True, mode='overwrite')

spark.stop()