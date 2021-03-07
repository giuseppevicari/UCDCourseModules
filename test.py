import pandas as pd
canadian_youtube = pd.read_csv("CAvideos.csv")
british_youtube = pd.read_csv("GBvideos.csv")
#print(british_youtube.head())
#print(british_youtube.info())
print(canadian_youtube.shape)
print(british_youtube.shape)
concat_data = pd.concat([canadian_youtube, british_youtube])
print(concat_data.shape)

left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])
join_data = left.join(right, lsuffix='_CAN', rsuffix='_UK')
merged_data = pd.merge(left, right, on = 'video_id')