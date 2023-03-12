import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

dataset_url_jan = 'divyansh22/flight-delay-prediction'
dataset_url_feb = 'divyansh22/february-flight-delay-prediction'

api= KaggleApi()
api.authenticate()

# January
api.dataset_download_file(dataset_url_jan, 'Jan_2019_ontime.csv')
api.dataset_download_file(dataset_url_jan, 'Jan_2020_ontime.csv')

with zipfile.ZipFile('Jan_2019_ontime.csv.zip', 'r') as zipref:
    zipref.extractall('./')
with zipfile.ZipFile('Jan_2020_ontime.csv.zip', 'r') as zipref:
    zipref.extractall('./')

