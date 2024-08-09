import requests
from django.core.management.base import BaseCommand

from draws.models import Drawing
from draws.models import WeatherDrawing

from sodapy import Socrata
import requests
import json

import datetime
from datetime import timedelta
from datetime import datetime
import pytz

import numpy as np
import pandas as pd

import sklearn
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


# Score the results on triangular numbers.
def predictionScore(prediction, sample):
    sample = list(sample)
    c = 0 
    for p in prediction:
        if p in sample:
            c += 1
            sample.remove(p)
    if c > 0:
        #Triangular number
        return ((c**2)+c)/(2)
    else:
        return c


class Command(BaseCommand):
    help = 'Fetch data from the external API and populate the database'

    def handle(self, *args, **kwargs):
        client = Socrata("data.ny.gov", None)
        results = client.get("8vkr-v8vh", limit=1, order="draw_date DESC", offset=0)
        draws_data = results
        #print(results)

        for draw in draws_data:
            date_obj = datetime.strptime(draw['draw_date'], '%Y-%m-%dT%H:%M:%S.%f')
            number_POS = draw['winning_numbers'].split()

            #get weather data
            response = requests.request("GET", "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/250%20Marriott%20Dr%2C%20Ste%20250%2C%20Tallahassee%2C%20FL/" + str(datetime.date(date_obj)) + "/" + str(datetime.date(date_obj)) + "?unitGroup=us&maxDistance=2000&elements=datetime%2Ctemp%2Cfeelslike%2Cdew%2Chumidity%2Cprecip%2Cprecipcover%2Cwindspeedmean%2Cwinddir%2Cpressure%2Ccloudcover%2Cmoonphase%2Csource&include=days%2Cobs%2Cremote&key=PBNL7ZQAHHHJNZ3KXGKZ56G4J&options=stnslevel1&contentType=json")
            weather_data = response.json()

            try:
                Drawing.objects.update_or_create(
                    draw_date = str(datetime.date(date_obj)),
                    draw_night = str(datetime.date(date_obj).weekday()),
                    pos1 = number_POS[0],
                    pos2 = number_POS[1],
                    pos3 = number_POS[2],
                    pos4 = number_POS[3],
                    pos5 = number_POS[4],
                    pwrb = number_POS[5],
                    Mx = draw['multiplier'],
                    mega = 0,
                    Barometer = weather_data['days'][0]['pressure'],
                    Dewpoint = weather_data['days'][0]['dew'],
                    Heat_Index = weather_data['days'][0]['feelslike'],
                    Hygrometer = weather_data['days'][0]['humidity'],
                    Thermometer = weather_data['days'][0]['temp'],
                    Percipitation = weather_data['days'][0]['precip'],
                    WindDirection = weather_data['days'][0]['winddir'],
                    CloudCover = weather_data['days'][0]['cloudcover'],
                    WindSpeed = weather_data['days'][0]['windspeedmean'],
                    MoonPhase = weather_data['days'][0]['moonphase'],
                    Station = weather_data['days'][0]['source'],
                )
            except:
                print("Exists")
            
            print(str(datetime.date(date_obj)), draw['winning_numbers'])

        # PREDICTION #
        utc=pytz.utc
        nextdraw_date = datetime.today()

        while nextdraw_date.weekday() != 0:
            if nextdraw_date.weekday() == 2:
                break
            if nextdraw_date.weekday() == 5:
                break
            else:
                nextdraw_date += timedelta(1)

        # or manually set date
        #nextdraw_date = nextdraw_date.replace(year=2024, month=6, day=24)

        nextdraw_date = nextdraw_date.replace(hour=22, minute=0)
        nextdraw_enddate = nextdraw_date.replace(hour=23, minute=0)
        nextdraw_date_utcepoch = int(nextdraw_date.astimezone(utc).timestamp())

        print(nextdraw_date.strftime("%Y-%m-%dT%H:%M:%SZ"), nextdraw_date.weekday())
        
        ## Get weather forecast for next draw date
        # 250 Marriott Dr | Tallahassee, FL 32301
        # lat: 30.439489
        # lon: -84.264712
        response = requests.request("GET", "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/250%20Marriott%20Dr%2C%20Ste%20250%2C%20Tallahassee%2C%20FL/" + nextdraw_date.strftime("%Y-%m-%d") + "/" + nextdraw_date.strftime("%Y-%m-%d") + "?unitGroup=us&key=PBNL7ZQAHHHJNZ3KXGKZ56G4J&contentType=json")
        forecast_data = response.json()
        #print(forecast_data) 
        
        try: 
            WeatherDrawing.objects.update_or_create(
                draw_date = str(nextdraw_date.strftime("%Y-%m-%d")),
                draw_night = str(nextdraw_date.weekday()),
                Barometer = forecast_data['days'][0]['pressure'],
                Dewpoint = forecast_data['days'][0]['dew'],
                Heat_Index = forecast_data['days'][0]['feelslike'],
                Hygrometer = forecast_data['days'][0]['humidity'],
                Thermometer = forecast_data['days'][0]['temp'],
                Percipitation = forecast_data['days'][0]['precip'],
                WindDirection = forecast_data['days'][0]['winddir'],
                CloudCover = forecast_data['days'][0]['cloudcover'],
                WindSpeed = forecast_data['days'][0]['windspeed'],
                MoonPhase = forecast_data['days'][0]['moonphase'],
                Station = forecast_data['days'][0]['source'],
            )
        except:
            print("duplicate")

        nextdraw_set = [forecast_data['days'][0]['pressure'],forecast_data['days'][0]['dew'],forecast_data['days'][0]['feelslike'],forecast_data['days'][0]['humidity'],forecast_data['days'][0]['temp'],forecast_data['days'][0]['precip'],forecast_data['days'][0]['winddir'],forecast_data['days'][0]['cloudcover'],forecast_data['days'][0]['windspeed'],forecast_data['days'][0]['moonphase'] ]
        
        # grab all data for training...
        training_data_raw = Drawing.objects.values_list('pos1','pos2','pos3','pos4','pos5','pwrb','Barometer','Dewpoint','Heat_Index','Hygrometer','Thermometer','Percipitation','WindDirection','CloudCover','WindSpeed','MoonPhase')

        # put rows in a dataframe
        df_w = pd.DataFrame(training_data_raw)

        # make everything numeric
        for column in df_w:
            # set all to numeric
            df_w[column] = pd.to_numeric(df_w[column], errors='coerce')
        print(df_w)

        df_w.head()

        # Split the data into independent and dependent variables
        X = df_w.iloc[:,6:16].values
        y = df_w.iloc[:,0:6].values
        print('The independent features set: ')
        print(X[:5,:])
        print('The dependent variable: ')
        print(y[:5])

        # Creating the Training and Test set from data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)#, random_state = 69)
        print(X_train[:3,:])
        #y_train = y_train.reshape(-1,1)
        print(y_train[:3])

        # Feature Scaling
        scaler = StandardScaler()
        X_fit_train = scaler.fit_transform(X_train)
        X_fit_test = scaler.transform(X_test)
        print('Scaled:')
        #X_fit_train = X_fit_train.reshape(-1,1)
        print(X_fit_train[:3,:])

        X_fit = scaler.fit_transform(X)
        print('X Scaled:')
        #X_fit = X_fit.reshape(-1,1)
        print(X_fit[:3,:])

        nextDrawResults = []

        ########################
        ### MAKE PREDICTIONS ###
        ########################
        iteration_throttle = 7

        ## Random Forest - entropy ##
        winningScore = 0
        winningState = 0
        RFcriterion = "entropy"

        irandom_state = 1

        while irandom_state <= 105:
            # Fitting Random Forest Classification to the Training set
            RFclassifier = RandomForestClassifier(n_estimators=irandom_state, criterion=RFcriterion, random_state=irandom_state)
            RFclassifier.fit(X_fit_train, y_train)
            #print (RFclassifier)
            
            # Compare the Test set results for Random Forest Classification
            y_pred = RFclassifier.predict(X_fit_test)

            print('----- ' + str(irandom_state) + ' ------')
            accuracyScore = 0
            totalMatchScore = 0
            for index, pred in enumerate(y_pred):
                accuracy = accuracy_score(y_test[index], pred)
                accuracyScore += accuracy

                comparePOS = list(set(pred[:5]) & set(y_test[index][:5]))
                comparePWB = list(set(pred[5:]) & set(y_test[index][5:]))

                matchScore = predictionScore(pred[:5], y_test[index][:5]) + (predictionScore(pred[5:], y_test[index][5:])*6)

                if matchScore >= 6:
                    totalMatchScore += matchScore
                    #print(pred, y_test[index], comparePOS, comparePWB, str(int(matchScore)))

                # Find the winner!
                if totalMatchScore >= winningScore:
                    winningScore = totalMatchScore
                    winningState = irandom_state

            print('-----> Accuracy: ' + str(round(accuracyScore,4)) + ' | Match Score: ' + str(int(totalMatchScore)) + ' <------')
            irandom_state += iteration_throttle

        print('-----> Winner is ' + str(int(winningState)) + ' with ' + str(int(winningScore)) + ' <------')
        # Fitting Random Forest Classification to the Training set
        RFclassifier = RandomForestClassifier(n_estimators=int(winningState), criterion=RFcriterion, random_state=int(winningState))
        RFclassifier.fit(X_fit, y)

        # Predicting the next draw set results
        nextdraw_pred = RFclassifier.predict([nextdraw_set])
        print(nextdraw_pred[0], nextdraw_set)

        nextDrawResults.append([{"numbers":list(map(int, nextdraw_pred[0]))},{"score":int(winningScore)},{"class":"weather - RandomForestClassifier - E"}])

        ## Random Forest - gini ##
        winningScore = 0
        winningState = 0
        RFcriterion = "gini"
        
        irandom_state = 1
        while irandom_state <= 105:
            # Fitting Random Forest Classification to the Training set
            RFclassifier = RandomForestClassifier(n_estimators=irandom_state, criterion=RFcriterion, random_state=irandom_state)
            RFclassifier.fit(X_fit_train, y_train)
            #print (RFclassifier)

            # Compare the Test set results for Random Forest Classification
            y_pred = RFclassifier.predict(X_fit_test)
            
            print('----- ' + str(irandom_state) + ' ------')
            accuracyScore = 0
            totalMatchScore = 0

            for index, pred in enumerate(y_pred):
                accuracy = accuracy_score(y_test[index], pred)
                accuracyScore += accuracy
                comparePOS = list(set(pred[:5]) & set(y_test[index][:5]))
                comparePWB = list(set(pred[5:]) & set(y_test[index][5:]))
                matchScore = predictionScore(pred[:5], y_test[index][:5]) + (predictionScore(pred[5:], y_test[index][5:])*6)

                if matchScore >= 6:
                    totalMatchScore += matchScore
                    #print(pred, y_test[index], comparePOS, comparePWB, str(int(matchScore)))

                # Find the winner!
                if totalMatchScore >= winningScore:
                    winningScore = totalMatchScore
                    winningState = irandom_state

            print('-----> Accuracy: ' + str(round(accuracyScore,4)) + ' | Match Score: ' + str(int(totalMatchScore)) + ' <------')
            irandom_state += iteration_throttle

        print('-----> Winner is ' + str(int(winningState)) + ' with ' + str(int(winningScore)) + ' <------')
        # Fitting Random Forest Classification to the Training set
        RFclassifier = RandomForestClassifier(n_estimators=int(winningState), criterion=RFcriterion, random_state=int(winningState))
        RFclassifier.fit(X_fit, y)

        # Predicting the next draw set results
        nextdraw_pred = RFclassifier.predict([nextdraw_set])
        print(nextdraw_pred[0], nextdraw_set)

        nextDrawResults.append([{"numbers":list(map(int, nextdraw_pred[0]))},{"score":int(winningScore)},{"class":"weather - RandomForestClassifier - G"}])

        ## KNN Classifier - uniform ##
        winningScore = 0
        winningState = 0

        in_neighbors = 1
        while in_neighbors <= 154:
            # Fitting KNN
            KNNWeights = "uniform" # weights{"uniform", "distance"}
            KNNAlgorithm = "auto" # algorithm{"auto", "ball_tree", "kd_tree", "brute"}
            neigh = KNeighborsClassifier(weights=KNNWeights, algorithm=KNNAlgorithm, n_neighbors=in_neighbors)
            neigh.fit(X_fit_train, y_train)
            
            # Predicting the Test set results
            y_n_pred = neigh.predict(X_fit_test)
            
            print('----- ' + str(in_neighbors) + ' ------')
            accuracyScore = 0
            totalMatchScore = 0
            for index, pred in enumerate(y_n_pred):
                accuracy = accuracy_score(y_test[index], pred)
                accuracyScore += accuracy
                comparePOS = list(set(pred[:5]) & set(y_test[index][:5]))
                comparePWB = list(set(pred[5:]) & set(y_test[index][5:]))
                matchScore = predictionScore(pred[:5], y_test[index][:5]) + (predictionScore(pred[5:], y_test[index][5:])*6)
                if matchScore >= 6:
                    totalMatchScore += matchScore
                    #print(pred, y_test[index], comparePOS, comparePWB, matchScore)

                # Find the winner!
                if totalMatchScore >= winningScore:
                    winningScore = totalMatchScore
                    winningState = in_neighbors
            print('-----> Accuracy: ' + str(round(accuracyScore,4)) + ' | Match Score: ' + str(totalMatchScore) + ' <------')
            in_neighbors += iteration_throttle

        print('-----> Winner is ' + str(int(winningState)) + ' with ' + str(int(winningScore)) + ' <------')
        # Fitting KNN
        neigh = KNeighborsClassifier(weights=KNNWeights, algorithm=KNNAlgorithm, n_neighbors=int(winningState))
        neigh.fit(X_fit, y)

        # Predicting the next draw set results
        nextdraw_n_pred = neigh.predict([nextdraw_set])
        print(nextdraw_n_pred[0], nextdraw_set)
        
        nextDrawResults.append([{"numbers":nextdraw_n_pred[0]},{"score":int(winningScore)},{"class":"weather - KNeighborsClassifier - U"}])

        ##  KNN Classifier - distance ##
        winningScore = 0
        winningState = 0

        in_neighbors = 1
        while in_neighbors <= 154:
            # Fitting KNN
            KNNWeights = "distance" # weights{"uniform", "distance"}
            KNNAlgorithm = "auto" # algorithm{"auto", "ball_tree", "kd_tree", "brute"}
            neigh = KNeighborsClassifier(weights=KNNWeights, algorithm=KNNAlgorithm, n_neighbors=in_neighbors)
            neigh.fit(X_fit_train, y_train)

            # Predicting the Test set results
            y_n_pred = neigh.predict(X_fit_test)

            print('----- ' + str(in_neighbors) + ' ------')
            accuracyScore = 0
            totalMatchScore = 0
            for index, pred in enumerate(y_n_pred):
                accuracy = accuracy_score(y_test[index], pred)
                accuracyScore += accuracy
                comparePOS = list(set(pred[:5]) & set(y_test[index][:5]))
                comparePWB = list(set(pred[5:]) & set(y_test[index][5:]))
                matchScore = predictionScore(pred[:5], y_test[index][:5]) + (predictionScore(pred[5:], y_test[index][5:])*6)
                if matchScore >= 6:
                    totalMatchScore += matchScore
                    #print(pred, y_test[index], comparePOS, comparePWB, matchScore)
                
                # Find the winner!
                if totalMatchScore >= winningScore:
                    winningScore = totalMatchScore
                    winningState = in_neighbors

            print('-----> Accuracy: ' + str(round(accuracyScore,4)) + ' | Match Score: ' + str(totalMatchScore) + ' <------')
            in_neighbors += iteration_throttle

        print('-----> Winner is ' + str(int(winningState)) + ' with ' + str(int(winningScore)) + ' <------')
        # Fitting KNN
        neigh = KNeighborsClassifier(weights=KNNWeights, algorithm=KNNAlgorithm, n_neighbors=int(winningState))
        neigh.fit(X_fit, y)

        # Predicting the next draw set results
        nextdraw_n_pred = neigh.predict([nextdraw_set])
        print(nextdraw_n_pred[0], nextdraw_set)

        nextDrawResults.append([{"numbers":nextdraw_n_pred[0]},{"score":int(winningScore)},{"class":"weather - KNeighborsClassifier - D"}])

        ## MLPRegressor - lbfgs ##
        winningScore = 0
        winningState = 0

        in_randoms = 1
        while in_randoms <= 104:
            # Fitting MLPC
            MLPCSolver = "lbfgs" # solver{"lbfgs", "sgd", "adam"}, default=’adam’
            MLPclf = MLPRegressor(hidden_layer_sizes=(4,4), max_iter=1536, alpha=0.0001,  verbose=10,
                    solver=MLPCSolver, random_state=in_randoms, tol=0.000000001)
            MLPclf.fit(X_train, y_train)

            # Predicting the Test set results
            y_r_pred = MLPclf.predict(X_test)

            totalMatchScore = 0
            print('----- ' + str(in_randoms) + ' ------ ')
            for index, pred in enumerate(y_r_pred):
                pred = list(map(round, pred))
                comparePOS = list(set(pred[:5]) & set(y_test[index][:5]))
                comparePWB = list(set(pred[5:]) & set(y_test[index][5:]))
                matchScore = predictionScore(pred[:5], y_test[index][:5]) + (predictionScore(pred[5:], y_test[index][5:])*6)
                if matchScore >= 6:
                    totalMatchScore += matchScore
                    #print(pred, y_test[index], comparePOS, comparePWB, matchScore)#, round(accuracy,3), X_test[index])

                # Find the winner!
                if totalMatchScore >= winningScore:
                    winningScore = totalMatchScore
                    winningState = in_randoms

            print('-----> Match Score: ' + str(totalMatchScore) + ' <------')
            in_randoms += iteration_throttle

        print('-----> Winner is ' + str(int(winningState)) + ' with ' + str(int(winningScore)) + ' <------')
        # Fitting RNC
        MLPclf = MLPRegressor(hidden_layer_sizes=(4,4), max_iter=1536, alpha=0.0001,  verbose=10,
                solver=MLPCSolver, random_state=int(winningState), tol=0.000000001)
        MLPclf.fit(X, y)

        # Predicting the next draw set results
        nextdraw_MLpred = MLPclf.predict([nextdraw_set])
        nextdraw_MLpred[0] = list(map(round, nextdraw_MLpred[0]))
        print(nextdraw_MLpred[0], nextdraw_set)

        nextDrawResults.append([{"numbers":list(map(round, nextdraw_MLpred[0]))},{"score":int(winningScore)},{"class":"weather - MLPRegressor - L"}])


        ### Show the results ###
        # Order the results by the best scores
        print("PB", nextdraw_date.strftime("%Y-%m-%d"))
        #for predictions in nextDrawResults:
        #    print(list(map(int, predictions[0]['numbers'])), predictions[1]['score'], predictions[2]['class'])
        nextDrawResultsSorted = sorted(nextDrawResults, key=lambda k: k[1]['score'], reverse=True)
        #print(nextDrawResultsSorted)
        for predictions in nextDrawResultsSorted:
            print(list(map(int, predictions[0]['numbers'])), predictions[1]['score'], predictions[2]['class'])

        ### Write Top Pick to DB ###
        predicted_numbers_arr = nextDrawResultsSorted[0][0]['numbers']
        #print(predicted_numbers_arr)
        
        WeatherDrawing.objects.filter(draw_date=nextdraw_date.strftime("%Y-%m-%d")).update(pos1 = predicted_numbers_arr[0])
        WeatherDrawing.objects.filter(draw_date=nextdraw_date.strftime("%Y-%m-%d")).update(pos2 = predicted_numbers_arr[1])
        WeatherDrawing.objects.filter(draw_date=nextdraw_date.strftime("%Y-%m-%d")).update(pos3 = predicted_numbers_arr[2])
        WeatherDrawing.objects.filter(draw_date=nextdraw_date.strftime("%Y-%m-%d")).update(pos4 = predicted_numbers_arr[3])
        WeatherDrawing.objects.filter(draw_date=nextdraw_date.strftime("%Y-%m-%d")).update(pos5 = predicted_numbers_arr[4])
        WeatherDrawing.objects.filter(draw_date=nextdraw_date.strftime("%Y-%m-%d")).update(pwrb = predicted_numbers_arr[5])


