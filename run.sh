## Lorenz Noisy data Experiments ##

python3 ./experiments/lorenz/lorenz_exp.py  --noise 0.5 --input_size 27 

## Real-world data experiments ##

#### Osaka wind data ####
python3 ./experiments/wind/wind_exp.py  --target 105

#### Fukushima wind data ####
python3 ./experiments/wind/wind_exp.py  --target 24

#### Operative temperature data ####
python3 ./experiments/weather/weather_exp.py  --target 15

#### Plankton N4 data ####
python3 ./experiments/plankton/plankton_exp.py  --target 3

#### plankton N12 data ####
python3 ./experiments/plankton/plankton_exp.py  --target 11

## refine experiments ##

#### Lorenz noisy data ####

###### refine ETS model ######
python3 ./experiments/lorenz/lorenz_exp.py  --noise 0.5 --input_size 27 --refine --refine_model ETS
###### refine ARIMA model #### 
python3 ./experiments/lorenz/lorenz_exp.py  --noise 0.5 --input_size 27 --refine --refine_model ARIMA
###### refine Theta model #### 
python3 ./experiments/lorenz/lorenz_exp.py  --noise 0.5 --input_size 27 --refine --refine_model Theta 
###### refine ARNN model #### 
python3 ./experiments/lorenz/lorenz_exp.py  --noise 0.5 --input_size 27 --refine --refine_model ARNN
###### refine RDE model #### 
python3 ./experiments/lorenz/lorenz_exp.py  --noise 0.5 --input_size 27 --refine --refine_model RDE

#### Osaka wind data ####

###### refine ETS model ######
python3 ./experiments/wind/wind_exp.py  --target 105 --refine --refine_model ETS
###### refine ARIMA model #### 
python3 ./experiments/wind/wind_exp.py  --target 105 --refine --refine_model ARIMA
###### refine Theta model #### 
python3 ./experiments/wind/wind_exp.py  --target 105 --refine --refine_model Theta 
###### refine ARNN model #### 
python3 ./experiments/wind/wind_exp.py  --target 105 --refine --refine_model ARNN 
###### refine RDE model #### 
python3 ./experiments/wind/wind_exp.py  --target 105 --refine --refine_model RDE

#### Fukushima wind data ####

###### refine ETS model ######
python3 ./experiments/wind/wind_exp.py  --target 24 --refine --refine_model ETS   
###### refine ARIMA model #### 
python3 ./experiments/wind/wind_exp.py  --target 24 --refine --refine_model ARIMA 
###### refine Theta model #### 
python3 ./experiments/wind/wind_exp.py  --target 24 --refine --refine_model Theta 
###### refine ARNN model #### 
python3 ./experiments/wind/wind_exp.py  --target 24 --refine --refine_model ARNN
###### refine RDE model #### 
python3 ./experiments/wind/wind_exp.py  --target 24 --refine --refine_model RDE


#### Operative temperature data ####

###### refine ETS model ######
python3 ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model ETS  
###### refine ARIMA model #### 
python3 ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model ARIMA 
###### refine Theta model #### 
python3 ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model Theta 
###### refine ARNN model #### 
python3 ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model ARNN 
###### refine RDE model #### 
python3 ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model RDE 

#### Plankton N4 data ####

###### refine ETS model ######
python3 ./experiments/plankton/plankton_exp.py  --target 3 --refine --refine_model ETS   
###### refine ARIMA model #### 
python3 ./experiments/plankton/plankton_exp.py  --target 3 --refine --refine_model ARIMA 
###### refine Theta model #### 
python3 ./experiments/plankton/plankton_exp.py  --target 3 --refine --refine_model Theta
###### refine ARNN model #### 
python3 ./experiments/plankton/plankton_exp.py  --target 3 --refine --refine_model ARNN 
###### refine RDE model ###### 
python3 ./experiments/plankton/plankton_exp.py  --target 3 --refine --refine_model RDE     

#### plankton N12 data ####

###### refine ETS model ######
python3 ./experiments/plankton/plankton_exp.py  --target 11 --refine --refine_model ETS   
###### refine ARIMA model ######
python3 ./experiments/plankton/plankton_exp.py  --target 11 --refine --refine_model ARIMA 
###### refine Theta model ######
python3 ./experiments/plankton/plankton_exp.py  --target 11 --refine --refine_model Theta
###### refine ARNN model ##### 
python3 ./experiments/plankton/plankton_exp.py  --target 11 --refine --refine_model ARNN
###### refine RDE model ##### 
python3 ./experiments/plankton/plankton_exp.py  --target 11 --refine --refine_model RDE  