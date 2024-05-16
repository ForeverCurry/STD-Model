## Lorenz Noisy data Experiments ##

python3 ./experiments/lorenz/lorenz_exp.py  --noise 0.5 --input_size 27 

## Real-world data experiments ##

#### wind data ####
python3 ./experiments/wind/wind_exp.py  

#### Operative temperature data ####
python3 ./experiments/weather/weather_exp.py  --target 15

#### Plankton data ####
python3 ./experiments/plankton/plankton_exp.py 

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
###### refine MVE model #### 
python3 ./experiments/lorenz/lorenz_exp.py  --noise 0.5 --input_size 27 --refine --refine_model MVE

#### wind data ####

###### refine ETS model ######
python3 ./experiments/wind/wind_exp.py   --refine --refine_model ETS
###### refine ARIMA model #### 
python3 ./experiments/wind/wind_exp.py   --refine --refine_model ARIMA
###### refine Theta model #### 
python3 ./experiments/wind/wind_exp.py   --refine --refine_model Theta 
###### refine ARNN model #### 
python3 ./experiments/wind/wind_exp.py   --refine --refine_model ARNN 
###### refine RDE model #### 
python3 ./experiments/wind/wind_exp.py   --refine --refine_model RDE
###### refine MVE model #### 
python3 ./experiments/wind/wind_exp.py   --refine --refine_model MVE

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
###### refine MVE model #### 
python3 ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model MVE
#### Plankton data ####

###### refine ETS model ######
python3 ./experiments/plankton/plankton_exp.py   --refine --refine_model ETS   
###### refine ARIMA model #### 
python3 ./experiments/plankton/plankton_exp.py   --refine --refine_model ARIMA 
###### refine Theta model #### 
python3 ./experiments/plankton/plankton_exp.py   --refine --refine_model Theta
###### refine ARNN model #### 
python3 ./experiments/plankton/plankton_exp.py   --refine --refine_model ARNN 
###### refine RDE model ###### 
python3 ./experiments/plankton/plankton_exp.py   --refine --refine_model RDE     
###### refine MVE model #### 
python3 ./experiments/plankton/plankton_exp.py   --refine --refine_model MVE
