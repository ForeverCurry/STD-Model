## Lorenz Noisy data Experiments ##

python ./experiments/lorenz/lorenz_exp.py  --noisy 0.5 --input_size 27 

## Real-world data experiments ##

#### wind data ####
python ./experiments/wind/wind_exp.py  

#### Operative temperature data ####
python ./experiments/weather/weather_exp.py 

#### Plankton data ####
python ./experiments/plankton/plankton_exp.py 

## refine experiments ##

#### Lorenz noisy data ####

###### refine ETS model ######
python ./experiments/lorenz/lorenz_exp.py  --noisy 0.5 --input_size 27 --refine --refine_model ETS
###### refine ARIMA model #### 
python ./experiments/lorenz/lorenz_exp.py  --noisy 0.5 --input_size 27 --refine --refine_model ARIMA
###### refine Theta model #### 
python ./experiments/lorenz/lorenz_exp.py  --noisy 0.5 --input_size 27 --refine --refine_model Theta 
###### refine ARNN model #### 
python ./experiments/lorenz/lorenz_exp.py  --noisy 0.5 --input_size 27 --refine --refine_model ARNN
###### refine RDE model #### 
python ./experiments/lorenz/lorenz_exp.py  --noisy 0.5 --input_size 27 --refine --refine_model RDE
###### refine MVE model #### 
python ./experiments/lorenz/lorenz_exp.py  --noisy 0.5 --input_size 27 --refine --refine_model MVE

#### wind data ####

###### refine ETS model ######
python ./experiments/wind/wind_exp.py   --refine --refine_model ETS
###### refine ARIMA model #### 
python ./experiments/wind/wind_exp.py   --refine --refine_model ARIMA
###### refine Theta model #### 
python ./experiments/wind/wind_exp.py   --refine --refine_model Theta 
###### refine ARNN model #### 
python ./experiments/wind/wind_exp.py   --refine --refine_model ARNN 
###### refine RDE model #### 
python ./experiments/wind/wind_exp.py   --refine --refine_model RDE
###### refine MVE model #### 
python ./experiments/wind/wind_exp.py   --refine --refine_model MVE

#### Operative temperature data ####

###### refine ETS model ######
python ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model ETS  
###### refine ARIMA model #### 
python ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model ARIMA 
###### refine Theta model #### 
python ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model Theta 
###### refine ARNN model #### 
python ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model ARNN 
###### refine RDE model #### 
python ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model RDE 
###### refine MVE model #### 
python ./experiments/weather/weather_exp.py  --target 15 --refine --refine_model MVE
#### Plankton data ####

###### refine ETS model ######
python ./experiments/plankton/plankton_exp.py   --refine --refine_model ETS   
###### refine ARIMA model #### 
python ./experiments/plankton/plankton_exp.py   --refine --refine_model ARIMA 
###### refine Theta model #### 
python ./experiments/plankton/plankton_exp.py   --refine --refine_model Theta
###### refine ARNN model #### 
python ./experiments/plankton/plankton_exp.py   --refine --refine_model ARNN 
###### refine RDE model ###### 
python ./experiments/plankton/plankton_exp.py   --refine --refine_model RDE     
###### refine MVE model #### 
python ./experiments/plankton/plankton_exp.py   --refine --refine_model MVE
