## Figure in main text

#### Noisy Lorenz system results
python ./common/Plot/lorenz_main.py 

#### Iteration process figure
python ./common/Plot/itr_main.py

#### Coefficient A figure
python ./common/Plot/coef_main.py

#### Refine results
python ./common/Plot/refine_main.py

#### Real-world dataset figure
python ./common/Plot/real_result_plot.py 
## Figure in SI

#### Coefficient A figure
python ./common/Plot/coef_plot.py

#### Refine results plot in supplement material
python ./common/Plot/refine_plot.py -d Lorenz -i 27 -o 12 -id 25 -id 35 -id 45 

python ./common/Plot/refine_plot.py -d weather -i 14 -o 6 -id 25 -id 35 -id 45 

python ./common/Plot/refine_plot.py -d N4 -i 10 -o 4 -id 25 -id 30 -id 45

python ./common/Plot/refine_plot.py -d N12 -i 10 -o 4 -id 28 -id 35 -id 40 

python ./common/Plot/refine_plot.py -d Osaka -i 48 -o 12 -id 50 -id 80 -id 90

python ./common/Plot/refine_plot.py -d Fukushima -i 48 -o 12 -id 50 -id 80 -id 90

#### real_world results plot 

python ./common/Plot/real_result_plot.py
