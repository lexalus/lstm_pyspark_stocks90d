ENV=mlenv
conda create -y -n $ENV python=3.6
conda install -y -n $ENV -c conda-forge pandas pandas-gbq numpy scikit-learn tensorflow keras yahoo-finance
source activate $ENV