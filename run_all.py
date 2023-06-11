from prepare_data import make_X_and_Y
from nn import run_nn
from test_nn import main_test

make_X_and_Y(1)
run_nn(iterations=100)
main_test(1)
