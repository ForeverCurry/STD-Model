"""
Common settings
"""
import os

STORAGE=os.path.abspath(os.path.join(os.getcwd()))

DATASETS_PATH=os.path.join(STORAGE, 'datasets')
EXPERIMENTS_PATH=os.path.join(STORAGE, 'experiments')
RESULT_PATH=os.path.join(STORAGE,'results')
TESTS_STORAGE_PATH=os.path.join(STORAGE, 'test')
SAVE_PATH = os.path.join(STORAGE,"summary")
SEED :int = 230823
