import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from dicts import learn_city_region  # noqa: E402
learn_city_region()
