My First Dummy Run
==================

Installation
------------

Clone the repository and install it locally from source (soon to be open source). (Hint: consider creating a virtual environment first `python3 -m venv my_venv`):

.. code-block:: bash

   git clone https://earth.bsc.es/gitlab/digital-twins/de_340-3/energy_indicators.git
   cd energy_indicators
   pip install .

   # to install it in edit mode, use `pip install -e .`.

My first run
------------

To perform your first run we will use the sample test data that comes with the package. This data is located in the `test_data/` directory.

.. code-block:: python
   
   from energy_indicators import run_capacity_factor_i
   run_capacity_factor_i("1990","01","01","1990","01","01","test_data/",".")

This will run the capacity factor indicator for turbine type I for the day 1990/01/01 given date range using the test data.
