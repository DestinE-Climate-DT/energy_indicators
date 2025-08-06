Tutorial
========

My first run
------------

To perform your first run we will use the sample test data that comes with the package. This data is located in the `test_data/` directory.

.. code-block:: python
   
   from energy_onshore import run_capacity_factor_i
   run_capacity_factor_i("2027","11","14","2027","11","14","test_data/",".")
   # This will run the capacity factor indicator for turbine type I for the day 2027/11/14 given date range using the test data.

