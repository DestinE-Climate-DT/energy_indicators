Introduction
============

Welcome to the documentation for the **Energy Indicators** package! This introduction provides an overview of the package, its purpose, and how it fits into the broader ClimateDT landscape.

.. contents:: Table of Contents
   :local:
   :depth: 2

What is Energy Indicators?
--------------------------

The **Energy Indicators** package is designed to provide energy-relevant indicators for climate change adaptation. It is tightly integrated with the Climate DT technical structure, enabling execution in streaming mode — that is, concurrently with climate simulations. It offers standard indicators for the wind energy sector and energy demand, and will also include indicators for the solar energy sector.

Key Features
------------

- **Production of well-established energy indicators**: Indicators for energy production and energy demand.
- **Ability to run in streaming mode**: Process data continuously from the earth system modles from the Climate Adaptation Digital Twin.
- **Modularity**: New indicators can be easily added without modifying other parts of the package.
- **Broad accessibility and open-source development**: The package welcomes external suggestions and contributions and is intended as a tool for ClimateDT users, but not exclusively.

Installation
------------

Clone the repository and install it locally from source (soon to be open source). (Hint: consider creating a virtual environment first `python3 -m venv my_venv`):

.. code-block:: bash

   git clone https://earth.bsc.es/gitlab/digital-twins/de_340-2/energy_onshore.git
   cd energy_onshore
   pip install .

Getting Started
---------------

To quickly get started, import the package in Python:

.. code-block:: python

   import energy_onshore

For a step-by-step guide, visit the :doc:`tutorial`.

Next Steps
----------

- See :doc:`how_to_contribute` if you'd like to contribute to the package.
- Visit the :doc:`about` section to understand the project's goals.
- Read the :doc:`testing` section to see the unit test coverage.

