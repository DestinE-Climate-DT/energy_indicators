General Description
===================

Project Overview
----------------
This documentation provides all the information needed to understand how
the Energy Indicators package works. It is structured into two main
sections, separating user usage from development aspects.

The **Energy Indicators** package provides energy-relevant indicators for
climate change adaptation, tailored for use in conjunction with the
ClimateDT workflow but also capable of running standalone. It is tightly
integrated with the Climate DT technical structure, enabling execution in
streaming mode — that is, concurrently with climate simulations. It offers
standard indicators for the wind energy sector and energy demand, and will
also include indicators for the solar energy sector.

Key Features
------------
- **Production of well-established energy indicators**: Indicators for energy production and energy demand.
- **Ability to run in streaming mode**: Process data continuously from the earth system modles from the Climate Adaptation Digital Twin.
- **Modularity**: New indicators can be easily added without modifying other parts of the package.
- **Broad accessibility and open-source development**: The package welcomes external suggestions and contributions and is intended as a tool for ClimateDT users, but not exclusively.

Relation to DestinE
-------------------
The current package Energy Indicators is developed as part of the `Destination Earth (DestinE) initiative <https://digital-strategy.ec.europa.eu/en/policies/destination-earth>`_, which aims to create a digital twin of the Earth to support climate change adaptation and mitigation efforts. The Energy Indicators application is integrated into the DestinE Climate Digital Twin (Climate DT) workflow, providing climate-derived metrics for the wind and solar energy sectors. By translating kilometre-scale climate simulations into actionable indicators, the application supports informed decision-making for near- to mid-term adaptation to climate change in the renewable energy sector.

Main developers
---------------
- Aleks Lacima-Nadolnik (BSC)
- Francesc Roura-Adserias (BSC)
- Sushovan Ghosh (BSC)
- Katherine Grayson (BSC)
- Christian Jané-Ippel (BSC)

High-Level Architecture
-----------------------

.. code-block:: text

    .
    ├── core.py                  # generic array/numerical utilities
    ├── demand.py                # demand-related indicators (heating/cooling degree days)
    ├── __init__.py
    ├── mask_processing.py       # utilities for land-sea mask handling
    ├── plot.py                  # plotting functionality
    ├── power_curves/            # power curves of representative turbines
    ├── run_energy_indicators.py # wrapper functions to run the lower-level indicators
    ├── solar.py                 # solar (energy production) indicators
    ├── utils.py                 # generic functions to check or convert physical variables
    └── wind.py                  # wind production and wind statistics indicators

Publications
------------

.. rubric:: References

Doblas-Reyes, F. J., Kontkanen, J., Sandu, I., Acosta, M., Al Turjmam, M. H., Alsina-Ferrer, I., Andrés-Martínez, M., Anerdi, C., Arriola, L., Axness, M., Batlle Martín, M., Bauer, P., Becker, T., Beltrán, D., Beyer, S., Bockelmann, H., Bretonnière, P.-A., Cabaniols, S., Caprioli, S., Castrillo, M., Chandrasekar, A., Cheedela, S., Correal, V., Danovaro, E., Davini, P., Enkovaara, J., Frauen, C., Früh, B., Gaya Àvila, A., Ghinassi, P., Ghosh, R., Ghosh, S., González, I., Grayson, K., Griffith, M., Hadade, I., Haine, C., Hartick, C., Haus, U.-U., Hearne, S., Järvinen, H., Jiménez, B., John, A., Juchem, M., Jung, T., Kegel, J., Kelbling, M., Keller, K., Kinoshita, B., Kiszler, T., Klocke, D., Kluft, L., Koldunov, N., Kölling, T., Kolstela, J., Kornblueh, L., Kosukhin, S., Lacima-Nadolnik, A., Leal Rojas, J. J., Lehtiranta, J., Lunttila, T., Luoma, A., Manninen, P., Medvedev, A., Milinski, S., Mohammed, A., Müller, S., Naryanappa, D., Nazarova, N., Niemelä, S., Niraula, B., Nortamo, H., Nummelin, A., Nurisso, M., Ortega, P., Paronuzzi, S., Pedruzo-Bagazgoitia, X., Pelletier, C., Peña, C., Polade, S., Pradhan, H. K., Quintanilla, R., Quintino, T., Rackow, T., Räisänen, J., Rajput, M. M., Redler, R., Reuter, B., Rocha Monteiro, N., Roura-Adserias, F., Ruppert, S., Sayed, S., Schnur, R., Sharma, T., Sidorenko, D., Sievi-Korte, O., Soret, A., Steger, C., Stevens, B., Streffing, J., Sunny, J., Tenorio, L., Thober, S., Tigerstedt, U., Tinto, O., Tonttila, J., Tuomenvirta, H., Tuppi, L., Van Thielen, G., Vitali, E., von Hardenberg, J., Wagner, I., Wedi, N., Wehner, J., Willner, S., Yepes-Arbós, X., Ziemen, F., and Zimmermann, J.: The Destination Earth digital twin for climate change adaptation, Geosci. Model Dev., 19, 2821–2848, https://doi.org/10.5194/gmd-19-2821-2026, 2026.

Grayson, K., Thober, S., Lacima-Nadolnik, A., Alsina-Ferrer, I., Lledó, L., Sharifi, E., & Doblas-Reyes, F. (2025). Statistical summaries for streamed data from climate simulations: one-pass algorithms. Geoscientific Model Development, 18(17), 5873–5890. https://doi.org/10.5194/gmd-18-5873-2025
