def load_turbines():
    """
    This is used to load the specifications of the different availalbe turbines in the energy indicators package
    Input

    Output: list of dictionaries with the specifications of the different turbines
    """
    class_i = {
        "turbine_model": "Enercon E70",
        "rotor_diameter": 71,
        "rated_power": 2300,
        "hub_height": 85,
        "cut_in_speed": 2.0,
        "rated_speed": 15.5,
        "cut_out_speed": 25.0,
    }
    class_i_ii = {
        "turbine_model": "Gamesa G80",
        "rotor_diameter": 80,
        "rated_power": 2000,
        "hub_height": 80,
        "cut_in_speed": 3.5,
        "rated_speed": 15.0,
        "cut_out_speed": 25.0,
    }
    class_ii = {
        "turbine_model": "Gamesa G87",
        "rotor_diameter": 87,
        "rated_power": 2000,
        "hub_height": 83.5,
        "cut_in_speed": 3.0,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }
    class_ii_iii = {
        "turbine_model": "Vestas V100",
        "rotor_diameter": 100,
        "rated_power": 2000,
        "hub_height": 100,
        "cut_in_speed": 3.5,
        "rated_speed": 12.0,
        "cut_out_speed": 20.0,
    }
    class_iii = {
        "turbine_model": "Vestas V110",
        "rotor_diameter": 110,
        "rated_power": 2000,
        "hub_height": 100,
        "cut_in_speed": 4.0,
        "rated_speed": 12.0,
        "cut_out_speed": 20.0,
    }
    class_s = {
        "turbine_model": "Vestas V164",
        "rotor_diameter": 164,
        "rated_power": 9500,
        "hub_height": 105,
        "cut_in_speed": 3.5,
        "rated_speed": 14.0,
        "cut_out_speed": 25.0,
    }
    
    turbine_class = [class_i, class_i_ii, class_ii, class_ii_iii, class_iii, class_s]
    
    return turbine_class