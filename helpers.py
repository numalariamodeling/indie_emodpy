

def scale_habitats_from_file(cb, habitat_df) :

    # handle node-specific habitat scaling
    hab_list = []
    for species_params in cb.get_param("Vector_Species_Params"):
        habitats = species_params["Larval_Habitat_Types"]
        hab_list += [h for (h, v) in habitats.items()]
    hab_list = list(set(hab_list))

    for hab_type in ['TEMPORARY_RAINFALL', 'CONSTANT', 'LINEAR_SPLINE', 'WATER_VEGETATION'] :
        if hab_type in hab_list :
            habitat_df[hab_type] = copy.copy(habitat_df['habitat_scale'])
    del habitat_df['habitat_scale']
    scale_larval_habitats(cb, habitat_df)