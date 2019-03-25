
# Types of templates after cleaning
types = {0: 'Noise', 1: 'SS', 2: 'SS_Contaminated', 3: 'SS_Putative', 4: 'MUA', 5: 'Unspesified_1',
         6: 'Unspecified_2',
         7: 'Unspecified_3'}


# spike_info column names
ORIGINAL_INDEX = 'original_index'
TIMES = 'times'
TEMPLATE_AFTER_CLEANING = 'template_after_cleaning'
TYPE_AFTER_CLEANING = 'type_after_cleaning'
TEMPLATE_AFTER_SORTING = 'template_after_sorting'
TYPE_AFTER_SORTING = 'type_after_sorting'
TEMPLATE_WITH_ALL_SPIKES_PRESENT = 'template_with_all_spikes_present'
TSNE_FILENAME = 'tsne_filename'
TSNE_X = 'tsne_x'
TSNE_Y = 'tsne_y'
PROBE_POSITION_X = 'probe_position_x'
PROBE_POSITION_Y = 'probe_position_z'

# file names
INDICES_OF_SPIKES_USED_FILENAME = 'indices_of_spikes_used.npy'
INDICES_OF_SMALL_TEMPLATES_FILENAME = 'indices_of_small_templates.npy'
INDICES_OF_LARGE_TEMPLATES_FILENAME = 'indices_of_large_templates.npy'
WEIGHTED_TEMPLATE_POSITIONS_FILENAME = 'weighted_template_positions.npy'
WEIGHTED_SPIKE_POSITIONS_FILENAME = 'weighted_spike_positions.npy'
SMALL_CLEAN_TEMPLATES_WITH_SPIKE_INDICES_PICKLE = "small_clean_templates_with_spike_indices.pkl"
LARGE_CLEAN_TEMPLATES_WITH_SPIKE_INDICES_PICKLE = "large_clean_templates_with_spike_indices.pkl"

TEMPLATE_MARKING_FILENAME = 'template_marking.npy'
SPIKE_TEMPLATES_FILENAME = 'spike_templates.npy'
SPIKE_TIMES_FILENAME = 'spike_times.npy'
TEMPLATE_FEATURES_FILENAME = 'template_features.npy'
TEMPLATE_FEATURES_INDEX_FILENAME = 'template_feature_ind.npy'
TEMPLATE_POSITIONS_FILENAME = 'weighted_template_positions.npy'
TEMPLATES_FILENAME = 'templates.npy'
SPIKE_INFO_FILENAME = 'spike_info.df'
SPIKE_INFO_AFTER_CLEANING_FILENAME = 'spike_info_after_cleaning.df'
SPIKE_INFO_AFTER_SORTING_FILENAME = 'spike_info_after_sorting.df'
TEMPLATE_INFO_FILENAME = 'template_info.df'
