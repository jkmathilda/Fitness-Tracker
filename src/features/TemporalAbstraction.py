##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 22-12-2022

import numpy as np

# Class to abstract a history of numerical values we can use as an attribute.
class NumericalAbstraction:

    # Abstract numerical columns specified given a window size (i.e. the number of time points from
    # the past considered) and an aggregation function.
    # Uses native pandas rolling methods for performance (10-100x faster than .apply()).
    def abstract_numerical(self, data_table, cols, window_size, aggregation_function):

        for col in cols:
            col_name = col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
            rolling = data_table[col].rolling(window_size)

            if aggregation_function == "mean":
                data_table[col_name] = rolling.mean()
            elif aggregation_function == "max":
                data_table[col_name] = rolling.max()
            elif aggregation_function == "min":
                data_table[col_name] = rolling.min()
            elif aggregation_function == "median":
                data_table[col_name] = rolling.median()
            elif aggregation_function == "std":
                data_table[col_name] = rolling.std()
            else:
                data_table[col_name] = np.nan

        return data_table