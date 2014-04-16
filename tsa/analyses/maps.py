import IPython
import numpy as np
import pandas as pd
import os
import subprocess
from tsa.science import numpy_ext as npx

from collections import Counter
# from datetime import datetime

import viz
from viz.geom import hist

from sklearn import metrics, cross_validation
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm
# from sklearn import cluster, decomposition, ensemble, neighbors, neural_network, qda
# from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from tsa.lib.text import CountVectorizer
# from sklearn.feature_selection import SelectPercentile, SelectKBest
# from sklearn.feature_selection import chi2, f_classif, f_regression

from tsa import stdout, stderr, root
from tsa.lib import tabular, datetime_extra
from tsa.lib.timer import Timer
from tsa.models import Source, Document, create_session
from tsa.science import features, models, timeseries
from tsa.science.corpora import MulticlassCorpus
from tsa.science.plot import plt, figure_path, distinct_styles, ticker
from tsa.science.summarization import metrics_dict, metrics_summary
# from tsa.science.summarization import explore_mispredictions, explore_uncertainty
import geo
import geo.types
import geo.shapefile.reader
import geo.shapefile.writer
import geojson

from tsa import logging
logger = logging.getLogger(__name__)


# shapefile_writer = shapefile.Writer()
# shapefile_writer.fields = shapefile_reader.field

# for record, shape in zip(shapefile_reader.iterRecords(), shapefile_reader.iterShapes()):
#     attributes = dict(zip(field_names, record))

#     #parts = shape.parts.tolist() + [len(shape.points)]
#     #polygons = [shape.points[i:j] for i, j in zip(parts, parts[1:])]

#     # yield polygons, shape.bbox, attributes
#     if attributes['STATE_NAME'] == 'Ohio':
#         print attributes, shape.parts
#         shapefile_writer.record(*record)
#         shapefile_writer.poly(parts=[shape.points])
# writer.save('/Users/chbrown/corpora-public/census-shapefiles/ESRI-Ohiocounties')
