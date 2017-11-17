from tsa import logging, stdout, stderr, root
import geo
import geo.types
import geo.shapefile.reader
import geo.shapefile.writer
import geojson

logger = logging.getLogger(__name__)


def main():
    shapefile_writer = shapefile.Writer()
    shapefile_writer.fields = shapefile_reader.field

    for record, shape in zip(shapefile_reader.iterRecords(), shapefile_reader.iterShapes()):
        attributes = dict(zip(field_names, record))

        #parts = shape.parts.tolist() + [len(shape.points)]
        #polygons = [shape.points[i:j] for i, j in zip(parts, parts[1:])]

        # yield polygons, shape.bbox, attributes
        if attributes['STATE_NAME'] == 'Ohio':
            print(attributes, shape.parts)
            shapefile_writer.record(*record)
            shapefile_writer.poly(parts=[shape.points])
    writer.save('/Users/chbrown/corpora-public/census-shapefiles/ESRI-Ohiocounties')
