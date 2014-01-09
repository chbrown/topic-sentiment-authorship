import sys
import csv
import openpyxl


def read_xlsx(filepath, sheet_name=None, limit=None):
    with open(filepath, 'rb') as fp:
        # guess_types=False is implied by use_iterators=True
        workbook = openpyxl.load_workbook(fp, use_iterators=True)
        # sheets = workbook.worksheets
        sheet = workbook.get_sheet_by_name(sheet_name) if sheet_name else workbook.get_active_sheet()
        # print sheet.title

        rows = sheet.iter_rows()
        # each row in rows is a RawCell, which looks something like the following:
        # RawCell(row=2, column='A', coordinate='A2', internal_value=4.17036113778479e-05, data_type='n', style_id=None, number_format=None)
        # RawCell(row=3, column='J', coordinate='J3', internal_value=u'MISSING', data_type='s', style_id='2', number_format='General')
        # RawCell(row=4, column='D', coordinate='D4', internal_value=datetime.datetime(2011, 7, 11, 12, 43, 19, 1), data_type='n', style_id='3', number_format='mm-dd-yy')
        header = rows.next()
        keys = [cell.internal_value for cell in header]
        for i, row in enumerate(rows):
            if i == limit:
                break
            # [map(unicode, row) for row in all_rows]
            # value=value.encode('utf8')
            values = (cell.internal_value for cell in row)
            yield dict(zip(keys, values))


def read_tsv(filepath, limit=None):
    with open(filepath, 'rU') as fp:
        reader = csv.DictReader(fp, delimiter='\t', quotechar='"')
        for i, row in enumerate(reader):
            if i == limit:
                break
            yield row


class Printer(object):
    '''
    As with Awk:
        FS = field separator (defaults to tab)
        RS = record separator (defaults to newline)
    '''
    def __init__(self, headers=[], output=sys.stdout, FS='\t', RS='\n'):
        self.headers = headers
        self.output = output
        self.FS = FS
        self.RS = RS

    def formatter(self, value):
        if isinstance(value, basestring):
            return unicode(value)
        else:
            return '%0.3f' % value

    def write_strings(self, values):
        print >> self.output, self.FS.join([self.formatter(value) for value in values]), self.RS,

    def write(self, dict_):
        new_headers = set(dict_.keys()) - set(self.headers)
        if len(new_headers) > 0:
            self.headers.extend(new_headers)
            # ^^^ signals an updated headers row
            print >> self.output, '^^^',
            self.write_strings(self.headers)
        self.write_strings(dict_.get(key, '') for key in self.headers)
