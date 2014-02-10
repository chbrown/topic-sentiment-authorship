import inspect
import pkgutil
import tsa.analyses


analyses = {}
for imp_importer, name, ispkg in pkgutil.iter_modules(tsa.analyses.__path__):
    fullname = tsa.analyses.__name__ + '.' + name
    imp_loader = imp_importer.find_module(fullname)
    module = imp_loader.load_module(fullname)

    # available analyses will include only methods in tsa.analyses.**.*
    # that take a positional argument called 'analysis_options'

    for member_name, member_value in inspect.getmembers(module):
        if inspect.isfunction(member_value):
            argspec = inspect.getargspec(member_value)
            if 'analysis_options' in argspec.args:
                # analyses[name + '.' + member_name] = member_value
                analyses[member_name] = member_value


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Primary entry point for tsa analyses')
    parser.add_argument('analysis', choices=analyses,
        help='Analysis to run')
    parser.add_argument('--verbose', action='store_true',
        help='Log extra information')
    # parser.add_argument('--version', action='version',
    #     version=tsa.__version__)
    opts, _ = parser.parse_known_args()

    # configuration
    from viz import terminal
    import numpy as np
    np.set_printoptions(edgeitems=25, threshold=100, linewidth=terminal.width())
    import pandas as pd
    pd.options.display.max_rows = 200
    pd.options.display.max_columns = 25
    pd.options.display.width = terminal.width()
    import logging
    level = logging.DEBUG if opts.verbose else logging.INFO
    logging.root.level = level  # SILLY < 10 <= DEBUG
    # logger.critical('logger init: root.level = %s, logger.level = %s', logging.root.level, logger.level)

    analyses[opts.analysis](opts)


if __name__ == '__main__':
    main()
