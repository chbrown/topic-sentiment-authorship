import IPython
from IPython.terminal.embed import InteractiveShellEmbed

# use like:
# from tsa.lib.debugger import shell
# shell(); raise SystemExit(99)

config = IPython.Config()
config.InteractiveShell.confirm_exit = False
config.InteractiveShell.pylab = 'osx'

start_message = 'Dropping into IPython shell'
end_message = 'Leaving IPython shell, resuming program.'

shell = InteractiveShellEmbed(config=config,
    banner1=start_message, exit_msg=end_message)
