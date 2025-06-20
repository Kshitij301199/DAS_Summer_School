# This is sh (dash) by default, not $SHELL
c.NotebookApp.terminado_settings = { "shell_command": ["bash"] }

c.ServerProxy.servers = {
  'http-server': {
    'command': ['python3', '-m', 'http.server', '{port}'],
    'absolute_url': False,
    'launcher_entry': {
      'title': "HTTP Server"
    }
  }
}

# adding this to try and get static build content being served Jupyter sessions launched from my gohugo-binder,
# based on https://github.com/martinagvilas/jupytercon_tutorial/pull/39 (that I liearned ov via pulse,
# https://github.com/martinagvilas/jupytercon_tutorial/pulse , when I was forking to try my 
# crazy idea spelled out at https://github.com/martinagvilas/jupytercon_tutorial/issues/14#issuecomment-702805680)
# and 
# https://github.com/martinagvilas/jupytercon_tutorial/issues/14#issuecomment-695891172
# This seems easier than propogatating the token to the css and javascript references, as suggested https://github.com/martinagvilas/jupytercon_tutorial/issues/14#issuecomment-701954323, if it works.