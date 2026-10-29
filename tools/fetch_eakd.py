"""Retrieve two verbatim author functions from an immutable official revision.

EA-KD's author repository currently has no explicit license. This downloader
does not grant redistribution rights; review upstream terms/authorization.
The resulting external file is intentionally excluded from the public archive.
"""
import ast
import json
from pathlib import Path
import urllib.request

COMMIT = '6fb4c0657d0af422e446a508f86705455f4c4a4b'
URL = 'https://raw.githubusercontent.com/cpsu00/EA-KD/'+COMMIT+'/mdistiller/distillers/KD.py'


def main():
    request = urllib.request.Request(URL, headers={'User-Agent':'Astra-reproduction'})
    with urllib.request.urlopen(request, timeout=60) as response:
        source = response.read().decode()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {'get_w','kd_loss'}]
    if {node.name for node in selected} != {'get_w','kd_loss'}:
        raise RuntimeError('Pinned upstream functions were not found')
    output = Path(__file__).resolve().parents[1]/'external'
    output.mkdir(exist_ok=True)
    text = '# Verbatim functions from '+URL+'\nimport torch\nimport torch.nn.functional as F\n\n'
    text += '\n\n'.join(''.join(lines[node.lineno-1:node.end_lineno]) for node in selected)
    (output/'EA_KD_author.py').write_text(text)
    (output/'origin.json').write_text(json.dumps(dict(url=URL,commit=COMMIT,
        license_status='No explicit upstream license found; not relicensed by this project'),indent=2)+'\n')
    print(output/'EA_KD_author.py')


if __name__=='__main__':
    main()
