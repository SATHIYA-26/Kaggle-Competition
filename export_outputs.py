import json
import re

def clean_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def export_notebook(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    lines = []
    lines.append('=' * 90)
    lines.append(f'NOTEBOOK CELL-BY-CELL CODE & OUTPUT ANALYSIS: {notebook_path}')
    lines.append('=' * 90 + '\n')

    cells = nb.get('cells', [])
    for idx, cell in enumerate(cells, 1):
        cell_type = cell.get('cell_type', 'unknown').upper()
        exec_count = cell.get('execution_count', None)
        exec_str = f" [Execution Count: {exec_count}]" if exec_count is not None else ""

        lines.append('=' * 90)
        lines.append(f'CELL #{idx} | TYPE: {cell_type}{exec_str}')
        lines.append('=' * 90)
        
        # Source
        source = ''.join(cell.get('source', []))
        if cell_type == 'MARKDOWN':
            lines.append('--- [MARKDOWN / EXPLANATION / QUESTION] ---')
        else:
            lines.append('--- [CODE / INPUT] ---')
            
        lines.append(source if source.strip() else '<Empty>')
        lines.append('')

        # Outputs
        if cell_type == 'CODE':
            outputs = cell.get('outputs', [])
            lines.append('--- [OUTPUT / RESULTS] ---')
            if not outputs:
                lines.append('<No output or Cell was not executed>')
            else:
                for out_idx, out in enumerate(outputs, 1):
                    out_type = out.get('output_type', 'unknown')
                    
                    if out_type == 'stream':
                        stream_name = out.get('name', 'stdout')
                        text = clean_ansi(''.join(out.get('text', [])))
                        lines.append(f'>> Stream ({stream_name}):')
                        lines.append(text.rstrip())
                    elif out_type in ('execute_result', 'display_data'):
                        data = out.get('data', {})
                        if 'text/plain' in data:
                            text = clean_ansi(''.join(data['text/plain']))
                            lines.append(f'>> Result (text/plain):')
                            lines.append(text.rstrip())
                        if 'image/png' in data:
                            lines.append('>> [Image / Plot Output: PNG rendering generated in notebook]')
                        if 'text/html' in data:
                            lines.append('>> [HTML / Table Output rendered]')
                    elif out_type == 'error':
                        ename = out.get('ename', 'Error')
                        evalue = out.get('evalue', '')
                        lines.append(f'>> ERROR: {ename}: {evalue}')
                        tb = [clean_ansi(t) for t in out.get('traceback', [])]
                        if tb:
                            lines.append('\n'.join(tb))
            lines.append('')
        lines.append('\n')

    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write('\n'.join(lines))

    print(f'Successfully exported {len(cells)} cells to {output_path}')

if __name__ == '__main__':
    export_notebook('prediction.ipynb', 'notebook_cells_and_outputs.txt')
