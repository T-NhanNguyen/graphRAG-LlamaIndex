import os
import ast
import re

def extract_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    info = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            lineno = node.lineno # 1-indexed line number of the def line
            
            # Find the description
            description = ""
            # Try to get docstring first
            docstring = ast.get_docstring(node)
            if docstring:
                description = docstring.split('\n')[0].strip()
            else:
                # Look for comment immediately following the definition
                # We need to find the line where the body starts
                # Since the signature can span multiple lines, we look after node.lineno
                # until we find a line that is either a comment or code.
                
                # Simple heuristic: look at subsequent lines for a comment
                search_idx = lineno # next line (0-indexed line index is lineno-1)
                while search_idx < len(lines):
                    line = lines[search_idx].strip()
                    if not line:
                        search_idx += 1
                        continue
                    if line.startswith('#'):
                        description = line.lstrip('#').strip()
                        break
                    # If we find code before a comment, there's no description comment
                    break
            
            if isinstance(node, ast.ClassDef):
                type_str = "Class"
            else:
                type_str = "Func"
            
            info.append({
                "name": name,
                "type": type_str,
                "description": description
            })
    
    # Sort by line number to maintain order in file
    def get_lineno(item_name):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == item_name:
                return node.lineno
        return 0
    
    info.sort(key=lambda x: get_lineno(x['name']))
    
    return info

def generate_markdown_for_file(full_path, base_dir):
    file_info = extract_info(full_path)
    if not file_info:
        return ""
    
    md_output = []
    for item in file_info:
        desc = f": {item['description']}" if item['description'] else ""
        md_output.append(f"  - **{item['type']}** `{item['name']}`{desc}")
    return "\n".join(md_output)

def update_toc_file(toc_path, target_dirs):
    with open(toc_path, 'r', encoding='utf-8') as f:
        toc_content = f.read()
    
    # We'll use a regex to find each section and then each file within the section
    for dirname in target_dirs:
        # Find the section starting with ## dirname/
        section_pattern = rf'(## `?{dirname}/?`?[^\n]*\n)(.*?)(?=\n##|$)'
        match = re.search(section_pattern, toc_content, re.DOTALL)
        if not match:
            continue
            
        header, body = match.groups()
        new_body = body
        
        # Now find file entries in the body: - `filename.py`: description
        # We'll search for all .py files in the directory
        py_files = []
        if os.path.exists(dirname):
            for root, _, files in os.walk(dirname):
                for file in files:
                    if file.endswith('.py'):
                        py_files.append(os.path.join(root, file))
        
        for full_path in py_files:
            rel_path = os.path.relpath(full_path, dirname)
            # Find the line for this file in the TOC
            file_pattern = rf'(- `?{re.escape(rel_path)}`?:?[^\n]*)'
            file_match = re.search(file_pattern, new_body)
            
            class_func_md = generate_markdown_for_file(full_path, dirname)
            if not class_func_md:
                continue
                
            if file_match:
                # File exists, append class/func info under it
                original_line = file_match.group(1)
                # Check if we already added info (avoid duplicate runs)
                # This is a bit tricky, so let's just replace if we find a previous marker or if it's the first time
                # For now, let's just replace the whole file entry if it was previously generated or append.
                # Actually, the user says "rewrite", so let's just replace.
                
                # To be safe and "update", let's replace the block starting with this file until the next file or header
                block_pattern = rf'(- `?{re.escape(rel_path)}`?:?[^\n]*)(\n  - \*\*.*)*'
                replacement = original_line + "\n" + class_func_md
                new_body = re.sub(block_pattern, replacement, new_body)
            else:
                # File not in TOC, add it at the end of the section
                new_body += f"\n- `{rel_path}`\n{class_func_md}\n"
        
        # Replace the section in the toc_content
        toc_content = toc_content.replace(body, new_body)

    with open(toc_path, 'w', encoding='utf-8') as f:
        f.write(toc_content)

if __name__ == "__main__":
    toc_file = "dir_table_of_contents.md"
    target_dirs = ["core", "api", "deployment", "tools"]
    
    # Use update_toc_file which now tries to be more surgical
    update_toc_file(toc_file, target_dirs)
    print(f"Updated {toc_file}")
