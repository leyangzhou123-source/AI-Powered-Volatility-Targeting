import os


def list_files(startpath):
    exclude = {'.git', '__pycache__', 'venv', '.ipynb_checkpoints', 'results'}

    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in exclude]

        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}📂 {os.path.basename(root)}/')

        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if not f.startswith('.'):
                print(f'{sub_indent} {f}')


if __name__ == "__main__":
    project_path = os.getcwd()
    print(f"\n🚀 Preview: {project_path}\n" + "=" * 50)
    list_files(project_path)
    print("=" * 50)